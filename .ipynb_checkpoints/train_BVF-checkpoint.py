import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim

import numpy as np
import argparse
import random
from tqdm import trange, tqdm
from utils.utils import Visdom

from dataset import pyramid_dataset
from models import BVF
from utils.criterion import *
from utils.utils import generate_vimg, backwardwarping3D, flow_visualize, weights_init
from utils.metric import Metrics

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # dataset information
    parser.add_argument('--train_folder', default='data/training_processed')
    parser.add_argument('--test_folder', default='data/testing_4')
    # training hyper-parameter
    parser.add_argument('--model_name', required=True)
    parser.add_argument('-b', '--batch_size', default=3, type=int)
    parser.add_argument('-l', '--lr', default=1e-4,  type=float)
    parser.add_argument('-e', '--epochs', default=1000, type=int)
    # device setting
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    # model hyper-parameter
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    # visdom
    parser.add_argument('--port', required=True, type=int)
    
    return parser.parse_args()

class Model_factory():
    def __init__(self, args):
        self.args = args
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpus = [i for i in range(self.args.gpus)]
        
        # Data loader
        trainDataset = pyramid_dataset.dataset(self.args.train_folder, 'random', mode='train')
        self.train_loader = DataLoader(dataset=trainDataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=True, 
                                       num_workers=self.args.num_workers)
        
        
        self.testDataset = pyramid_dataset.dataset(self.args.test_folder, 0.5, mode='test')
        self.test_loader = DataLoader(dataset=self.testDataset,
                                      batch_size=1,
                                      shuffle=True, 
                                      num_workers=self.args.num_workers)
        
        # Initialize Model
        print(">>Initialize {} Model...".format(self.args.model_name), end='')
        self.model = BVF.Model()
        self.model = DataParallel(self.model, device_ids=gpus).to(device)
        print("Done")
        
        # Training setting
        self.charbonnier = Charbonnier_loss().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
    
        self.set_input = lambda x: x.type(torch.cuda.FloatTensor)
        self.get_numpy = lambda x: x[0, 0, 1:-1, :, :].cpu().numpy()
        
    def forward(self, ED, ES, Et, t):
        # multi-scale
        self.ED = self.set_input(ED)
        self.ES = self.set_input(ES)
        self.Et = self.set_input(Et)
        self.t = self.set_input(t)
        
        self.ED1_2 = F.interpolate(self.ED, scale_factor=0.5, mode='trilinear')
        self.ES1_2 = F.interpolate(self.ES, scale_factor=0.5, mode='trilinear')
        self.Et1_2 = F.interpolate(self.Et, scale_factor=0.5, mode='trilinear')

        self.ED1_4 = F.interpolate(self.ED1_2, scale_factor=0.5, mode='trilinear')
        self.ES1_4 = F.interpolate(self.ES1_2, scale_factor=0.5, mode='trilinear')
        self.Et1_4 = F.interpolate(self.Et1_2, scale_factor=0.5, mode='trilinear')
        
        # predict multi-flow
        flow, flow1_2, flow1_4 = self.model(self.ED, self.ES, self.t)

        # backward warping
        self.flowt0, self.flowt1 = torch.chunk(flow, 2, dim=1)
        self.Et_D = backwardwarping3D(self.ED, self.flowt0)
        self.Et_S = backwardwarping3D(self.ES, self.flowt1)
        
        self.flowt0_2, self.flowt1_2 = torch.chunk(flow1_2, 2, dim=1)
        self.Et_D1_2 = backwardwarping3D(self.ED1_2, self.flowt0_2)
        self.Et_S1_2 = backwardwarping3D(self.ES1_2, self.flowt1_2)
        
        self.flowt0_4, self.flowt1_4 = torch.chunk(flow1_4, 2, dim=1)
        self.Et_D1_4 = backwardwarping3D(self.ED1_4, self.flowt0_4)
        self.Et_S1_4 = backwardwarping3D(self.ES1_4, self.flowt1_4)
        
        
    def train(self, e, vis):
        self.model.train()
        all_loss = []
        for idx, (ED, ES, Et, t) in enumerate(tqdm(self.train_loader, desc='Training')):
            self.forward(ED, ES, Et, t)

            recon_err_0 = self.charbonnier(self.Et_D   , self.Et) +\
                          self.charbonnier(self.Et_D1_2, self.Et1_2) +\
                          self.charbonnier(self.Et_D1_4, self.Et1_4)
            recon_err_1 = self.charbonnier(self.Et_S   , self.Et) +\
                          self.charbonnier(self.Et_S1_2, self.Et1_2) +\
                          self.charbonnier(self.Et_S1_4, self.Et1_4)

            loss = recon_err_0 + recon_err_1
            
            all_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        all_loss_mean = np.mean(all_loss)
        print("Loss: {:0.5f}".format(all_loss_mean))
        vis.Line(all_loss_mean=all_loss_mean, epoch=e)
        return all_loss_mean
    
    def test(self, e, vis):
        with torch.no_grad():
            self.model.eval()
            Et_D_metrics = Metrics()
            Et_S_metrics = Metrics()
            for ED, ES, Et, t in tqdm(self.test_loader, desc='Testing'):
                self.forward(ED, ES, Et, t)

                flow_colorst0 = flow_visualize(self.flowt0)
                flow_colorst1 = flow_visualize(self.flowt1)
                    
                ED_ = self.get_numpy(self.ED)
                ES_ = self.get_numpy(self.ES)
                Et_ = self.get_numpy(self.Et)
                Et_D_ = self.get_numpy(self.Et_D)
                Et_S_ = self.get_numpy(self.Et_S)
                
                Et_D_metrics.update(Et_, Et_D_)
                Et_S_metrics.update(Et_, Et_S_)

        avg_Et_D_metrics = Et_D_metrics.average()
        avg_Et_S_metrics = Et_S_metrics.average()
        print("Et_D Metric:", avg_Et_D_metrics)
        print("Et_S Metric:", avg_Et_S_metrics)
        vis.MetricLine(metrics=avg_Et_D_metrics, epoch=e, name='Et_D')
        vis.MetricLine(metrics=avg_Et_S_metrics, epoch=e, name='Et_S')
        
        vis.Images(self.testDataset.unormalize, ED=ED_, ES=ES_, Et=Et_)
        vis.Images(self.testDataset.unormalize, Et_D=Et_D_, Et_S=Et_S_)
        vis.Images(self.testDataset.unormalize, Flowt0=flow_colorst0, Flowt1=flow_colorst1)
        
        return (avg_Et_D_metrics['psnr'] + avg_Et_S_metrics['psnr']) / 2

    def save(self, e, best_loss, name):
        ckpt = {
            "parameter": self.model.state_dict(),
            "epoch": e,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(ckpt, 'ckpts/{}.ckpt'.format(name))
        
    def load(self, ckpt):
        self.model.load_state_dict(ckpt['parameter'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
def main():
    ### Main
    args = parse()
    # Visdom for real-time visualizing
    vis = Visdom(env=args.model_name, port=args.port)

    # Initialize model
    Model = Model_factory(args)
    
    start_epochs = 1
    best_psnr = 0

    if args.resume:
        print("Resume training...")
        ckpt = torch.load('ckpts/{}.ckpt'.format(args.model_name))
        start_epochs = ckpt['epoch'] + 1
        best_loss = ckpt['best_loss']
        Model.load(ckpt)

    if args.train:
        for e in range(start_epochs, args.epochs+1):
            print("Epoch: {}/{}".format(e, args.epochs))
            loss_value = Model.train(e, vis)

            # test every 5 epoch
            if e % 5 == 0:
                avg_psnr = Model.test(e, vis)

            # save checkpoint every 10 epoch
            if e % 10 == 0:
                print("Saving...")
                Model.save(e, best_psnr, name=args.model_name)
                
    elif args.test:
        print("Loading checkpoint...")
        ckpt = torch.load('ckpts/{}.ckpt'.format(args.model_name))
        Model.load(ckpt)
        
        Model.test_loader = DataLoader(dataset=Model.testDataset,
                                       batch_size=1,
                                       shuffle=False, 
                                       num_workers=args.num_workers)
        
        Model.test(None, vis)

if __name__ == '__main__':
    main()