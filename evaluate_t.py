import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import numpy as np
import argparse
from tqdm import trange, tqdm
from utils.utils import Visdom

from dataset import pyramid_dataset_t
from models import BVF, PyFu
from utils.utils import generate_vimg, backwardwarping3D, flow_visualize
from MPVF.metric import Metrics

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # dataset information
    parser.add_argument('--test_folder', default='data/testing_4')
    parser.add_argument('--t', '-t', type=float)
    # training hyper-parameter
    parser.add_argument('--flow_model_name')
    parser.add_argument('--fusion_model_name')
    # device setting
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    # model hyper-parameter
    # visdom
    parser.add_argument('--port', required=True, type=int)
    
    return parser.parse_args()

class Model_factory():
    def __init__(self, args):
        self.args = args
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpus = [i for i in range(self.args.gpus)]
        
        # Data loader
        self.testDataset = pyramid_dataset_t.dataset(self.args.test_folder, self.args.t)
        self.test_loader = DataLoader(dataset=self.testDataset,
                                      batch_size=1,
                                      shuffle=True, 
                                      num_workers=self.args.num_workers)
        
        # Initialize Flow Model
        print(">>Initialize {} Model...".format(self.args.flow_model_name), end='')
        self.load_flow_model(device, gpus)
        # Initialize Model
        print(">>Initialize {} Model...".format(self.args.fusion_model_name), end='')
        self.model = PyFu.Model(input_feature=8*1, intermediate_features=[16, 32, 64], out_kernel_sizes=[5, 5, 5])
        self.model = DataParallel(self.model, device_ids=gpus).to(device)
        print("Done")
    
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
        with torch.no_grad():
            t_flow, t_flow1_2, t_flow1_4 = self.flow_model(self.ED, self.ES, self.t)
            zeros = self.set_input(torch.zeros([3]))
            z_flow, z_flow1_2, z_flow1_4 = self.flow_model(self.ED, self.ES, zeros)
            ones = self.set_input(torch.ones([3]))
            o_flow, o_flow1_2, o_flow1_4 = self.flow_model(self.ED, self.ES, ones)
            
        self.flowt0, self.flowt1 = torch.chunk(t_flow, 2, dim=1)
        _          , self.flow01 = torch.chunk(z_flow, 2, dim=1)
        self.flow10, _           = torch.chunk(o_flow, 2, dim=1)
        
        # flow approximation
        tt = self.t.view([-1, 1, 1, 1, 1])
        self.t0_01 = self.flow01 * (-tt)
        self.t1_01 = self.flow01 * (1-tt)
        self.t0_10 = self.flow10 * (tt)
        self.t1_10 = self.flow10 * (-(1-tt))
        
        # backward warping
        self.Et_D = backwardwarping3D(self.ED, self.flowt0)
        self.Et_S = backwardwarping3D(self.ES, self.flowt1)
        self.Et_D_01 = backwardwarping3D(self.ED, self.t0_01)
        self.Et_S_01 = backwardwarping3D(self.ES, self.t1_01)
        self.Et_D_10 = backwardwarping3D(self.ED, self.t0_10)
        self.Et_S_10 = backwardwarping3D(self.ES, self.t1_10)
        
        self.Et_D_2 = backwardwarping3D(self.ED1_2, F.interpolate(self.flowt0, scale_factor=0.5, mode='trilinear'))
        self.Et_S_2 = backwardwarping3D(self.ES1_2, F.interpolate(self.flowt1, scale_factor=0.5, mode='trilinear'))
        self.Et_D_01_2 = backwardwarping3D(self.ED1_2, F.interpolate(self.t0_01, scale_factor=0.5, mode='trilinear'))
        self.Et_S_01_2 = backwardwarping3D(self.ES1_2, F.interpolate(self.t1_01, scale_factor=0.5, mode='trilinear'))
        self.Et_D_10_2 = backwardwarping3D(self.ED1_2, F.interpolate(self.t0_10, scale_factor=0.5, mode='trilinear'))
        self.Et_S_10_2 = backwardwarping3D(self.ES1_2, F.interpolate(self.t1_10, scale_factor=0.5, mode='trilinear'))
        
        self.Et_D_4 = backwardwarping3D(self.ED1_4, F.interpolate(self.flowt0, scale_factor=0.25, mode='trilinear'))
        self.Et_S_4 = backwardwarping3D(self.ES1_4, F.interpolate(self.flowt1, scale_factor=0.25, mode='trilinear'))
        self.Et_D_01_4 = backwardwarping3D(self.ED1_4, F.interpolate(self.t0_01, scale_factor=0.25, mode='trilinear'))
        self.Et_S_01_4 = backwardwarping3D(self.ES1_4, F.interpolate(self.t1_01, scale_factor=0.25, mode='trilinear'))
        self.Et_D_10_4 = backwardwarping3D(self.ED1_4, F.interpolate(self.t0_10, scale_factor=0.25, mode='trilinear'))
        self.Et_S_10_4 = backwardwarping3D(self.ES1_4, F.interpolate(self.t1_10, scale_factor=0.25, mode='trilinear'))
        
        candidate0 = [self.ED, self.Et_D_01, self.Et_D, self.Et_D_10,
                      self.Et_S_10, self.Et_S, self.Et_S_01, self.ES]
        candidate1 = [self.ED1_2, self.Et_D_01_2, self.Et_D_2, self.Et_D_10_2,
                      self.Et_S_10_2, self.Et_S_2, self.Et_S_01_2, self.ES1_2]
        candidate2 = [self.ED1_4, self.Et_D_01_4, self.Et_D_4, self.Et_D_10_4,
                      self.Et_S_10_4, self.Et_S_4, self.Et_S_01_4, self.ES1_4]
        
        self.gen0, self.gen1, self.gen2 = self.model(candidate0, candidate1, candidate2, self.t)
        
    def test(self, e, vis):
        with torch.no_grad():
            self.model.eval()
            Et_metrics = Metrics()
            for ED, ES, Et, t in tqdm(self.test_loader, desc='Testing'):
                self.forward(ED, ES, Et, t)
         
                ED_ = self.get_numpy(self.ED)
                ES_ = self.get_numpy(self.ES)
                Et_ = self.get_numpy(self.Et)
                gen_ = self.get_numpy(self.gen0)
                
                Et_metrics.update(Et_, gen_)

        avg_Et_metrics = Et_metrics.average()
        print("Et Metric:", avg_Et_metrics)
        vis.MetricLine(metrics=avg_Et_metrics, epoch=e, name='Et')
        
        vis.Images(self.testDataset.unormalize, ED=ED_, ES=ES_, Et=Et_)
        vis.Images(self.testDataset.unormalize, gen=gen_)

    def load(self, ckpt):
        self.model.load_state_dict(ckpt['parameter'])
        
    def load_flow_model(self, device, gpus):
        self.flow_model = BVF.Model()
        self.flow_model = DataParallel(self.flow_model, device_ids=gpus).to(device)
        ckpt = torch.load('ckpts/{}.ckpt'.format(self.args.flow_model_name))
        self.flow_model.load_state_dict(ckpt['parameter'])
        for p in self.flow_model.parameters():
            p.requires_grad = False
        self.flow_model.eval()
        del ckpt
        print("Done")
    
def main():
    ### Main
    args = parse()
    # Visdom for real-time visualizing
    vis = Visdom(env=args.fusion_model_name, port=args.port)

    # Initialize model
    Model = Model_factory(args)

    print("Loading checkpoint...")
    ckpt = torch.load('ckpts/{}.ckpt'.format(args.fusion_model_name))
    Model.load(ckpt)
    del ckpt

    Model.test_loader = DataLoader(dataset=Model.testDataset,
                                   batch_size=1,
                                   shuffle=True, 
                                   num_workers=args.num_workers)

    Model.test(None, vis)

if __name__ == '__main__':
    main()