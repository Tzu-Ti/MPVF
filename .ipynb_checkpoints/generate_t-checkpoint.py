import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import numpy as np
import argparse
from tqdm import trange, tqdm
from utils.utils import Visdom

from models import BVF, PyFu
from utils.utils import generate_vimg, backwardwarping3D, flow_visualize
from utils.metric import Metrics

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import glob
import os
import numpy as np
import random
from scipy.ndimage import zoom

class dataset(Dataset):
    def __init__(self, folder, t=0.5):
        self.all_patients = glob.glob('{}/patient*'.format(folder))
        self.shape = (160, 160, 1)
        self.zero = np.zeros(self.shape)
        self.t = t

    def __getitem__(self, index):
        patient_path = self.all_patients[index]
        
        # get ED ES index from Info.cfg
        info_path = os.path.join(patient_path, 'Info.cfg')
        with open(info_path, 'r') as f:
            infos = [line for line in f.readlines()]
        ED_index = int(infos[0].split(':')[1].strip())
        ES_index = int(infos[1].split(':')[1].strip())
        
        img1_index = ED_index
        img2_index = ES_index
        
        # load ct
        img1 = self.load(patient_path, img1_index)
        img2 = self.load(patient_path, img2_index)
            
        # preprocess
        img1 = self.process(img1)
        img2 = self.process(img2)
        
        return img1, img2, self.t
        
    def __len__(self):
        return len(self.all_patients)
    
    def normalize(self, x):
        return (x - x.mean()) / (x.max() - x.min())
    
    def unormalize(self, x):
        return (x - x.min()) / (x.max() - x.min()) * 255.0
    
    def augmentation(self, x):
        return torch.rot90(x, 1, dims=(2, 3))
        
    def load(self, patient_path, index):
        path = '{}/{:02d}.npy'.format(patient_path, index)
        ct = np.load(path)
        
        return ct
        
    def process(self, ct):
        # pad data in z-axis by 0, reduce the border effects of 3D convolution
        ct = np.concatenate((self.zero, ct, self.zero), axis=2)
        
        # transpose to (D, H, W)
        ct = np.transpose(ct, (2, 0, 1))
        
        # add channel
        ct = np.expand_dims(ct, axis=0)
        
        # normalization
        ct = np.clip(ct, -200, 2000)
        ct = self.normalize(ct)
        
        return torch.from_numpy(ct)

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
        self.testDataset = dataset(self.args.test_folder, self.args.t)
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
        
    def forward(self, ED, ES, t):
        # multi-scale
        self.ED = self.set_input(ED)
        self.ES = self.set_input(ES)
        self.t = self.set_input(t)
        
        self.ED1_2 = F.interpolate(self.ED, scale_factor=0.5, mode='trilinear')
        self.ES1_2 = F.interpolate(self.ES, scale_factor=0.5, mode='trilinear')

        self.ED1_4 = F.interpolate(self.ED1_2, scale_factor=0.5, mode='trilinear')
        self.ES1_4 = F.interpolate(self.ES1_2, scale_factor=0.5, mode='trilinear')

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
            for ED, ES, t in self.test_loader:
                self.forward(ED, ES, t)
         
                ED_ = self.get_numpy(self.ED)
                ES_ = self.get_numpy(self.ES)
                gen_ = self.get_numpy(self.gen0)
                
                break
        
        vis.Images(self.testDataset.unormalize, ED=ED_, ES=ES_)
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
    torch.random.manual_seed(8)
    Model.test_loader = DataLoader(dataset=Model.testDataset,
                                   batch_size=1,
                                   shuffle=True, 
                                   num_workers=args.num_workers)

    Model.test(None, vis)

if __name__ == '__main__':
    main()