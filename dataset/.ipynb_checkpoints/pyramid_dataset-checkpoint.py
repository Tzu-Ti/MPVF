import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
import torchvision

import glob
import os
import numpy as np
import random

class dataset(Dataset):
    def __init__(self, folder, t, train=True):
        self.all_patients = glob.glob('{}/patient*'.format(folder))
        self.shape = (160, 160, 1)
        self.zero = np.zeros(self.shape)
        self.t = t
        self.train = train
        self.max = 2000
        self.min = -200

    def __getitem__(self, index):
        patient_path = self.all_patients[index]
        
        # get ED ES index from Info.cfg
        info_path = os.path.join(patient_path, 'Info.cfg')
        with open(info_path, 'r') as f:
            infos = [line for line in f.readlines()]
        ED_index = int(infos[0].split(':')[1].strip())
        ES_index = int(infos[1].split(':')[1].strip())

        if self.t == 'random':
            img1_index = ED_index
            img2_index = ES_index
            imgt_index = random.choice([_ for _ in range(ED_index, ES_index+1)])
        elif self.t == 0.5:
            img1_index = ED_index
            img2_index = ES_index
            imgt_index = (ED_index + ES_index) // 2
        elif self.t == 0:
            img1_index = ED_index
            img2_index = ES_index
            imgt_index = ED_index
        elif self.t == 1:
            img1_index = ED_index
            img2_index = ES_index
            imgt_index = ES_index

        ranges = img2_index - img1_index
        t = (imgt_index-img1_index) / ranges
        
        # load ct
        img1 = self.load(patient_path, img1_index)
        img2 = self.load(patient_path, img2_index)
        imgt = self.load(patient_path, imgt_index)
            
        # preprocess
        img1 = self.process(img1)
        img2 = self.process(img2)
        imgt = self.process(imgt)
        
        # calc max and min of these three imgs
        # and normalize
        M, m = self.calcMm(img1, img2, imgt)
        img1 = self.normalize(img1, M, m)
        img2 = self.normalize(img2, M, m)
        imgt = self.normalize(imgt, M, m)
        
        if np.random.choice([True, False]) and (self.train):
            img1 = self.augmentation(img1)
            img2 = self.augmentation(img2)
            imgt = self.augmentation(imgt)
        
        return img1, img2, imgt, t
        
    def __len__(self):
        return len(self.all_patients)
    
    def calcMm(self, img1, img2, imgt):
        lst = [img1, img2, imgt]
        Ms = [img.max() for img in lst]
        ms = [img.min() for img in lst]
        
        return max(Ms), min(ms)
    
    def normalize(self, x, M, m):
        x = (x - m) / (M - m)
        # x = (x - self.min) / (self.max - self.min)
        # x = (x - x.min()) / (x.max() - x.min())
        x = (x - 0.5) / 0.5
        return x
    
    def augmentation(self, x):
        return torch.rot90(x, 1, dims=(2, 3))
        
    def load(self, patient_path, index):
        path = '{}/{:02d}.npy'.format(patient_path, index)
        img = np.load(path)
        
        return img
        
    def process(self, img):
        # pad data in z-axis by 0, reduce the border effects of 3D convolution
        img = np.concatenate((self.zero, img, self.zero), axis=2)
        
        # transpose to (D, H, W)
        img = np.transpose(img, (2, 0, 1))
        # add channel
        img = np.expand_dims(img, axis=0)
        # normalization
        img = np.clip(img, self.min, self.max)
        return img
        img = self.normalize(img)
        
        img = torch.from_numpy(img).type(torch.float32)
        return img

if __name__ == '__main__':
    D = dataset(folder="../data/training_processed", t='random', train=True)
    for img1, img2, imgt, t in D:
        print(img1.shape, img2.shape, imgt.shape)
        print(t)
        break