import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import glob
import os
import numpy as np
import random
from scipy.ndimage import zoom

class dataset(Dataset):
    def __init__(self, folder, t, train):
        self.all_patients = glob.glob('{}/patient*'.format(folder))
        self.shape = (160, 160, 1)
        self.zero = np.zeros(self.shape)
        self.t = t
        self.train = train

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
        
        if np.random.choice([True, False]) and (self.train):
            img1 = self.augmentation(img1)
            img2 = self.augmentation(img2)
            imgt = self.augmentation(imgt)
        
        return img1, img2, imgt, t
        
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
        
        return torch.from_numpy(ct).type(torch.float32)