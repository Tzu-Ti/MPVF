__author__ = 'Titi'

'''
- Load 4D data from .nii.gz
- Resample data to (160, 160, 10)
'''

import nibabel as nib
from scipy.ndimage import zoom

import glob
import numpy as np
import os
from tqdm import trange
import shutil
import argparse

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--folder')
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    return parser.parse_args()

args = parse()
resample_size = [160, 160, 10]
if args.train:
    processed_folder = '{}_processed'.format(args.folder)

    for i in trange(1, 101):
        file_path = '{}/patient{:03d}/patient{:03d}_4d.nii.gz'.format(args.folder, i, i)
        data = nib.load(file_path).get_fdata()
        print("Origin data shape:", data.shape)

        for t in range(data.shape[-1]):
            data_3d = data[:, :, :, t]
            data_3d_shape = data_3d.shape

            resample_scale = np.divide(resample_size, data_3d_shape)
            resample_data = zoom(data_3d, resample_scale)

            save_folder = '{}/patient{:03d}'.format(processed_folder, i)
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

            save_path = os.path.join(save_folder, "{:02d}.npy".format(t+1))
            np.save(save_path, resample_data)

        # copy Info.cfg
        info_path = '{}/patient{:03d}/Info.cfg'.format(args.folder, i)
        new_infos_path = '{}/patient{:03d}/Info.cfg'.format(processed_folder, i)
        shutil.copyfile(info_path, new_infos_path)
        break
    
if args.test:
    processed_folder = '{}_4'.format(args.folder)
    for i in trange(101, 151):
        file_path = '{}/patient{:03d}/patient{:03d}_4d.nii.gz'.format(args.folder, i, i)
        data = nib.load(file_path).get_fdata()
        print("Origin data shape:", data.shape)
        
        info_path = '{}/patient{:03d}/Info.cfg'.format(args.folder, i)
        
        # get ED ES index from Info.cfg
        with open(info_path, 'r') as f:
            infos = [line for line in f.readlines()]
        ED_index = int(infos[0].split(':')[1].strip())
        ES_index = int(infos[1].split(':')[1].strip())
        
        if (ES_index - ED_index + 1) in [5, 9, 13, 17]:
            for t in range(data.shape[-1]):
                data_3d = data[:, :, :, t]
                data_3d_shape = data_3d.shape

                resample_scale = np.divide(resample_size, data_3d_shape)
                resample_data = zoom(data_3d, resample_scale)

                save_folder = '{}/patient{:03d}'.format(processed_folder, i)
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)

                save_path = os.path.join(save_folder, "{:02d}.npy".format(t+1))
                np.save(save_path, resample_data)
                
            # copy Info.cfg
            new_infos_path = '{}/patient{:03d}/Info.cfg'.format(processed_folder, i)
            shutil.copyfile(info_path, new_infos_path)