def Unormalize(x):
    x = x.clip(-1, 1)
    x = (x+1) / 2
    return x

import torch
import torch.nn.functional as F
def backwardwarping3D(x, flo, pad_mode='reflection'):
    '''
    x: [B, C, D, H, W] (volume)
    flo: [B, 3, D, H, W] (flow)
    '''
    B, C, D, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, W).expand(B, 1, D, H, W)
    yy = torch.arange(0, H).view(1, H, 1).expand(B, 1, D, H, W)
    zz = torch.arange(0, D).view(D, 1, 1).expand(B, 1, D, H, W)
    
    grid = torch.cat((xx, yy, zz), 1).type_as(x) # [1, 3, 10, 160, 160]

    vgrid = grid + flo 

#     # scale grid to [-1,1]
    vgrid[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :, :].clone() / max(D - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 4, 1)
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode=pad_mode)

    mask = torch.ones(x.shape).type_as(x)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask
    
import flow_vis
import numpy as np
def flow_visualize(flow):
    B, C, D, H, W = flow.shape
    flow = flow[0].permute([1, 2, 3, 0]) # [D, H, W, 3]
    flow_colors = torch.zeros([D-2, 3, H, W])
    for i, f in enumerate(flow[1:-1]):
        f_uv = f[:, :, :-1].cpu().detach().numpy()
        flow_color = flow_vis.flow_to_color(f_uv, convert_to_bgr=False)
        flow_colors[i] = torch.from_numpy(flow_color.transpose([2, 0, 1])) / 255.0
        
    return flow_colors

import pytorch_lightning as pl
from torchmetrics import MeanMetric
# from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import mean_squared_error
from skimage.metrics import structural_similarity as compare_ssim
from torch import nn

def structural_similarity_index_measure(pred, target, data_range):
    pred = pred[0, 0].cpu().detach().numpy()
    target = target[0, 0].cpu().detach().numpy()
    ssim = compare_ssim(pred, target, data_range=data_range)
    return ssim

class MeasureMetric(pl.LightningModule):
    def __init__(self, metrics=['PSNR', 'SSIM', 'MSE']):
        super().__init__()
        self.metrics = metrics
        for m in metrics:
            setattr(self, m, MeanMetric())

    def update(self, pred, target, unormalize=True):
        if unormalize:
            pred = Unormalize(pred)
            target = Unormalize(target)
            
        if 'SSIM' in self.metrics:
            ssim = structural_similarity_index_measure(pred, target, data_range=1.0)
            self.SSIM.update(ssim)
        if 'PSNR' in self.metrics:
            psnr = peak_signal_noise_ratio(pred, target, data_range=1.0)
            self.PSNR.update(psnr)
        if 'MSE' in self.metrics:
            mse = mean_squared_error(pred, target)
            self.MSE.update(mse)

    def compute(self):
        results = {}
        for m in self.metrics:
            meanmetric = getattr(self, m)
            results[m] = meanmetric.compute()

        return results
        
import torch.nn as nn
def conv1x1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=bias)
def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=bias)
def conv3x3x3_trans(in_planes, out_planes, stride=1):
    return nn.ConvTranspose3d(in_planes,
                              out_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              output_padding=1,
                              bias=False)
def conv7x7x7(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=padding,
                     bias=False)

class LayerNorm(nn.Module):
    def __init__(self, shape, dim, eps=1e-5):
        super().__init__()
    
        self.dim = dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = x.std(dim=self.dim, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1.)
    elif classname.find('BatchNorm3d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
    elif classname.find('PReLU') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)