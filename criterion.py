__author__ = 'Titi'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Charbonnier_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        diff = torch.add(inputs, -targets)
        error = torch.sqrt(diff*diff + self.eps)
        loss = torch.mean(error)
        return loss
    
class Smoothness_loss(nn.Module):
    '''
    Smoothness loss
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        D_variance = torch.mean(torch.abs(inputs[:, :, :-1, :, :] - inputs[:, :, 1:, :, :]))
        H_variance = torch.mean(torch.abs(inputs[:, :, :, :-1, :] - inputs[:, :, :, 1:, :]))
        W_variance = torch.mean(torch.abs(inputs[:, :, :, :, :-1] - inputs[:, :, :, :, 1:]))
        loss = D_variance + H_variance + W_variance
        return loss