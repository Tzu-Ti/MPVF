import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

class DFnet(nn.Module):
    def __init__(self, input_feature, intermediate_features, out_kernel_sizes):
        super().__init__()
        
        self.in_block = NetInBlock(input_feature, intermediate_features[0], 2)
        self.down_block1 = NetDownBlock(intermediate_features[0], intermediate_features[1], 2)
        self.down_block2 = NetDownBlock(intermediate_features[1], intermediate_features[2], 2)
        
        self.TE0 = TimeEncoding(shape=[16, 12, 160, 160])
        self.TE1 = TimeEncoding(shape=[32, 6, 80, 80])
        self.TE2 = TimeEncoding(shape=[64, 3, 40, 40])
        
        # Transformer on multi-scale
        transformer0 = [EncoderLayer(16, 12, 160, 160, k_size=3, pad_size=1) for _ in range(3)]
        self.transformer0 = nn.Sequential(*transformer0)
        transformer1 = [EncoderLayer(32, 6, 80, 80, k_size=3, pad_size=1) for _ in range(3)]
        self.transformer1 = nn.Sequential(*transformer1)
        transformer2 = [EncoderLayer(64, 3, 40, 40, k_size=3, pad_size=1) for _ in range(3)]
        self.transformer2 = nn.Sequential(*transformer2)

#         resgroup0 = [ResGroup(num_block=6, C=16) for _ in range(3)]
#         self.resgroup0 = nn.Sequential(*resgroup0)
#         resgroup1 = [ResGroup(num_block=6, C=32) for _ in range(3)]
#         self.resgroup1 = nn.Sequential(*resgroup1)
#         resgroup2 = [ResGroup(num_block=6, C=64) for _ in range(3)]
#         self.resgroup2 = nn.Sequential(*resgroup2)
        
        self.tail2_1 = NetInBlock(intermediate_features[2], 128, 1)
        self.tail2_2 = NetInBlock(128, 256, 1)
        self.tail2_3 = NetOutSingleBlock(256, out_kernel_sizes[2] ** 3 * 6)
        self.up_block2 = NetUpBlock(intermediate_features[2], intermediate_features[1], 80, 2)
        
        self.tail1_1 = NetInBlock(80, 128, 1)
        self.tail1_2 = NetInBlock(128+256, 256, 1)
        self.tail1_3 = NetOutSingleBlock(256, out_kernel_sizes[1] ** 3 * 6)
        self.up_block1 = NetUpBlock(80, intermediate_features[0], 80, 2)
        
        self.tail0_1 = NetInBlock(80, 128, 1)
        self.tail0_2 = NetInBlock(128+256, 256, 1)
        self.tail0_3 = NetOutSingleBlock(256, out_kernel_sizes[0] ** 3 * 6)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        
        
    def forward(self, candidate, t):
        combined = torch.cat(candidate, dim=1)
        c0 = self.in_block(combined)
        c1 = self.down_block1(c0)
        c2 = self.down_block2(c1)
        
#         add time encoding
        t0 = self.TE0(c0, t)
        t1 = self.TE1(c1, t)
        t2 = self.TE2(c2, t)

        # Transformer on multi-scale
        t0 = self.transformer0(t0) # [-1, 32, 12, 160, 160]
        t1 = self.transformer1(t1) # [-1, 64, 6, 80, 80]
        t2 = self.transformer2(t2) # [-1, 128, 3, 40, 40]

#         # test no CAT no t, 20220613
#         t0 = c0
#         t1 = c1
#         t2 = c2

#         # CAIN resgroup on multi-scale
#         t0 = self.resgroup0(c0) # [-1, 32, 12, 160, 160]
#         t1 = self.resgroup1(c1) # [-1, 64, 6, 80, 80]
#         t2 = self.resgroup2(c2) # [-1, 128, 3, 40, 40]
        
        out2 = self.tail2_1(t2)
        out2 = self.tail2_2(out2)
        kernel2 = self.tail2_3(out2)
        br1 = self.up_block2(t2, t1)
        
        out1 = self.tail1_1(br1)
        out1 = torch.cat([out1, self.upsample(out2)], dim=1)
        out1 = self.tail1_2(out1)
        kernel1 = self.tail1_3(out1)
        br0 = self.up_block1(br1, t0)
        
        out0 = self.tail0_1(br0)
        out0 = torch.cat([out0, self.upsample(out1)], dim=1)
        out0 = self.tail0_2(out0)
        kernel0 = self.tail0_3(out0)

        return [kernel0, kernel1, kernel2]
    
class DynamicFilter(nn.Module):
    def __init__(self, out_kernel_size):
        super().__init__()
        
        self.padding = out_kernel_size // 2
        
        self.filter_localexpand = nn.Parameter(torch.reshape(
            torch.eye(out_kernel_size**3),
            (out_kernel_size**3, 1, out_kernel_size, out_kernel_size, out_kernel_size)
        ), requires_grad=False)
        
    def forward(self, candidate, filters):
        combined = torch.cat(candidate, dim=1)
        x_localexpand = []
        for c in range(combined.size(1)):
            x_localexpand.append(F.conv3d(combined[:, c:c+1, :, :, :], self.filter_localexpand, padding=self.padding))
            
        x_localexpand = torch.cat(x_localexpand, dim=1)
        x = torch.sum(torch.mul(x_localexpand, filters), dim=1, keepdim=True)

        return x
    
class Model(nn.Module):
    def __init__(self, input_feature, intermediate_features, out_kernel_sizes):
        super().__init__()
        
        self.dfnet = DFnet(input_feature, intermediate_features, out_kernel_sizes=out_kernel_sizes)
        self.filtering0 = DynamicFilter(out_kernel_size=out_kernel_sizes[0])
        self.filtering1 = DynamicFilter(out_kernel_size=out_kernel_sizes[1])
        self.filtering2 = DynamicFilter(out_kernel_size=out_kernel_sizes[2])
        self.act = nn.Softmax(dim=1)
        
    def forward(self, candidate0, candidate1, candidate2, t):
        filters = self.dfnet(candidate0, t)
        gen0 = self.filtering0(candidate0[1:-1], self.act(filters[0]))
        gen1 = self.filtering1(candidate1[1:-1], self.act(filters[1]))
        gen2 = self.filtering2(candidate2[1:-1], self.act(filters[2]))
        
        return [gen0, gen1, gen2]