import torch
import torch.nn as nn
import torch.nn.functional as F

### Transformer
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
class ChannelAttention(nn.Module):
    '''
    param:
        channel: channel of input feature
    input:
        x: feature
    return:
        out: channel attention feature
    '''
    def __init__(self, channel, ratio=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel//ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel//ratio, channel, kernel_size=1, stride=1, padding=0),
        )
#         self.ff = nn.Sequential(
#             NetInBlock(channel, channel//ratio, 1, k_size=1, pad_size=0),
#             nn.ReLU(True),
#             NetInBlock(channel//ratio, channel, 1, k_size=1, pad_size=0)
#         )
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = max_out
        out = avg_out + max_out
#         out = self.ff(self.avg_pool(x))
        
        return self.act(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        
        return self.act(y)
class EncoderLayer(nn.Module):
    def __init__(self, C, D, H, W, k_size=3, pad_size=1):
        super().__init__()
        self.norm1 = LayerNorm([C, D, H, W], [1, 2, 3, 4])
        self.norm2 = LayerNorm([C, D, H, W], [1, 2, 3, 4])
        self.CA = ChannelAttention(C)
        self.SA = SpatialAttention()
        self.ff = nn.Sequential(
            NetInBlock(C, C, 1, k_size=k_size, pad_size=pad_size),
            nn.ReLU(True),
            NetInBlock(C, C, 1, k_size=k_size, pad_size=pad_size)
        )
        
    def forward(self, x):
        out = self.norm1(x)
        out = self.CA(out) * out
        out = self.SA(out) * out
        x = x + out
        out = self.norm2(x)
        y = self.ff(out)
        y = x + y
        
        return y
class ResBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.CA = ChannelAttention(C)
        
    def forward(self, x):
        out = self.CA(x) * x
        out = x + out
        
        return out
class ResGroup(nn.Module):
    def __init__(self, num_block, C, k_size=3, pad_size=1):
        super().__init__()
        body = [ResBlock(C) for _ in range(num_block)]
        body.append(NetInBlock(C, C, 1, k_size=k_size, pad_size=pad_size))
        body.append(nn.ReLU(True))
        body.append(NetInBlock(C, C, 1, k_size=k_size, pad_size=pad_size))
        self.body = nn.Sequential(*body)
        
    def forward(self, x):
        out = self.body(x)
        out = x + out
        return out
        
        
class TimeEncoding():
    def __init__(self, shape):
        self.shape = shape
        C, D, H, W = shape
        d_hid = C
        self.angle = [torch.pow(torch.tensor(10000), 2 * hid_j / d_hid) for hid_j in range(d_hid//2)]
        
    def encode(self, t):
        code = []
        for angle in self.angle:
            code.append(torch.sin(t * angle))
            code.append(torch.cos(t * angle))
        code = torch.cuda.FloatTensor(code)
        code = code.view(-1, 1, 1, 1).expand(self.shape)
        return code
        
    def __call__(self, x, t):
        y = x.clone()

        for i, (b_x, b_t) in enumerate(zip(x, t)): 
            y[i] = b_x + self.encode(b_t)
        return y
    
# SVIN package
class NetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, kernel_sz=3, pad=1):
        '''
        ConvBlock = consistent convs
        for each conv, conv(3x3) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(NetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sz, padding=pad))
        self.bns.append(nn.BatchNorm3d(out_channels))
        self.afs.append(nn.PReLU(out_channels))

        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d(out_channels, out_channels, kernel_size=kernel_sz, padding=pad))
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            out = self.afs[i](out)
        return out
    
class NetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetDownBlock, self).__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        down = self.af(down)
        out = self.convb(down)
        out = torch.add(out, down)
        return out
    
class NetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(NetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock(out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out
    
class NetUpBlock_(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers, kernel_sizes, strides):
        super(NetUpBlock_, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock(out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out
    
class NetJustUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers, kernel_sizes: tuple, strides: tuple):
        super(NetJustUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = self.convb(up)
        return out
    
class NetJustUpBlock_(nn.Module):
    def __init__(self, in_channels, out_channels, layers, kernel_sizes: tuple, strides: tuple):
        super(NetJustUpBlock_, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides)
        self.bn = nn.BatchNorm3d(out_channels)
        self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        up = self.up(x)
        up = self.bn(up)
        up = self.af(up)
        out = self.convb(up)
        return out
    
class NetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, k_size=3, pad_size=1):
        super(NetInBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.convb = NetConvBlock(in_channels, out_channels, layers=layers, kernel_sz=k_size, pad=pad_size)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        return out
    
class NetOutSingleBlock(nn.Module):
    def __init__(self, in_channels, classes):
        super(NetOutSingleBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm3d(classes)
        self.af_out = nn.PReLU(classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_out(out)
        out = self.af_out(out)
        return out