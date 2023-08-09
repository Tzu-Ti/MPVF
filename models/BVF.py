import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import glob
import numpy as np

from tqdm import tqdm, trange

from .utils import *
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)
# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, vol_size: tuple, patch_size: tuple, num_classes, dim, depth, channels, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        vol_d, vol_h, vol_w = vol_size
        patch_d, patch_h, patch_w = patch_size
        
        assert vol_d % patch_d == 0 and vol_h % patch_h == 0 and vol_w % patch_w == 0,\
               'Volume dimensions must be divisible by the patch size.'
        
        num_patches = np.prod(vol_size) // np.prod(patch_size)
        patch_dim = channels * patch_d * patch_h * patch_w
        
        assert pool in {'cls', 'mean'},\
               'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)',
                      pd=patch_d, ph=patch_h, pw=patch_w),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, vol):
        x = self.to_patch_embedding(vol)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
class Decoder(nn.Module):
    def __init__(self,
                 decoder_dim=1024, decoder_depth=6, decoder_heads=8, decoder_dim_head=64):
        super().__init__()

        decoder_inplanes = [256, 128, 64, 32, 16]
        
        self.up1 = NetUpBlock_(decoder_inplanes[0], decoder_inplanes[1], decoder_inplanes[1], layers=2, kernel_sizes=2, strides=2)
        self.up0 = NetUpBlock_(decoder_inplanes[1], decoder_inplanes[2], decoder_inplanes[2], layers=2, kernel_sizes=2, strides=2)
        
        self.out2_0 = NetInBlock(decoder_inplanes[0], decoder_inplanes[1], 2)
        self.out2_1 = NetInBlock(decoder_inplanes[1], decoder_inplanes[2], 2)
        self.out2_2 = NetOutSingleBlock(decoder_inplanes[2], 3)
        # self.out2_2 = nn.Conv3d(decoder_inplanes[2], 3, kernel_size=1)
        
        self.out1_0 = NetInBlock(decoder_inplanes[1], decoder_inplanes[2], 2)
        self.out1_1 = NetInBlock(decoder_inplanes[2]+decoder_inplanes[2], decoder_inplanes[3], 2)
        self.out1_2 = NetOutSingleBlock(decoder_inplanes[3], 3)
        # self.out1_2 = nn.Conv3d(decoder_inplanes[3], 3, kernel_size=1)
        
        self.out0_0 = NetInBlock(decoder_inplanes[2], decoder_inplanes[3], 2)
        self.out0_1 = NetInBlock(decoder_inplanes[3]+decoder_inplanes[3], decoder_inplanes[4], 2)
        self.out0_2 = NetOutSingleBlock(decoder_inplanes[4], 3)
        # self.out0_2 = nn.Conv3d(decoder_inplanes[4], 3, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        
    def forward(self, C1_4, C1_2, C1_1):
        # up2 = self.to_1_4(resample, C1_4)
        up1 = self.up1(C1_4, C1_2)
        up0 = self.up0(up1, C1_1)
        
        out2 = self.out2_0(C1_4)
        out2 = self.out2_1(out2)
        flow2 = self.out2_2(out2)
        
        out1 = self.out1_0(up1)
        out1 = torch.cat([out1, self.upsample(out2)], dim=1)
        out1 = self.out1_1(out1)
        flow1 = self.out1_2(out1)
        
        out0 = self.out0_0(up0)
        out0 = torch.cat([out0, self.upsample(out1)], dim=1)
        out0 = self.out0_1(out0)
        flow0 = self.out0_2(out0)

        return flow2, flow1, flow0
    
class ContextEncoder(nn.Module):
    def __init__(self, encoder_inplanes=[32, 64, 128]):
        super().__init__()
        
        self.feature_extractor_1 = nn.ModuleList([
            NetInBlock(1, encoder_inplanes[0], 2),
            NetDownBlock(encoder_inplanes[0], encoder_inplanes[1], 2),
            NetDownBlock(encoder_inplanes[1], encoder_inplanes[2], 2)
        ])
        
        self.feature_extractor_2 = nn.ModuleList([
            NetInBlock(1, encoder_inplanes[0], 2),
            NetDownBlock(encoder_inplanes[0], encoder_inplanes[1], 2),
            NetDownBlock(encoder_inplanes[1], encoder_inplanes[2], 2)
        ])
        
    def forward(self, x1, x2):
        concated_feature = []
        for e1, e2 in zip(self.feature_extractor_1, self.feature_extractor_2):
            x1 = e1(x1)
            x2 = e2(x2)
            concated_feature.append(torch.cat([x1, x2], dim=1))
        
        return concated_feature
    
class TimeEncoding():
    def __init__(self, shape):
        self.shape = shape
        d_hid = C = shape[0]
        self.angle = [torch.pow(torch.tensor(10000), 2 * hid_j / d_hid) for hid_j in range(d_hid//2)]
        
    def encode(self, t):
        code = []
        for angle in self.angle:
            code.append(torch.sin(t * angle))
            code.append(torch.cos(t * angle))
        code = torch.cuda.FloatTensor(code)
        code = code.view(-1, 1, 1, 1).expand(self.shape)
        # code = code.view(-1, 1).expand(self.shape)
        return code
        
    def __call__(self, x, t):
        y = x.clone()

        for i, (b_x, b_t) in enumerate(zip(x, t)): 
            y[i] = b_x + self.encode(b_t)
        return y
    
class Model(nn.Module):
    def __init__(self, vol_size: tuple=(12, 160, 160), patch_size: tuple=(4, 8, 8), decoder_dim: int=512):
        super().__init__()
        
        self.ContextEncoder = ContextEncoder(encoder_inplanes=[32, 64, 128])
        
        self.Encoder = ViT(vol_size=vol_size,
                           patch_size=patch_size,
                           channels=2,
                           num_classes=1000,
                           dim=512, depth=3, heads=4, mlp_dim=1024)
        
        B, num_patches, encoder_dim = self.Encoder.pos_embedding.shape
        self.to_patch, self.patch_to_emb = self.Encoder.to_patch_embedding
        
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        self.concatenate = nn.Sequential(
            Transpose(1, 2),
            Rearrange('b f (d h w) -> b f d h w', d=vol_size[0]//patch_size[0], h=vol_size[1]//patch_size[1], w=vol_size[2]//patch_size[2]))
        
        self.resample = nn.Sequential(
            # nn.Conv3d(decoder_dim, 256, kernel_size=1, stride=1),
            nn.ConvTranspose3d(decoder_dim, 256, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(256),
            nn.PReLU(256)
        )

        # self.read_mlp = nn.Linear(1024, 512)

        # self.TE = TimeEncoding(shape=[num_patches-1, decoder_dim])
        self.TE = TimeEncoding(shape=[512, 3, 40, 40])

        self.to_C1_4 = NetInBlock(512, 256, 2)

        self.decodert0 = Decoder(decoder_dim=decoder_dim)
        self.decodert1 = Decoder(decoder_dim=decoder_dim)

    def forward(self, x1, x2, t):
        C1_1, C1_2, C1_4 = self.ContextEncoder(x1, x2)
        
        x12 = torch.cat([x1, x2], dim=1) # b, 2c, d, h, w
        patches = self.to_patch(x12) # b, n, pd*ph*pw*c
        batch, num_patches, _ = patches.shape

        tokens = self.patch_to_emb(patches)
        
        cls_tokens = repeat(self.Encoder.cls_token, '1 n d -> b n d', b = batch)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens += self.Encoder.pos_embedding[:, :(num_patches+1)]
        
        encoded_tokens = self.Encoder.transformer(tokens)
        
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        read = decoder_tokens[:, 1:] # ignore cls token
        
        ## add cls token
        # read = read + decoder_tokens[:, 0]

        ## projection cls token
        # class_token = decoder_tokens[:, 0].unsqueeze(1)
        # class_token = repeat(class_token, 'b 1 d -> b n d', n=num_patches)
        # read = torch.cat([read, class_token], dim=-1)
        # read = self.read_mlp(read)

        vol = self.concatenate(read)
        vol = self.resample(vol)

        C1_4 = torch.cat([C1_4, vol], dim=1)
        C1_4 = self.TE(C1_4, t) # add time encoding
        C1_4 = self.to_C1_4(C1_4)

        flowt0_4, flowt0_2, flowt0 = self.decodert0(C1_4, C1_2, C1_1)
        flowt1_4, flowt1_2, flowt1 = self.decodert1(C1_4, C1_2, C1_1)
        
        flow1_4 = torch.cat([flowt0_4, flowt1_4], dim=1)
        flow1_2 = torch.cat([flowt0_2, flowt1_2], dim=1)
        flow1_1 = torch.cat([flowt0  , flowt1  ], dim=1)
        
        
        return flow1_1, flow1_2, flow1_4
            
if __name__ == '__main__':
    model = Model().to('cuda')
    x1 = torch.randn([1, 1, 12, 160, 160]).to('cuda')
    x2 = torch.randn([1, 1, 12, 160, 160]).to('cuda')
    t = torch.randn([1]).to('cuda')
    output = model(x1, x2, t)