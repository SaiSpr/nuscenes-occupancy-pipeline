"""Small 3D U-Net with XY pooling; height is retained at every level."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a,b):
    return nn.Sequential(nn.Conv3d(a,b,3,padding=1),nn.GroupNorm(1,b),nn.GELU(),
                         nn.Conv3d(b,b,3,padding=1),nn.GroupNorm(1,b),nn.GELU())


class OccupancyUNet3D(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,base_channels=8,multi_level=False):
        super().__init__()
        c=base_channels
        self.encoders=nn.ModuleList([block(in_channels,c),block(c,2*c),block(2*c,4*c)])
        self.bottleneck=block(4*c,8*c)
        self.decoders=nn.ModuleList([block(12*c,4*c),block(6*c,2*c),block(3*c,c)])
        self.head=nn.Conv3d(c,out_channels,1)
        self.multi_level=multi_level
        self.aux=nn.ModuleList([nn.Conv3d(4*c,out_channels,1),nn.Conv3d(2*c,out_channels,1)]) if multi_level else nn.ModuleList()

    def forward(self,x):
        if min(x.shape[-2:])<8:raise ValueError('U-Net needs X and Y >= 8')
        skips=[]
        for enc in self.encoders:
            x=enc(x);skips.append(x);x=F.max_pool3d(x,(1,2,2))
        x=self.bottleneck(x);aux=[]
        for i,(dec,skip) in enumerate(zip(self.decoders,reversed(skips))):
            x=F.interpolate(x,size=skip.shape[-3:],mode='trilinear',align_corners=False)
            x=dec(torch.cat([x,skip],dim=1))
            if self.multi_level and i<2:aux.append(self.aux[i](x))
        main=self.head(x)
        return (main,aux) if self.multi_level else main
