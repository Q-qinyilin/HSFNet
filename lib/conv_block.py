import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import mmcv
import sys
import os
from einops import rearrange
import numbers
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from lib.conv_ import ConvModule,PPM


from lib.deform_conv import DeformableConv2d as dcn2d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding =0, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        
        self.bn_acti = bn_acti
        
        self.conv = nn.Conv2d(nIn, nOut, kernel_size = kSize,
                              stride=stride, padding=padding,
                              dilation=dilation,groups=groups,bias=bias)
        
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)
            
    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output  
    
    
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        
        return output

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)    

from timm.models.layers import DropPath, to_2tuple
class ConvMlp(nn.Module):
    """ 使用 1x1 卷积保持空间维度的 MLP
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    


class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (dcn2d, nn.GELU, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class Hblock(nn.Module):
    def __init__(self, channels):
        super(Hblock, self).__init__()

        self.conv1 = nn.Conv2d(channels*3, channels+3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels*3)

        self.conv2 = nn.Conv2d(channels*2, channels*2, kernel_size=5, stride=1, padding=2, bias=True, groups = channels)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(channels*2)
        self.focal_level = 3
        self.act = nn.GELU()
        self.h = nn.Conv2d(3*channels, channels, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, input1, input2, input3):  # 128 44 44 / 128 44 44 /128  44 44
        B, C, H, W = input3.shape

        if input1.size()[2:] != input3.size()[2:]:
            input1 = F.interpolate(input1, size=input3.size()[2:], mode='bilinear')
        if input2.size()[2:] != input3.size()[2:]:
            input2 = F.interpolate(input2, size=input3.size()[2:], mode='bilinear')

        fuse = torch.cat((input1, input2, input3), 1) 
        fuse = self.act(self.conv1(self.bn1(fuse)))
        
        q, gates = torch.split(fuse, (C, 3), 1)  #   1 128 44 44             1 3 44 44

        # context aggreation
        ctx_all = 0 
        input1 = input1*(gates[:,0,:,:].unsqueeze(1)) # 1 1 44 44
        input2 = input2*(gates[:,1,:,:].unsqueeze(1))
        input3 = input3*(gates[:,2,:,:].unsqueeze(1))
        ctx_all = torch.cat([input1,input2,input3],dim=1)

        modulator = self.h(ctx_all)
        x_out = torch.cat([q,modulator],dim=1)
        
        fuse = self.act(self.conv2(self.bn2(x_out)))
        fuse = self.conv3(self.bn3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)


class SELayer(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                        #   dict(type='HSigmoid', bias=3.0, divisor=6.0),
                          dict(type='Sigmoid')
                          )
                          ):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            # out_channels=make_divisible(channels // ratio, 8),
            out_channels=channels // ratio,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            # in_channels=make_divisible(channels // ratio, 8),
            in_channels=channels // ratio,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class FrequencyExtract(nn.Module):
    def __init__(self, 
                in_channels,
                k_list=[2],
                # freq_list=[2, 3, 5, 7, 9, 11],
                fs_feat='feat',
                lp_type='freq_channel_att',
                act='sigmoid',
                channel_res=True,
                spatial='conv',
                spatial_group=1,
                compress_ratio=16,
                ):
        super().__init__()
        k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        self.channel_res = channel_res
        if spatial_group > 64: spatial_group=in_channels
        self.spatial_group = spatial_group
        if spatial == 'conv':
            self.freq_weight_conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=(len(k_list) + 1) * self.spatial_group, 
                                            stride=1,
                                            kernel_size=3, padding=1, bias=True) 
            # self.freq_weight_conv.weight.data.zero_()
            # self.freq_weight_conv.bias.data.zero_()   
        elif spatial == 'cbam': 
            self.freq_weight_conv = SpatialGate(out=len(k_list) + 1)
        else:
            raise NotImplementedError
        
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                nn.ReflectionPad2d(padding= k // 2),
                # nn.ZeroPad2d(padding= k // 2),
                nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
            ))
        elif self.lp_type == 'freq':
            pass
        elif self.lp_type in ('freq_channel_att', 'freq_channel_att_reduce_high'):
            # self.channel_att= nn.ModuleList()
            # for i in 
            self.channel_att_low = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            # self.channel_att_low[3].weight.data.zero_()
            # self.channel_att_low[3].bias.data.zero_()

            self.channel_att_high = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.in_channels // compress_ratio, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels // compress_ratio, self.in_channels, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            # self.channel_att_high[3].weight.data.zero_()
            # self.channel_att_high[3].bias.data.zero_()
            # self.channel_att.weight.data.zero_()
        elif self.lp_type in ('freq_eca', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = nn.ModuleList(
                [eca_layer(self.in_channels, k_size=9) for _ in range(len(k_list) + 1)]
            )
        elif self.lp_type in ('freq_channel_se', ):
            # self.channel_att_list = nn.ModuleList()
            # for i in 
            self.channel_att = SELayer(self.in_channels, ratio=16)
        else:
            raise NotImplementedError
        
        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        self.freq_thres=0.25 * 1.4

    def forward(self, x):
        pass
        return res

"""contrastive learning"""
from abc import ABC
class ProjectionHead(nn.Module):
    def __init__(self, dim=32):
        super(ProjectionHead, self).__init__()

        self.pro = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=(1,1))
        )

    def forward(self, x):
        proj = self.pro(x)
        return F.normalize(proj, p=2, dim=1)
