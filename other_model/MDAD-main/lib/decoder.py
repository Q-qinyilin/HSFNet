import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from PIL import Image
#import cv2
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import scipy.misc   

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class DenseAttentionGate(nn.Module):
    def __init__(self, current_channel, previous_channels):
        super(DenseAttentionGate, self).__init__()
        total_previous_channels = sum(previous_channels)
        total_channels = current_channel + total_previous_channels
        self.conv1 = nn.Conv2d(total_channels, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
     

    def forward(self, current_fm, previous_fm):

        #up_previous_fm = []
       


        if len(previous_fm) <= 1:

            previous_fm = torch.tensor(previous_fm[0])

            temp_up = nn.Upsample(scale_factor= 2)
            up_previous_fm = temp_up(previous_fm)


            #up_previous_fm = torch.unsqueeze(previous_fm, dim=0)
        else:
            for i in range(0, len(previous_fm)):
               

                #temp_up = nn.Upsample(scale_factor= (i+1)*2)
                temp_up = nn.Upsample(scale_factor= 2**(i+1))
         
               
                temp_fm = torch.tensor(temp_up(previous_fm[i]))

         
                if i == 0:
                    up_previous_fm = temp_fm
                else:

                    up_previous_fm = torch.cat([up_previous_fm, temp_fm], dim=1)
                #up_previous_fm.append(temp_fm)
            
                #up_previous_fm = torch.tensor(up_previous_fm, dtype=torch.float32)
            


        

        concat_fm = torch.cat([current_fm, up_previous_fm], dim=1)
        concat_fm = self.conv1(concat_fm)
        attn = self.sigmoid(concat_fm)

        out_fm = attn * current_fm

        return out_fm
    

class DenseDecoding_block(nn.Module):
    def __init__(self, current_channel, previous_channels, current_decoding_channel, next_decoding_channel):
        super(DenseDecoding_block, self).__init__()
        self.dense_attngate = DenseAttentionGate(current_channel, previous_channels, )
        self.cfm = CFM(current_decoding_channel)
        self.conv = conv_block(current_decoding_channel+current_channel, next_decoding_channel )
        self.up = nn.Upsample(scale_factor= 2)

    def forward(self, current_fm, previous_fm, previous_decoding_fm ):
        dense_attn_fm = self.dense_attngate(current_fm, previous_fm)
    
        prev_decoder_fm = self.cfm(previous_decoding_fm)
        prev_decoder_fm = self.up(prev_decoder_fm)

        update_fm = torch.cat([dense_attn_fm, prev_decoder_fm], dim=1)
        update_fm = self.conv(update_fm)

        return update_fm
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)  
    

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        
        self.CA = ChannelAttention(channel)
        self.SA = SpatialAttention()

    def forward(self, x):
        x = self.CA(x) * x
        x = self.SA(x) * x 

        return x

class Decoding_layer1(nn.Module):
    """
    list: [64, 128, 320, 512]

    """
    def __init__(self, previous_layer_channels):
        super(Decoding_layer1, self).__init__()

        self.decoding_block_0 = DenseDecoding_block(previous_layer_channels[0], previous_layer_channels[1:], previous_layer_channels[1], previous_layer_channels[0])
        self.decoding_block_1 = DenseDecoding_block(previous_layer_channels[1], previous_layer_channels[2:], previous_layer_channels[2], previous_layer_channels[1])
        self.decoding_block_2 = DenseDecoding_block(previous_layer_channels[2], previous_layer_channels[3:], previous_layer_channels[3], previous_layer_channels[2])
        

    def forward(self, previous_layer_features):



        decoding_layer1_fm = []

        decoding_block2_fm = self.decoding_block_2(previous_layer_features[2], [previous_layer_features[3]], previous_layer_features[3])

        decoding_block1_fm = self.decoding_block_1(previous_layer_features[1], previous_layer_features[2:], decoding_block2_fm )
        decoding_block0_fm = self.decoding_block_0(previous_layer_features[0], previous_layer_features[1:], decoding_block1_fm)
       
        decoding_layer1_fm.append(decoding_block0_fm)
        decoding_layer1_fm.append(decoding_block1_fm)
        decoding_layer1_fm.append(decoding_block2_fm)

        return decoding_layer1_fm


class Decoding_layer2(nn.Module):
    """
    list: [64, 128, 320, 512]

    """
    def __init__(self, previous_layer_channels):
        super(Decoding_layer2, self).__init__()

        self.decoding_block_20 = DenseDecoding_block(previous_layer_channels[0], previous_layer_channels[1:], previous_layer_channels[1], previous_layer_channels[0])
        self.decoding_block_21 = DenseDecoding_block(previous_layer_channels[1], previous_layer_channels[2:], previous_layer_channels[2], previous_layer_channels[1])
       

    def forward(self, previous_layer_features):


        decoding_layer2_fm = []


        decoding_block21_fm = self.decoding_block_21(previous_layer_features[1], previous_layer_features[2:], previous_layer_features[2])

        decoding_block20_fm = self.decoding_block_20(previous_layer_features[0], previous_layer_features[1:], decoding_block21_fm)

       
        decoding_layer2_fm.append(decoding_block20_fm)
        decoding_layer2_fm.append(decoding_block21_fm)


        return decoding_layer2_fm

    
class Decoding_layer3(nn.Module):
    """
    list: [64, 128, 320, 512]

    """
    def __init__(self, previous_layer_channels):
        super(Decoding_layer3, self).__init__()

        self.decoding_block_30 = DenseDecoding_block(previous_layer_channels[0], previous_layer_channels[1:], previous_layer_channels[1], previous_layer_channels[0])
       

    def forward(self, previous_layer_features):


        decoding_layer3_fm = []

        decoding_block30_fm = self.decoding_block_30(previous_layer_features[0], previous_layer_features[1:],  previous_layer_features[1])      
        decoding_layer3_fm.append(decoding_block30_fm)



        return decoding_layer3_fm

