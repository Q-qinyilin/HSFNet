import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from scipy import ndimage

from pvtv2 import pvt_v2_b2
from decoder import Decoding_layer1, Decoding_layer2, Decoding_layer3


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

    
class PVT_Dense_Decoder(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_Dense_Decoder, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        self.decoding_layer1 = Decoding_layer1([64, 128, 320, 512])
        self.decoding_layer2 = Decoding_layer2([64, 128, 320])
        self.decoding_layer3 = Decoding_layer3([64,128])
        self.out_head1 = nn.Conv2d(64, n_class, 1)
        self.out_head2 = nn.Conv2d(64, n_class, 1)
        self.out_head3 = nn.Conv2d(64, n_class, 1)



       
    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        d1, d2, d3 = self.decoding_layer1([x1, x2, x3, x4])
        out1 = self.out_head1(d1)
        out1  = F.interpolate(out1, scale_factor=4, mode='bilinear') 
        

        d11, d12= self.decoding_layer2([d1, d2, d3 ])
        out2 = self.out_head2(d11)
        out2  = F.interpolate(out2, scale_factor=4, mode='bilinear') 

  

        d21 = self.decoding_layer3([d11, d12])
        out3 = self.out_head3(d21[0])
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear') 
   
        return out1, out2, out3

 





if __name__ == '__main__':
    model = PVT_Dense_Decoder()
    input_tensor = torch.randn(4, 3, 352, 352)

    out1, out2, out3 = model(input_tensor)




    
