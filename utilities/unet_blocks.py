# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:20:17 2020

@author: Sadhana
"""
import torch
from torch import nn as nn
from torch.nn import functional as F


def conv3d(input_channels, output_channels, kernel_size, padding, groups = 1, bias = True, stride = 1):
    return nn.Conv3d(input_channels, output_channels, kernel_size, padding = int(padding), stride = stride, bias = bias, groups = groups)
    
    
def upconv3d( input_channels, output_channels, mode, padding, groups = 1):
    
    if mode == 'transpose':
        return nn.ConvTranspose3d(input_channels, output_channels, groups = groups, kernel_size = 3, stride = (2,2,2), padding = int(padding), output_padding = 1)
        
    else:
        return nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2), nn.Conv3d(input_channels, output_channels, kernel_size = 1, groups = groups))
    
        #trilinear - align_corners = False
def normalization(input_channels,norm_type = 'gn'):
    
    if norm_type == 'bn':
        m = nn.BatchNorm3d(input_channels)
    elif norm_type == 'gn':
        m = nn.GroupNorm(1,input_channels)   
    return m

def interleaved_concat(f1, f2):
    f1_shape = list(f1.shape)
    f2_shape = list(f2.shape)
    c1 = f1_shape[1]
    c2 = f2_shape[1]
    
    f1_shape_new = f1_shape[:1] + [c1, 1] + f1_shape[2:]
    f2_shape_new = f2_shape[:1] + [c2, 1] + f2_shape[2:]

    f1_reshape = torch.reshape(f1, f1_shape_new)
    f2_reshape = torch.reshape(f2, f2_shape_new)
    output     = torch.cat((f1_reshape, f2_reshape), dim = 2)
    out_shape  = f1_shape[:1] + [c1 + c2] + f1_shape[2:]
    output     = torch.reshape(output, out_shape)
    return output 
                  
class DownConv(nn.Module):

#Encoder building block that performs 2 convolutions and 1 max pool
#ReLU activation follows each convolution 

    def __init__(self, input_channels, output_channels, padding, pooling=True, norm = 'gn', groups = 1):
        super(DownConv, self).__init__()

        group1 = 1 if (input_channels < 8) else groups
        
        self.conv1 = conv3d(input_channels, output_channels, padding = padding, groups = group1, kernel_size = 3)
        self.conv2 = conv3d(output_channels, output_channels,  padding = padding, groups = groups, kernel_size = 3)        
        self.norm = normalization(output_channels, norm)        
        self.pooling = pooling        
        self.dropout = nn.Dropout3d(p = 0.3)
        
        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        
        
        # try removing one dropout
        x = F.relu(self.norm(self.dropout(self.conv1(x))))
        x = F.relu(self.norm(self.conv2(x)))
        
        #take out
       #x = F.relu(self.norm(self.dropout(self.conv2(x))))

        before_pool = x
        
        if self.pooling:
            x = self.pool(x)
       
        return x, before_pool
 
 
class UpConv(nn.Module):     
#A helper Module that performs 2 convolutions and 1 UpConvolution.
#A ReLU activation follows each convolution.

    def __init__(self,input_channels, output_channels, padding, norm, up_mode):
        
        super(UpConv, self).__init__()
 
        self.upconv = upconv3d(input_channels, output_channels,  padding = padding, mode=up_mode)

        ## concatenation makes the input double again
        self.conv1 = conv3d(input_channels,output_channels,  padding = padding, kernel_size = 3)
        self.conv2 = conv3d(output_channels, output_channels, padding = padding, kernel_size = 3)
        self.dropout = nn.Dropout3d(p = 0.3)
        
        self.norm = normalization(output_channels, norm)
        
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width, layer_depth = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_z = (layer_width - target_size[2]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1]),  diff_z : (diff_z + target_size[2])
        ]

    def forward(self, x, from_encoder):
        
        #Up-sample
        x = self.upconv(x)
        
        #Concatenate - need to concatenate with cropped encoder output due to lack of padding (decrease in feature size)
        
        crop = self.center_crop(from_encoder,x.shape[2:])
        x = torch.cat([x, crop], 1)
        # Double convolution
        x = F.relu(self.norm(self.dropout(self.conv1(x))))
        x = F.relu(self.norm(self.conv2(x)))
        
        #take out
        #x = F.relu(self.norm(self.dropout(self.conv2(x))))
        return x
    
class UpConv_MG(nn.Module):     
#A helper Module that performs 2 convolutions and 1 UpConvolution.
#A ReLU activation follows each convolution.

    def __init__(self,input_channels, output_channels,  norm, up_mode, groups = 1):
        
        super(UpConv_MG, self).__init__()
 
        self.upconv = upconv3d(input_channels, output_channels, mode=up_mode, groups = groups)

        ## concatenation makes the input double again
        self.conv1 = conv3d(output_channels*2,output_channels, kernel_size = 3, groups = groups)
        self.conv2 = conv3d(output_channels, output_channels, kernel_size = 3, groups = groups)
        self.dropout = nn.Dropout3d(p = 0.3)
        
        self.norm = normalization(output_channels, norm)

    def forward(self, x, from_encoder):
        
        #Up-sample
        x = self.upconv(x)
        #Concatenate
        x = interleaved_concat(x, from_encoder)
        # Double convolution
        x = F.relu(self.norm(self.dropout(self.conv1(x))))
        x = F.relu(self.norm(self.conv2(x)))
        
        #take out
        #x = F.relu(self.norm(self.dropout(self.conv2(x))))
        
        return x
