# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:45:18 2020

@author: Sadhana
"""

import torch
import torch.nn as nn
from unet_blocks import DownConv, UpConv,conv3d, UpConv_MG
from torch.utils.tensorboard import SummaryWriter
    
class UNet(nn.Module):
    
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """
    
    def __init__(self,  num_class, in_channels = 1, init_feature_number = 16, num_levels = 5, norm = 'gn'):
        super(UNet, self).__init__()
        
        encoder = []
        decoder = []
        output_channels = 0
        
        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):            
            input_channels = in_channels if i ==0 else output_channels
            output_channels = init_feature_number * 2 ** i            
            if i < (num_levels - 1):
                down_conv = DownConv(input_channels, output_channels, pooling = True, norm= norm) 
            else:
                down_conv = DownConv(input_channels, output_channels, pooling = False, norm = norm)                 
            encoder.append(down_conv)
            
        self.encoders = nn.ModuleList(encoder)
        
        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            up_conv = UpConv(input_channels, output_channels, up_mode='upsample', norm = norm)
            decoder.append(up_conv)
            
        self.decoders = nn.ModuleList(decoder)
            
        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number, num_class, kernel_size = 1)

    def forward(self, x):
        
        # Encoder part
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            x, before_pool = encoder(x)
            encoder_features.append(before_pool)
        
        # decoder part
        for i, decoder in enumerate(self.decoders):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
        x = self.final_conv(x)
        
        return x
    

class MGNet(nn.Module):
    
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """
    
    def __init__(self,  num_class, num_groups = 4, in_channels = 1, init_feature_number = 16, num_levels = 5, norm = 'gn'):
        super(MGNet, self).__init__()
        
        encoder = []
        decoder = []
        output_channels = 0
        self.num_groups = num_groups
        
        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):    
            
            input_channels = in_channels if i ==0 else output_channels
            if i < (num_levels - 1):
                output_channels = num_groups* init_feature_number * 2 ** i     
                pooling = True
                ngroups = num_groups
            else:
                output_channels = input_channels ## only have one group in the final block  
                pooling = False
                ngroups = 1
                
            down_conv = DownConv(input_channels, output_channels, pooling = pooling, norm = norm, groups = ngroups)                 
            encoder.append(down_conv)
            
        self.encoders = nn.ModuleList(encoder)
        
        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        for i in range (num_levels - 1):
            input_channels = output_channels
            if i == 0:
                output_channels = input_channels 
            else:
                output_channels = input_channels // 2
            up_conv = UpConv_MG(input_channels, output_channels, groups = num_groups, up_mode='upsample', norm = norm)
            decoder.append(up_conv)
            
        self.decoders = nn.ModuleList(decoder)
            
        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number*num_groups, num_class*num_groups, kernel_size = 1, groups = num_groups)

    def forward(self, x):
        
        # Encoder part
#        print(x.shape)
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            x, before_pool = encoder(x)
    
            encoder_features.append(before_pool)
        
        # decoder part
        for i, decoder in enumerate(self.decoders):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
           
            
        x = self.final_conv(x)
        
        output_list = torch.chunk(x, self.num_groups, dim = 1)    
        
        return output_list

    
#input_image =torch.randn(1,1, 64,64,64)
#net = UNet(4)
#writer = SummaryWriter('graph')
#writer.add_graph(net, input_image)