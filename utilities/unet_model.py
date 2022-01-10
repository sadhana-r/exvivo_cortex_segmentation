# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:45:18 2020

@author: Sadhana
"""

import sys
sys.path.append('/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/exvivo_cortex_segmentation/crfasrnn')

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_blocks import DownConv, UpConv,conv3d, UpConv_MG
from torch.utils.tensorboard import SummaryWriter
from SOR import Successive_over_relaxation
from crfrnn import CrfRnn
    
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
        padding (bool): Whether to include zero padding in convolutional layers (SAME) or not (VALID).When padding
        isn't used, the output size is smaller than input size.
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """
    
    def __init__(self,  num_class, padding = False, in_channels = 1, init_feature_number = 16, num_levels = 5, norm = 'gn'):
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
                down_conv = DownConv(input_channels, output_channels, padding = padding, pooling = True, norm= norm) 
            else:
                down_conv = DownConv(input_channels, output_channels, padding = padding, pooling = False, norm = norm)                 
            encoder.append(down_conv)
            
        self.encoders = nn.ModuleList(encoder)
        
        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            up_conv = UpConv(input_channels, output_channels, padding = padding, up_mode='upsample', norm = norm)
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
#            print(x.shape)
            encoder_features.append(before_pool)
        
        # decoder part
        for i, decoder in enumerate(self.decoders):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
#            print(x.shape)
            
        x = self.final_conv(x)
        print(x.shape)
        predictions = F.softmax(input = x, dim = 1 )
        
        return x, predictions
    

class UNet_wDeepSupervision(nn.Module):
    
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
    
    def __init__(self,  num_class, patch_size, padding = False, in_channels = 1, init_feature_number = 16, num_levels = 5, norm = 'gn'):
        super(UNet_wDeepSupervision, self).__init__()
        
        encoder = []
        decoder = []
        deconv_levels = []
        output_channels = 0
        self.num_class = num_class
        
        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):            
            input_channels = in_channels if i ==0 else output_channels
            output_channels = init_feature_number * 2 ** i            
            if i < (num_levels - 1):
                down_conv = DownConv(input_channels, output_channels, padding = padding, pooling = True, norm= norm) 
            else:
                down_conv = DownConv(input_channels, output_channels,  padding = padding, pooling = False, norm = norm)                 
            encoder.append(down_conv)
            
        self.encoders = nn.ModuleList(encoder)
        
        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        
        if padding:
            output_size = patch_size
        else:
            output_size = (56,56,56)
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            # Define decoder
            up_conv = UpConv(input_channels, output_channels,  padding = padding, up_mode='upsample', norm = norm)
            decoder.append(up_conv)
            
            #Define additional deconvoltuion for deep supervision
            deconv = nn.Sequential(nn.Conv3d(output_channels,num_class, kernel_size = 3),
                                   nn.Upsample(mode='nearest', size = output_size))
            deconv_levels.append(deconv)
            
        self.decoders = nn.ModuleList(decoder)
        self.deconv = nn.ModuleList(deconv_levels)
        
        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number, num_class, kernel_size = 1)
        

    def forward(self, x):
        
#        patch_size = x.shape[2]
        # Encoder part
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            #print(x.shape)
            x, before_pool = encoder(x)
            encoder_features.append(before_pool)
#            print(x.shape)
        
        # decoder part
        deeps_outputs = []
        for i, (decoder,deep_output) in enumerate(zip(self.decoders,self.deconv)):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            #print(x.shape)
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
#            print(x.shape)
            
            ## For deep supervision, also upsample x and save output
            x_ds = deep_output(x)
#            print(x_ds.shape)
            deeps_outputs.append(x_ds)
            
        x = self.final_conv(x)
#        print(x.shape)
        predictions = F.softmax(input = x, dim = 1 )
        
        return x, predictions, deeps_outputs
    

class UNet_CRFRNN(nn.Module):
    """
    CRF-RNN netwprk with UNet backbone. CRF-RNN implementation based on this paper:
         
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
             
    https://github.com/sadeepj/crfasrnn_pytorch/
    
    """
    
    def __init__(self, num_class, patch_size, padding = False, in_channels = 1, init_feature_number = 16, num_levels = 5, norm = 'gn'):
        super(UNet_CRFRNN, self).__init__()
    
        self.unet = UNet_wDeepSupervision(num_class = num_class, patch_size = patch_size,in_channels =in_channels,
                                num_levels = num_levels,init_feature_number = init_feature_number, padding = False)
        self.crfrnn = CrfRnn(num_labels = 4, num_iterations = 5)

    def forward(self, x):
        
        img = x
        x, predictions, _ = self.unet(img)
        
        x = self.crfrnn(img, x)
        predictions = F.softmax(input = x, dim = 1 )
        
        return x, predictions
    

class UNet_DistanceRecon(nn.Module):
    
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 32
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """
    
    def __init__(self,  num_class, patch_size, padding = False, in_channels = 1, init_feature_number = 32, num_levels = 3, norm = 'gn'):
        super(UNet_DistanceRecon, self).__init__()
        
        encoder = []
        decoder = []
        deconv_levels = []
        output_channels = 0
        self.num_class = num_class
        
        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):            
            input_channels = in_channels if i ==0 else output_channels
            output_channels = init_feature_number * 2 ** i            
            if i < (num_levels - 1):
                down_conv = DownConv(input_channels, output_channels, padding = padding, pooling = True, norm= norm) 
            else:
                down_conv = DownConv(input_channels, output_channels,  padding = padding, pooling = False, norm = norm)                 
            encoder.append(down_conv)
            
        self.encoders = nn.ModuleList(encoder)
        
        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        
        if padding:
            output_size = patch_size
        else:
            output_size = (56,56,56)  # for input patch size of 96
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            # Define decoder
            up_conv = UpConv(input_channels, output_channels,  padding = padding, up_mode='upsample', norm = norm)
            decoder.append(up_conv)
            
            #Define additional deconvoltuion for deep supervision
            deconv = nn.Sequential(nn.Conv3d(output_channels,num_class, kernel_size = 3),
                                   nn.Upsample(mode='nearest', size = output_size))
            deconv_levels.append(deconv)
            
        self.decoders = nn.ModuleList(decoder)
        self.deconv = nn.ModuleList(deconv_levels)
        
        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number, num_class, kernel_size = 1)
        
        
        #distance_transformer
        self.distance_transformer = nn.Sequential(nn.Conv3d(num_class,init_feature_number, kernel_size = 3),
                                                  nn.ReLU(),
                                                  nn.Conv3d(init_feature_number,init_feature_number * 2, kernel_size = 3),
                                                  nn.ReLU(),
                                                  nn.ConvTranspose3d(init_feature_number*2,init_feature_number * 4, kernel_size = 3),
                                                  nn.ReLU(),
                                                  nn.ConvTranspose3d(init_feature_number*4,init_feature_number*2, kernel_size = 3),
                                                  nn.ReLU()
                                                  )
        self.final_conv_dmap = nn.Conv3d(init_feature_number*2, 1, kernel_size = 1)
        

    def forward(self, x):
        
#        patch_size = x.shape[2]
        # Encoder part
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            #print(x.shape)
            x, before_pool = encoder(x)
            encoder_features.append(before_pool)
#            print(x.shape)
        
        # decoder part
        deeps_outputs = []
        for i, (decoder,deep_output) in enumerate(zip(self.decoders,self.deconv)):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            #print(x.shape)
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
#            print(x.shape)
            
            ## For deep supervision, also upsample x and save output
            x_ds = deep_output(x)
#            print(x_ds.shape)
            deeps_outputs.append(x_ds)
            
        out_seg = self.final_conv(x)
#        print(x.shape)
        predictions_seg = F.softmax(input = out_seg, dim = 1 )
        
        #generate distance map
        dmap = self.distance_transformer(out_seg)
        dmap = self.final_conv_dmap(dmap)
        
        return out_seg, predictions_seg, deeps_outputs, dmap
    

class UNet_SOR(nn.Module):
    
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 32
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """
    
    def __init__(self,  num_class, patch_size, padding = False, in_channels = 1, init_feature_number = 32, num_levels = 3, norm = 'gn'):
        super(UNet_SOR, self).__init__()
        
        encoder = []
        decoder = []
        deconv_levels = []
        output_channels = 0
        self.num_class = num_class
        
        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):            
            input_channels = in_channels if i ==0 else output_channels
            output_channels = init_feature_number * 2 ** i            
            if i < (num_levels - 1):
                down_conv = DownConv(input_channels, output_channels, padding = padding, pooling = True, norm= norm) 
            else:
                down_conv = DownConv(input_channels, output_channels,  padding = padding, pooling = False, norm = norm)                 
            encoder.append(down_conv)
            
        self.encoders = nn.ModuleList(encoder)
        
        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        
        if padding:
            output_size = patch_size
        else:
            output_size = (56,56,56)  # for input patch size of 96
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            # Define decoder
            up_conv = UpConv(input_channels, output_channels,  padding = padding, up_mode='upsample', norm = norm)
            decoder.append(up_conv)
            
            #Define additional deconvoltuion for deep supervision
            deconv = nn.Sequential(nn.Conv3d(output_channels,num_class, kernel_size = 3),
                                   nn.Upsample(mode='nearest', size = output_size))
            deconv_levels.append(deconv)
            
        self.decoders = nn.ModuleList(decoder)
        self.deconv = nn.ModuleList(deconv_levels)
        
        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number, num_class, kernel_size = 1)
        
        self.compute_sor = Successive_over_relaxation()
                

    def forward(self, x):
        
#        patch_size = x.shape[2]
        # Encoder part
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            #print(x.shape)
            x, before_pool = encoder(x)
            encoder_features.append(before_pool)
#            print(x.shape)
        
        # decoder part
        deeps_outputs = []
        for i, (decoder,deep_output) in enumerate(zip(self.decoders,self.deconv)):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            #print(x.shape)
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
#            print(x.shape)
            
            ## For deep supervision, also upsample x and save output
            x_ds = deep_output(x)
#            print(x_ds.shape)
            deeps_outputs.append(x_ds)
            
        out_seg = self.final_conv(x)
        
        #Probability map
        prediction_prob = F.softmax(input = out_seg, dim = 1 )
        prediction_seg = torch.argmax(prediction_prob, 1)
        
        #generate distance map
        
        dmap,_,_ = self.compute_sor(prediction_seg.squeeze(), source_label = 2, sink_label = 3)
        dmap = torch.unsqueeze(dmap, 0)
        
        return out_seg, prediction_prob, deeps_outputs, dmap
    


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

    
#input_image =torch.randn(1,1, 96,96,96)
#net = UNet_CRFRNN(4, patch_size = (96,96,96), padding = False, num_levels = 3)
##net = UNet(4, padding = False, num_levels = 3)
##x,_,_, dmap = net(input_image)
#x, p = net(input_image)
#print(x.shape)