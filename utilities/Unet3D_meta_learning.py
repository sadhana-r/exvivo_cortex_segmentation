# -*- coding: utf-8 -*-
"""
3D U-net code in Pytorch
"""
import math
import numpy as np
from ops import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, meta_loss, meta_step_size, stop_gradient, dropout=0.3, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        #print("Down", self.meta_loss, self.meta_step_size, self.stop_gradient)

        self.first = first
        #print(self.first)
        #self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        #self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):

        #print("Down---forward", self.meta_loss, self.meta_step_size, self.stop_gradient)
        if not self.first:
            x = maxpool3D(x, kernel_size=2)

        #layer 1 conv, bn
        #print(">>>" ,x, self.conv1.weight, self.conv1.bias)
        x = conv3d(x, self.conv1.weight, self.conv1.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = conv3d(x, self.conv2.weight, self.conv2.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        y = self.bn2(y)
        y = relu(y)

        #droupout if required
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)

        #layer 3 conv, bn
        z = conv3d(y, self.conv3.weight, self.conv3.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        z = self.bn3(z)
        z = relu(z) #was not there

        #final relu
        #k = relu(x + z)
        
        return z

class ConvU(nn.Module):
    def __init__(self, planes, meta_loss, meta_step_size, stop_gradient, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient

        #print("Up", self.meta_loss, self.meta_step_size, self.stop_gradient)

        self.first = first
        #print(self.first)

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        #print("Up---forward", self.meta_loss, self.meta_step_size, self.stop_gradient)
        # final output is the localization layer

        #layer 1 conv, bn, relu
        if not self.first:
            x = conv3d(x, self.conv1.weight, self.conv1.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
            x = self.bn1(x)
            x = relu(x)

        #upsample, layer 2 conv, bn, relu
        y = upsample(x)
        y = conv3d(y, self.conv2.weight, self.conv2.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient, kernel_size=1, stride=1, padding=0)
        y = self.bn2(y)
        y = relu(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = conv3d(y, self.conv3.weight, self.conv3.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient)
        y = self.bn3(y)
        y = relu(y)

        return y


class Unet3D(nn.Module):
    def __init__(self, c=1, n=16, dropout=0.3, norm='gn', num_classes=2):
        super(Unet3D, self).__init__()
        meta_loss = None
        meta_step_size = 0.01
        stop_gradient = False

        self.meta_loss = meta_loss
        self.meta_step_size = meta_step_size
        self.stop_gradient = stop_gradient
        
        self.convd1 = ConvD(c,     n, meta_loss, meta_step_size, stop_gradient, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, meta_loss, meta_step_size, stop_gradient, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, meta_loss, meta_step_size, stop_gradient, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, meta_loss, meta_step_size, stop_gradient, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, meta_loss, meta_step_size, stop_gradient, dropout, norm)

        self.convu4 = ConvU(16*n, meta_loss, meta_step_size, stop_gradient, norm, first=True)
        self.convu3 = ConvU(8*n, meta_loss, meta_step_size, stop_gradient, norm)
        self.convu2 = ConvU(4*n, meta_loss, meta_step_size, stop_gradient, norm)
        self.convu1 = ConvU(2*n, meta_loss, meta_step_size, stop_gradient, norm)

        #self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        #self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)
        
        """
        y3 = conv3d(y3, self.seg3.weight, self.seg3.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient, kernel_size=None, stride=1, padding=0)
        y2 = conv3d(y2, self.seg2.weight, self.seg2.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient, kernel_size=None, stride=1, padding=0) + upsample(y3)
        """
        y1 = conv3d(y1, self.seg1.weight, self.seg1.bias, meta_loss=self.meta_loss, meta_step_size=self.meta_step_size, stop_gradient=self.stop_gradient, kernel_size=None, stride=1, padding=0) #+ upsample(y2)

        predictions = F.softmax(input=y1, dim=1) #softmax , dim=1
       # print(y1)
        return y1, predictions



def Net(is_meta_learning=False, meta_step_size=0.01, meta_loss = None, stop_gradient=False, num_classes = 2):
    model=Unet3D(c=1, num_classes=num_classes)
    return model

"""
from tensorboardX import SummaryWriter
model = Net()
input_image = torch.randn(1, 1, 64, 64, 64)
y = model(input_image)
print(y.shape)
writer = SummaryWriter('3Dunet-graph-metalearn')
writer.add_graph(model, input_image)
writer.close()
"""
