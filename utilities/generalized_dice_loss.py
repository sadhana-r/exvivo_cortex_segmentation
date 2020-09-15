#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:23:04 2020

@author: sadhana-ravikumar
"""

import numpy as np
from torch.autograd import Variable 



def generalized_dice_loss(y_prob, y_true):
    
   # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
   # Inputs are the predicted probabilities and the ground truth segmentation
   
   #calculate weights
   y_true = y_true.float()
   r_sum = y_true.sum(-1)
   eps = 1e-5
   class_weights = Variable(1./(r_sum*r_sum).clamp(min = eps), requires_grad = False)
#    Ncl = y_pred.shape[-1]
#    w = np.zeros((Ncl,))
#    for l in range(0,Ncl): w[l] = np.sum( np.asarray(y_true[:,:,:,:,l]==1,np.int8) )
#    w = 1/(w**2+0.00001)
#
#    # Compute gen dice coef:
#    numerator = y_true*y_pred
#    numerator = w*.sum(numerator,(0,1,2,3))
#    numerator = K.sum(numerator)
#    
#    denominator = y_true+y_pred
#    denominator = w*K.sum(denominator,(0,1,2,3))
#    denominator = K.sum(denominator)
#    
#    gen_dice_coef = numerator/denominator
#    
   return 0 #1-2*gen_dice_coef
#    