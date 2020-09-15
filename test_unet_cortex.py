#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:35:44 2020

@author: sadhana-ravikumar
"""
import sys
sys.path.append('./utilities')
import config_cortex as config
import torch
import preprocess_data as p
from unet_model import UNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os.path as osp
import os

c = config.Config_Unet()
dir_names = config.Config()

def Net(num_class = 4):
	net = UNet(num_class)
	return net

def computeGeneralizedDSC(gt, seg):
    
     gt_seg = gt[gt > 0]
     myseg = seg[gt > 0]
     
     gdsc = 100*(sum(gt_seg == myseg)/ len(gt_seg))
     
     return gdsc
      

def generate_prediction(output):    
    """
    Generates predictions based on the output of the network
    """    
    #convert output to probabilities
    probability = F.softmax(output, dim = 1)
    _, preds_tensor = torch.max(probability, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    
    return preds, probability


## Set up GPU if available    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model or not. If load model, the modelDir and tfboardDir should have existed. Otherwise they
# will be created forcefully, wiping off the old one.
load_model = True

#Set up directories
root_dir = dir_names.root_dir
experiment_name = "Experiment_14092020_2" #14072020 undersegments compared to Exp 3. 
tfboard_dir = dir_names.tfboard_dir + '/' + experiment_name
model_dir = dir_names.model_dir + '/' + experiment_name + '/'
test_dir = dir_names.test_dir + '/' + experiment_name + '/'

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

if not load_model:
    c.force_create(model_dir)
    c.force_create(tfboard_dir)
    
#Define image dataset (reads in full images and segmentations)
test_dataset = p.ImageDataset(csv_file = c.final_test_csv)

num_class = 4
model_file = model_dir + 'model_5.pth'
net = Net(num_class)
net.load_state_dict(torch.load(model_file, map_location=device))
net.eval()
net.to(device)

pad_size = c.half_patch[0]
        
with torch.no_grad():
    for i in range(0,len(test_dataset)):
        
        sample = test_dataset[i]
        test_patches = p.GeneratePatches(sample, is_training = False, transform =False)
        
        testloader = DataLoader(test_patches, batch_size = c.batch_size, shuffle = False, num_workers = c.num_thread)    
        image_id = sample['id']
        print("Generating test patches for ", image_id )
        
        image_shape = sample['image'].shape
        affine = sample['affine']
        
        ## For assembling image
        im_shape_pad = [x + pad_size*2 for x in image_shape]
        prob = np.zeros([num_class] + list(im_shape_pad))
        rep = np.zeros([num_class] + list(im_shape_pad))
        
        pred_list = []
        for j, patch_batched in enumerate(testloader):
            
                print("batch", j)                
                img = patch_batched['image'][:,None,...].to(device)
                seg = patch_batched['seg'].to(device)
                cpts = patch_batched['cpt']
                
                output = net(img)
                probability = F.softmax(output, dim = 1).cpu().numpy()
                
                #Crop the patch to only use the center part
                #probability = probability[:,:,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size]
                                
                ## Assemble image in loop!
                n, C, hp, wp, dp = probability.shape
                half_shape = torch.tensor([hp, wp,dp])/2
#                half_shape = half_shape.astype(int)
                hs, ws, ds = half_shape
                
                for cpt, pred in zip(list(cpts), list(probability)):
                    #if np.sum(pred)/hs/ws/ds < 0.1:
                    prob[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred
                    rep[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += 1
                                
                    
#                pred_list.append((probability, cpts))
                
         #Crop the image since we added padding when generating patches
        prob = prob[:,pad_size:-pad_size, pad_size:-pad_size,pad_size:-pad_size]
        rep = rep[:,pad_size:-pad_size,pad_size:-pad_size,pad_size:-pad_size]
        rep[rep==0] = 1e-6
    
        # Normalized by repetition
        prob = prob/rep
    
        seg_pred = np.argmax(prob, axis = 0).astype('float')
        prob = np.moveaxis(prob,0,-1)
        rep = np.moveaxis(rep,0,-1)
        rep = rep[:,:,:,1].squeeze()
        
        
        
#        gdsc = computeGeneralizedDSC(sample['seg'], seg_pred)
#        print("Prediction accuracy", gdsc)
        
        nib.save(nib.Nifti1Image(prob, affine), osp.join(test_dir, "prob_" + str(image_id) + ".nii.gz"))
        nib.save(nib.Nifti1Image(seg_pred, affine), osp.join(test_dir, "seg_" + str(image_id) + ".nii.gz" ))
        nib.save(nib.Nifti1Image(rep, affine), osp.join(test_dir, "rep_" + str(image_id) + ".nii.gz"))
        
        print("Done!")
        
        
        
        
                
                
                
                
            
            
            
            


