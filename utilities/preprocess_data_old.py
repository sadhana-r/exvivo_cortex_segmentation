#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:23:59 2020

@author: sadhana-ravikumar
"""

import numpy as np
import SimpleITK as sitk
import config
from torch.utils.data import Dataset
#from tps_sampler_3D import TPSRandomSampler3D
import torch
import pandas as pd
import patch_gen as p
import random
import nibabel as nib

c = config.Config_Unet()

#This function define standardization to image
def standardize_image(image_np):
    image_voxels = image_np[image_np>0] # Get rid of the background
    image_np = (image_np - np.mean(image_voxels)) / np.std(image_voxels)
    return image_np

def generate_patches(sample, is_training = True, num_pos = 100, num_neg = 100):
    
    img = sample['image']
    seg = sample['seg']
    
    if is_training:
        all_patches = False
        spacing = None
    else:
        all_patches = True
        spacing = c.test_patch_spacing
        # Pad test image to take care of boundary condition
        pad_size = c.half_patch[0]
        #Need to patch image and segmentation with 8 pixels
        img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode = "constant", constant_values = 0)
        seg = np.pad(seg, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode = "constant", constant_values = 0)
    
    img_std = standardize_image(img)
    img_norm = p.Normalize(img_std)    
    #Returns center points of patches - useful when reconstructing test image
    img_patches, seg_patches, cpts = p.get_patches(img_norm, seg, patch_size = c.segsize[0], num_pos = c.num_pos, num_neg = c.num_neg, all_patches = all_patches, spacing = spacing)
    
    return img_patches, seg_patches, cpts
    
    
class ImageDataset(Dataset):
    
    def __init__(self, csv_file, transform = None):
        
        self.image_list = pd.read_csv(csv_file, header = None)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_list.iloc[idx,0]))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(self.image_list.iloc[idx,1]))
        image_id = self.image_list.iloc[idx,2]
        
        #Alternative in nibabel
        image = nib.load(self.image_list.iloc[idx,0])
        affine = image.affine
        image = image.get_fdata().astype(np.float32)
        
        seg = nib.load(self.image_list.iloc[idx,1])
        seg = seg.get_fdata().astype(np.float32)
        
        sample = {'image':image, 'seg':seg, 'id':image_id, 'affine':affine}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
class PatchDataset(Dataset):
    
    def __init__(self, sample,is_training = True,transform = False ):
        
        self.sample = sample            
        
        self.transform = transform
        img_patches, seg_patches, cpts = generate_patches(sample, is_training = is_training, 
                                       num_pos = c.num_pos, num_neg = c.num_neg)
        
        if self.transform:
            
            #Apply the transformation to a random subset of patches
            rnd_idx = random.sample(range(0,len(img_patches)), k = 300)
            
#            img_aug = []
#            seg_aug = []
            for i in rnd_idx:
                img_elastic, seg_elastic = p.elastic_deformation(img_patches[i], seg_patches[i])
                img_rot, seg_rot = p.elastic_deformation(img_elastic, seg_elastic)
                
                img_patches[i] = img_rot
                seg_patches[i] = seg_rot
#                img_aug.extend(img_elastic)
#                seg_aug.extend(seg_elastic)
#                
#            img_patches.extend(img_aug)
#            seg_patches.extend(seg_aug)
            
        self.image_patches = img_patches
        self.seg_patches = seg_patches
        self.cpts = cpts
        
#        patches = generate_deepmedic_patches(self.sample, is_training = is_training, 
#                                       num_pos = c.num_pos, num_neg = c.num_neg, 
#                                       aug = c.aug, num_thread = c.num_thread)
#        # shuffle the patches
#        patch_index = list(range(len(patches)))
#            
#        self.image_patches = []
#        self.seg_patches = []
#            
#        for i in patch_index:
#            p = patches[i]
#            img = np.asarray(p[0])
#            seg = np.asarray(p[1])
#                
#            self.image_patches.append(img)
#            self.seg_patches.append(seg)
        
                     
    def __len__(self):
        return len(self.image_patches)
        
    def __getitem__(self, idx):
        
        sample = {'image':self.image_patches[idx], 'seg':self.seg_patches[idx], 'cpt':self.cpts[:,idx]}
        
        return sample
    
# Auxiliary function for loading image and segmentation
def loadImageSegPair(dirDiseaseTuple):
    imageDir, segDir, patient, disease = dirDiseaseTuple
    
    # Load the image and segmentation
    #imageDir = patientDir + "FLAIR/FLAIR_1x1x1.nii.gz"
    #segDir = patientDir + "FLAIR/ManSeg_1x1x1.nii.gz"
    
    # Read in image and segmentation
    image_np_orig = sitk.GetArrayFromImage(sitk.ReadImage(imageDir))
    seg_np_orig = sitk.GetArrayFromImage(sitk.ReadImage(segDir))
    
    return image_np_orig, seg_np_orig, patient, disease