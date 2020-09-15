#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:23:59 2020

@author: sadhana-ravikumar
"""

import numpy as np
import SimpleITK as sitk
import config_cortex as config
#import config_srlm as config
from torch.utils.data import Dataset
import torch
import pandas as pd
import patch_gen as p
import random
import nibabel as nib
import csv
import os.path as osp
import resample_util
import torchvision.transforms as transforms
from numpy import random as rnd

c = config.Config_Unet()
dir_names = config.Config()

#Writes a patch (imaeg and seg) to file. Saves control point in csv
def write_patch_to_file(image_patch_list, label_patch_list, sample, cpt_list, prior = None):
    
    print("Writing patches to file")
    print(cpt_list.shape[1])
    for idx in range(cpt_list.shape[1]):
        
        image = image_patch_list[idx]
        seg = label_patch_list[idx]
        cpt = cpt_list[:,idx]

        img_id = sample['id']
        
        # If there is a prior, then concatenate with image patch as second channel
        if prior is not None:
            print("Extract corresponding patch from prior")
            patch_size = image.shape[0]
            idx1_sg = cpt[0] - int(patch_size/2)
            idx1_cr = cpt[1] - int(patch_size/2)
            idx1_ax = cpt[2] - int(patch_size/2)

            prior_patch = prior[idx1_sg:idx1_sg + patch_size, idx1_cr:idx1_cr+patch_size, idx1_ax: idx1_ax + patch_size]            
            #add prior to image as a second channel
            image = np.stack((image, prior_patch), axis = -1)

        if (sample['type'] == 'train'):
            img_filename = osp.join(dir_names.patch_dir, "training_data","img_"+ str(img_id) + '_' +str(idx) + ".nii.gz")
            seg_filename = osp.join(dir_names.patch_dir, "training_data" ,"seg_"+ str(img_id) +  '_' +str(idx) + ".nii.gz" )
            csv_file = dir_names.train_patch_csv
        else:
            img_filename = osp.join(dir_names.patch_dir, "validation_data","img_"+ str(img_id) + '_' +str(idx) + ".nii.gz")
            seg_filename = osp.join(dir_names.patch_dir, "validation_data" ,"seg_" + str(img_id) + '_' +str(idx) + ".nii.gz" )
            csv_file = dir_names.val_patch_csv
        
        # standardize patch before writing to file
        image = p.standardize_image(image) 
        
        nib.save(nib.Nifti1Image(image, sample['affine']), img_filename)
        nib.save(nib.Nifti1Image(seg, sample['affine']), seg_filename)

        #save filenames and corresponding control point in csv
        data_list = (img_filename,seg_filename, cpt)
        with open(csv_file, 'a') as f:
            wr = csv.writer(f)
            wr.writerow(data_list)

# This function converts center points to multiple resolution
def multiple_resolution_cpts(cpts, patchsize_multi_res):
    cpts_multi_res = []
    for pr in patchsize_multi_res:
        res, _ = pr
        cpts_res = (cpts * res).astype(np.int32)
        cpts_multi_res.append(cpts_res)
    return cpts_multi_res

# This function crops patches around center points with patch size
# input:
#   image: input image, [H, W, D, C]
#   seg: segmentation, [H, W, D]
#   cpts: center points, [3, N]
#   patch_size: tuple, (HP, WP, DP)
# output:
#   patches: list of (img_patch, seg_patch, cpt)
def crop_patch_by_cpts(image, label, cpts, patch_size):
    image_patch_list = []
    label_patch_list = []
    patch_width = int(patch_size/2)
    for i in range(cpts.shape[1]):
        idx1_sg = cpts[0][i] - patch_width
        idx1_cr = cpts[1][i] - patch_width
        idx1_ax = cpts[2][i] - patch_width

        idx2_sg = idx1_sg + patch_size
        idx2_cr = idx1_cr + patch_size
        idx2_ax = idx1_ax + patch_size

        image_patch = image[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
        label_patch = label[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]

#        print(image_patch.shape)
#        if sum(image_patch.shape) != 192:
#            print("error patch")
        
        """
        remove the standardization to try Experiment_14072020 cortex
        """        
        image_patch_list.append(image_patch)
        label_patch_list.append(label_patch)

    return image_patch_list, label_patch_list


# This function sample 3D patches from 3D volume in multiple resolution around same center, used deepmedic or 
# similar style network
# input:
#   image: the image in numpy array, dimension [H, W, D, C] 
#   label: segmentation of the image, dimension [H, W, D], right now assuming this is binary    
#   patchsize_multi_res: this is the patch size in multi-resolution [(1, (25, 25, 25)), (0.33, (19, 19, 19))]
#                   this means it will sample patch size (25, 25, 25) in resolution 1x, patch size (19, 19, 19) in resolution 0.33x etc    
#   num_pos: number of positive patches that contains lesion to sample. If there is no enough patches to sample,
#            it will return all indexes that contains lesion
#   num_negative: number of negative background patches that doesn't contain lesion to sample.
def get_multi_resolution_patches(image, label, patchsize_multi_res, num_pos = 100, num_neg = 100, all_patches=False, spacing=0):
# Sample center points
    if not all_patches:

        min_res = 1    
        patch_size = 0
            # When choosing control points, must be within the largest patch size ( technically it should be the patch size at full resolution)
        for pr in patchsize_multi_res:
            res, patch_size_array = pr
            if patch_size_array[0] > patch_size:
                    patch_size = patch_size_array[0]
            if min_res > res:
                    min_res = res

        cpts_pos_sampled, cpts_neg_sampled = p.sample_center_points(label, num_pos, num_neg, patch_size, res = min_res)
    
        # Get center pts in multi resolution
        cpts_pos_multi_res = multiple_resolution_cpts(cpts_pos_sampled, patchsize_multi_res)
        cpts_neg_multi_res = multiple_resolution_cpts(cpts_neg_sampled, patchsize_multi_res)

        #patches_pos_multi_res = []
        #patches_neg_multi_res = []
        image_patches_multires = []
        label_patches_multires = []
        cpts_multires = []
        for idx, pr in enumerate(patchsize_multi_res):

            res, patch_size_array = pr

            # Downsample the image and segmentation
            image_resize, seg_resize = resample_util.resample_by_resolution(image, label, res)

            cpts_max = np.array(image_resize.shape[:3]) - 1
            cpts_max = cpts_max[:, None]

            # Fetch positive patches
            cpts_pos = cpts_pos_multi_res[idx]
            cpts_pos = np.minimum(cpts_max, cpts_pos) # Limit the range
            # Due to numerical rounding the cpts in different resolution may not match the 
            # resize image exactly. So need to hard constraint it
            
            image_patch_list, label_patch_list = crop_patch_by_cpts(image_resize, seg_resize, cpts_pos, patch_size_array[0])
#                    patches_pos_multi_res.append([image_patch_list, label_patch_list, res])

            image_patches_multires.extend(image_patch_list)
            label_patches_multires.extend(label_patch_list)
            
            # Fetch negative patches
            cpts_neg = cpts_neg_multi_res[idx]
            cpts_neg = np.minimum(cpts_max, cpts_neg) # Limit the range.
            image_patch_list, label_patch_list = crop_patch_by_cpts(image_resize, seg_resize, cpts_neg, patch_size_array[0])
#                    patches_neg_multi_res.append([image_patch_list, label_patch_list, res])
            
            image_patches_multires.extend(image_patch_list)
            label_patches_multires.extend(label_patch_list)
            
            cpts_multires.append(np.concatenate([cpts_pos,cpts_neg],1))
        
        cpts_multires = np.hstack(cpts_multires)
        
        return image_patches_multires, label_patches_multires,cpts_multires

    else:
            # Regularly grid center points
            cpts = p.grid_center_points(image.shape, spacing)
            cpts_multi_res = multiple_resolution_cpts(cpts, patchsize_multi_res)
            patches_multi_res = []

            for idx, pr in enumerate(patchsize_multi_res):
                res, patch_size = pr
                # Downsample the image and segmentation
                image_resize, seg_resize = resample_util.resample_by_resolution(image, label, res)
                
                # Fetch patches
                cpts_res = cpts_multi_res[idx]
                image_patch_list, label_patch_list = crop_patch_by_cpts(image_resize, seg_resize, cpts_res, patch_size)
                patches_multi_res.append([image_patch_list, label_patch_list, res])
        
            return image_patch_list,label_patch_list,cpts_res

"""
get image, and label patches
Default: returns one randomly selected patch of the given size, else
returns a list of all the patches.
"""
def get_patches(image, label, num_pos = 100, num_neg = 100, all_patches=False, patch_size=48, patch_shape= (48,48,48), spacing=0):

    image_shape = np.shape(image)
    sg_size = image_shape[0]
    cr_size = image_shape[1]
    ax_size = image_shape[2]

    if any(i < (patch_size+1) for i in image_shape): 
        pad_size = patch_size - np.array(image_shape)
        pad_size[pad_size < 0] = 0
        image = np.pad(image, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2],pad_size[2])), mode = "constant", constant_values = 0)
        label = np.pad(label, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2],pad_size[2])), mode = "constant", constant_values = 0)
        
        image_shape = np.shape(image)
        [sg_size, cr_size, ax_size] = image_shape

        
    if not all_patches:

            cpts_pos_sampled, cpts_neg_sampled = p.sample_center_points(label,num_pos,num_neg,patch_size, res = 1)
            
            image_patch_list, label_patch_list = crop_patch_by_cpts(image, label, cpts_pos_sampled, patch_size)
            
            image_neg_patch_list, label_neg_patch_list = crop_patch_by_cpts(image, label, cpts_neg_sampled, patch_size)
            
            image_patch_list.extend(image_neg_patch_list)
            label_patch_list.extend(label_neg_patch_list)
            
            cpts = np.concatenate((cpts_pos_sampled, cpts_neg_sampled), axis = 1)
            
            
    else:

            cpts = p.grid_center_points(image.shape, spacing)
            
            # Only include points not near boundary
            sg_idx = np.where(((patch_size/2) < cpts[0]) & (cpts[0] < (sg_size - (patch_size/2))))
            cpts = cpts[:,sg_idx[0]]
            cr_idx = np.where(((patch_size/2) < cpts[1]) & (cpts[1] < (cr_size - (patch_size/2))))
            cpts = cpts[:, cr_idx[0]]
            ax_idx = np.where(((patch_size/2) < cpts[2]) & (cpts[2] < (ax_size - (patch_size/2))))
            cpts = cpts[:, ax_idx[0]]

            # For test set, not writing to file, so standardize here
            image_patch_list, label_patch_list = crop_patch_by_cpts(image, label, cpts, patch_size)

    print(len(image_patch_list))
    return image_patch_list, label_patch_list, cpts


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
        img_type = self.image_list.iloc[idx,4] #train or val
    
        #Alternative in nibabel
        #image = nib.Nifti1Image.from_filename(self.image_list.iloc[idx,0])
        image = nib.load(self.image_list.iloc[idx,0])
        affine = image.affine
        image = image.get_fdata().astype(np.float32)
        
        seg = nib.load(self.image_list.iloc[idx,1])
        seg = seg.get_fdata().astype(np.float32)
    
        sample = {'image':image, 'seg':seg, 'id':image_id, 'affine':affine, 'type':img_type}
    
        if self.transform:
            sample = self.transform(sample)
    
        return sample

class PatchDataset(Dataset):
    
    def __init__(self, csv_file):
    
        self.image_list = pd.read_csv(csv_file, header = None)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
    
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        #Alternative in nibabel
        image = nib.load(self.image_list.iloc[idx,0])
        affine = image.affine
        image = image.get_fdata().astype(np.float32)
    
        seg = nib.load(self.image_list.iloc[idx,1])
        seg = seg.get_fdata().astype(np.float32)
    
        cpt = self.image_list.iloc[idx,2]
    
        sample = {'image':image, 'seg':seg, 'affine':affine, 'cpt':cpt}
    
    
        return sample

class GeneratePatches(Dataset):

    def __init__(self, sample,is_training = False,transform = False, multires = False, prior = None ):
    
        self.sample = sample            
    
        self.transform = transform
    
        img = sample['image']
        seg = sample['seg']
        self.affine = sample['affine']
    
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
    
        ## Standardize and normalize full image
        img_std = p.standardize_image(img)
        img_norm = p.Normalize(img_std)   
    
        #Returns center points of patches - useful when reconstructing test image
        if multires:
            
            img_patches, seg_patches, cpts = get_multi_resolution_patches(img_norm,seg,c.patchsize_multi_res, 
                                                                          num_pos = c.num_pos, num_neg = c.num_neg, 
                                                                          all_patches=all_patches,spacing=spacing)
             
        else:
            img_patches, seg_patches, cpts = get_patches(img_norm, seg,patch_size = c.segsize[0],
                                                     patch_shape = c.segsize, num_pos = c.num_pos, num_neg = c.num_neg, 
                                                     all_patches = all_patches, spacing = spacing)
             
        if self.transform:


            #Apply the transformation to a random subset of patches
            rnd_idx = random.sample(range(0,len(img_patches)), k = 100)
 
            for r in range(len(rnd_idx)):
                i = rnd_idx[r]
                img_elastic, seg_elastic = p.elastic_deformation(img_patches[i], seg_patches[i])
                img_rot, seg_rot = p.elastic_deformation(img_elastic, seg_elastic)
                
                ## adjuts brightness and contarst of patch (gamma augmentation)
                mu = 0
                sigma = 0.5
#                img_rot[img_rot == 0] = 1e-8
                img_rot = p.Normalize(img_rot)
                img_int = np.power(img_rot,np.exp(rnd.normal(mu,sigma)))
#                alpha = rnd.rand(c.segsize[0],c.segsize[0],c.segsize[0])*3 # alpha in range 0-3
#                beta = rnd.rand(c.segsize[0],c.segsize[0],c.segsize[0])*0.5
#                img_int = np.multiply(alpha,img_rot)*beta
                
#                img_int = p.standardize_image(img_int)
    
                img_patches.append(img_int)
                seg_patches.append(seg_rot)
                cpts = np.column_stack([cpts,cpts[:,i]])
    
        if is_training:
            
            if prior is not None:
                
                 write_patch_to_file(img_patches,seg_patches, sample, cpts, prior = prior)
                
            else:
                
                #Write patch/image and control points to csv and save image
                write_patch_to_file(img_patches,seg_patches, sample, cpts)
        else:
            
            if prior is not None:
                
                prior = np.pad(prior, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode = "constant", constant_values = 0)
                for idx in range(cpts.shape[1]):
        
                    image = img_patches[idx]
                    seg = seg_patches[idx]
                    cpt = cpts[:,idx]

                    patch_size = image.shape[0]
                    idx1_sg = cpt[0] - int(patch_size/2)
                    idx1_cr = cpt[1] - int(patch_size/2)
                    idx1_ax = cpt[2] - int(patch_size/2)
        
                    prior_patch = prior[idx1_sg:idx1_sg + patch_size, idx1_cr:idx1_cr+patch_size, idx1_ax: idx1_ax + patch_size]            
                    #add prior to image as a second channel and add back to list

                    image_prior = np.stack((image, prior_patch), axis = 0)
                    img_patches[idx] = image_prior
            else:
                
                for idx in range(cpts.shape[1]):
        
                    image = img_patches[idx]

                    img_patches[idx] = p.standardize_image(image)
                
                
    
        self.image_patches = img_patches
        self.seg_patches = seg_patches
        self.cpts = cpts
    
    
    def __len__(self):
        return len(self.image_patches)
    
    def __getitem__(self, idx):
        sample = {'image':self.image_patches[idx], 'seg':self.seg_patches[idx], 'cpt':self.cpts[:,idx]}
    
        return sample



## Includes a prior as an additional output
class ImageDataset_withPrior(Dataset):

    def __init__(self, csv_file, transform = None):

        self.image_list = pd.read_csv(csv_file, header = None)
        self.transform = transform
 
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
    
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
#        image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_list.iloc[idx,0]))
#        seg = sitk.GetArrayFromImage(sitk.ReadImage(self.image_list.iloc[idx,1]))
        image_id = self.image_list.iloc[idx,2]
        img_type = self.image_list.iloc[idx,4] #train or val
    
        #Alternative in nibabel
        #image = nib.Nifti1Image.from_filename(self.image_list.iloc[idx,0])
        image = nib.load(self.image_list.iloc[idx,0])
        affine = image.affine
        image = image.get_fdata().astype(np.float32)
        
        seg = nib.load(self.image_list.iloc[idx,1])
        seg = seg.get_fdata().astype(np.float32)
        
        prior = nib.load(self.image_list.iloc[idx,5])
        prior = prior.get_fdata().astype(np.float32)
#        seg_def = p.random_transform(seg)
#        
##        image_def, seg_def = p.elastic_deformation(image, seg)
#        nib.save(nib.Nifti1Image(seg_def, affine), "test.nii.gz" )

        # concatenate seg with image as a second channel
#        image_seg = np.stack((image, seg), axis = 0)
        
        sample = {'image':image, 'seg':seg, 'prior':prior, 'id':image_id, 'affine':affine, 'type':img_type}
    
        if self.transform:
            sample = self.transform(sample)
    
        return sample
    
## Auxiliary function for cropping patches
#def cropPatches(ispairAndParam):
#    image_np, seg_np, is_training, num_pos, num_neg = ispairAndParam    
#    # Crop patches
#    if is_training:
#       # patches_pos_multi_res, patches_neg_multi_res = patch_util.multi_resolution_patcher_3D(image_np[:, :, :, None], seg_np, c.patchsize_multi_res, is_training = is_training, num_pos = num_pos, num_neg = num_neg)    
#        patches_pos_multi_res, patches_neg_multi_res = patch_util.single_resolution_patcher_3D(image_np[:, :, :, None], seg_np,c.segsize, is_training = is_training, num_pos = num_pos, num_neg = num_neg)                            
#        # Fit the patch to deepmedic format
#        patch_pos = patch_to_Unet_format(patches_pos_multi_res, c.segsize)
#        patch_negative = patch_to_Unet_format(patches_neg_multi_res, c.segsize)
#
#        return patch_pos + patch_negative
#    else:
#        # Fit the patch to deepmedic format
#        #patches_multi_res = patch_util.single_resolution_patcher_3D(image_np[:, :, :, None], seg_np, c.patchsize_multi_res, is_training = is_training, spacing = c.test_patch_spacing)
#        patches_multi_res = patch_util.single_resolution_patcher_3D(image_np[:, :, :, None], seg_np,c.segsize, is_training = is_training, spacing = c.test_patch_spacing)
#        patches_multi_res = patch_to_Unet_format(patches_multi_res, c.segsize)
#        return patches_multi_res
#    
## This function converts the patch from patch generator function to one that deepmedic needs
#def patch_to_Unet_format(patches_multi_res, seg_size):
#    patches = []
#    #patches_high_res = patches_multi_res[0][0]
#    patches_high_res = patches_multi_res
#    
#    
#    for ph in patches_high_res:
#        seg = ph[1]
#        cpts = ph[2]
#        shape = ph[3]
#        patches.append((ph[0], seg, cpts, shape))
#    return patches

#def generate_deepmedic_patches(sample, is_training = True, num_pos = 100, num_neg = 100, aug = 1, num_thread = 1):
#  
#    image_np = sample['image']
#    seg_np = sample['seg']
#    
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#        
#    #Define tps sampler for data augmentation
#    height, width, depth = c.patchsize_multi_res[0][1]
#    
#    tps_sampler = TPSRandomSampler3D(height, width, depth,
#                 vertical_points=10, horizontal_points=10, depth_points=10,
#                 rotsd=10.0, scalesd=0.0, transsd=0.0, 
#                 warpsd=(0.00, 0.00),
#                 cache_size=0, pad=False, device = device)
#    
#    patches = []  
#    
#    # No augmentation in test case
#    #print("Augmenting...")
#    if not is_training:
#        image_np = image_np.astype(np.float32)
#        seg_np = seg_np
#        image_np = normalize_image(image_np)
#        image_seg_pairs_aug = (image_np, seg_np, is_training, num_pos, num_neg)
#            
#    # Augment the image in training case - since it uses gpu, only serial execution
#    else:
#        image_np_orig = image_np.astype(np.float32)
#        seg_np_orig = seg_np
#        
#        for j in range(aug):
#            if aug == 1: # No augmentation
#                image_np, seg_np = image_np_orig, seg_np_orig
#            else:
#                # Apply tps on GPU and then move back to cpu
#                ## Add single dimensions
#                im_torch = image_np_orig[None, None, ...]
#                im_torch = torch.from_numpy(im_torch).to(device)
#                
#                seg_torch = seg_np_orig[None, None, ...]
#                seg_torch = torch.from_numpy(seg_torch).to(device)
#                
#                image_np, seg_np = tps_sampler((im_torch, seg_torch))
#                image_np = image_np.squeeze().cpu().numpy()            
#                seg_np = seg_np.squeeze().cpu().numpy()
#                    
#            # Normalize the image
#            image_np = image_np.astype(np.float32)
#            image_np = normalize_image(image_np)
#            image_seg_pairs_aug = (image_np, seg_np, is_training, num_pos, num_neg)
#     
#     # Parallelize the process of generating patches
#    #print("Cropping patches...")
#        
#    patches = cropPatches(image_seg_pairs_aug)
#
#    return patches   

# Auxiliary function for loading image and segmentation
#def loadImageSegPair(dirDiseaseTuple):
#    imageDir, segDir, patient, disease = dirDiseaseTuple
#    
#    # Load the image and segmentation
#    #imageDir = patientDir + "FLAIR/FLAIR_1x1x1.nii.gz"
#    #segDir = patientDir + "FLAIR/ManSeg_1x1x1.nii.gz"
#    
#    # Read in image and segmentation
#    image_np_orig = sitk.GetArrayFromImage(sitk.ReadImage(imageDir))
#    seg_np_orig = sitk.GetArrayFromImage(sitk.ReadImage(segDir))
#    
#    return image_np_orig, seg_np_orig, patient, disease
