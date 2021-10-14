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

c = config.Config_BaselineUnet()
dir_names = config.Setup_Directories()

#Writes a patch (image and seg) to file. Saves control point in csv
def write_patch_to_file(image_patch_list, label_patch_list, sample, cpt_list, second_chan_patch_list = None):
    
    print("Writing patches to file")
    print(cpt_list.shape[1])
    for idx in range(cpt_list.shape[1]):
        
        image = image_patch_list[idx]
        seg = label_patch_list[idx]
        cpt = cpt_list[:,idx]

        img_id = sample['id']
        
        # standardize patch before writing to file
        image = p.standardize_image(image) 
        
        # If there is a prior, then concatenate with image patch as second channel
#        if second_chan is not None:
#            print("Extract corresponding patch from prior")
#            patch_size = image.shape[0]
#            idx1_sg = cpt[0] - int(patch_size/2)
#            idx1_cr = cpt[1] - int(patch_size/2)
#            idx1_ax = cpt[2] - int(patch_size/2)
#
#            second_chan_patch = second_chan[idx1_sg:idx1_sg + patch_size, idx1_cr:idx1_cr+patch_size, idx1_ax: idx1_ax + patch_size]            
            #add prior to image as a second channel
            #image = np.stack((image, second_chan_patch), axis = -1)

        if (sample['type'] == 'train'):
            img_filename = osp.join(dir_names.patch_dir, "training_data","img_"+ str(img_id) + '_' +str(idx) + ".nii.gz")
            seg_filename = osp.join(dir_names.patch_dir, "training_data" ,"seg_"+ str(img_id) +  '_' +str(idx) + ".nii.gz" )
            dmap_filename = osp.join(dir_names.patch_dir, "training_data" ,"dmap_"+ str(img_id) +  '_' +str(idx) + ".nii.gz" )
            csv_file = dir_names.train_patch_csv
        else:
            img_filename = osp.join(dir_names.patch_dir, "validation_data","img_"+ str(img_id) + '_' +str(idx) + ".nii.gz")
            seg_filename = osp.join(dir_names.patch_dir, "validation_data" ,"seg_" + str(img_id) + '_' +str(idx) + ".nii.gz" )
            dmap_filename = osp.join(dir_names.patch_dir, "validation_data" ,"dmap_"+ str(img_id) +  '_' +str(idx) + ".nii.gz" )
            csv_file = dir_names.val_patch_csv
        
        nib.save(nib.Nifti1Image(image, sample['affine']), img_filename)
        
        if second_chan_patch_list is not None:
            
            second_chan_patch = second_chan_patch_list[idx]
            if (sample['type'] == 'train'):
                dmap_filename = osp.join(dir_names.patch_dir, "training_data" ,"dmap_"+ str(img_id) +  '_' +str(idx) + ".nii.gz" )
            else:
                dmap_filename = osp.join(dir_names.patch_dir, "validation_data" ,"dmap_"+ str(img_id) +  '_' +str(idx) + ".nii.gz" )
                
            nib.save(nib.Nifti1Image(second_chan_patch, sample['affine']), dmap_filename)
              
             #save filenames and corresponding control point in csv
            data_list = (img_filename,seg_filename, cpt, dmap_filename)
            with open(csv_file, 'a') as f:
                  wr = csv.writer(f)
                  wr.writerow(data_list)
        else:
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
def crop_patch_by_cpts(image, cpts, patch_size):
    image_patch_list = []
#    label_patch_list = []
    patch_width = int(patch_size/2)
    for i in range(cpts.shape[1]):
        idx1_sg = cpts[0][i] - patch_width
        idx1_cr = cpts[1][i] - patch_width
        idx1_ax = cpts[2][i] - patch_width

        idx2_sg = idx1_sg + patch_size
        idx2_cr = idx1_cr + patch_size
        idx2_ax = idx1_ax + patch_size

        image_patch = image[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
#        label_patch = label[idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
    
        image_patch_list.append(image_patch)
#        label_patch_list.append(label_patch)

    return image_patch_list #, label_patch_list


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
            
            image_patch_list = crop_patch_by_cpts(image, cpts_pos_sampled, patch_size)
            label_patch_list = crop_patch_by_cpts(label, cpts_pos_sampled, patch_size)
            
            image_neg_patch_list = crop_patch_by_cpts(image, cpts_neg_sampled, patch_size)
            label_neg_patch_list = crop_patch_by_cpts(label, cpts_neg_sampled, patch_size)
            
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
            image_patch_list = crop_patch_by_cpts(image, cpts, patch_size)
            label_patch_list = crop_patch_by_cpts(label, cpts, patch_size)
    print(len(image_patch_list))
    return image_patch_list, label_patch_list, cpts

"""
get image, and label patches
Default: returns one randomly selected patch of the given size, else
returns a list of all the patches.
"""
def get_patches_withsecondchan(image, label, second_chan,num_pos = 100, num_neg = 100, all_patches=False, patch_size=48, patch_shape= (48,48,48), spacing=0):

    image_shape = np.shape(image)
    sg_size = image_shape[0]
    cr_size = image_shape[1]
    ax_size = image_shape[2]

    if any(i < (patch_size+1) for i in image_shape): 
        pad_size = patch_size - np.array(image_shape)
        pad_size[pad_size < 0] = 0
        image = np.pad(image, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2],pad_size[2])), mode = "constant", constant_values = 0)
        label = np.pad(label, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2],pad_size[2])), mode = "constant", constant_values = 0)
        second_chan = np.pad(second_chan, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2],pad_size[2])), mode = "constant", constant_values = 0)
        
        image_shape = np.shape(image)
        [sg_size, cr_size, ax_size] = image_shape

        
    if not all_patches:

            cpts_pos_sampled, cpts_neg_sampled = p.sample_center_points(label,num_pos,num_neg,patch_size, res = 1)
            
            image_patch_list = crop_patch_by_cpts(image, cpts_pos_sampled, patch_size)
            label_patch_list = crop_patch_by_cpts(label, cpts_pos_sampled, patch_size)
            second_chan_patch_list = crop_patch_by_cpts(second_chan, cpts_pos_sampled, patch_size)
            
            image_neg_patch_list = crop_patch_by_cpts(image, cpts_neg_sampled, patch_size)
            label_neg_patch_list = crop_patch_by_cpts(label, cpts_neg_sampled, patch_size)
            second_chan_neg_patch_list = crop_patch_by_cpts(second_chan, cpts_neg_sampled, patch_size)          
            
            image_patch_list.extend(image_neg_patch_list)            
            label_patch_list.extend(label_neg_patch_list)
            second_chan_patch_list.extend(second_chan_neg_patch_list)
            
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
            image_patch_list = crop_patch_by_cpts(image, cpts, patch_size)
            label_patch_list = crop_patch_by_cpts(label, cpts, patch_size)
            second_chan_patch_list = crop_patch_by_cpts(second_chan, cpts, patch_size)

    print(len(image_patch_list))
    return image_patch_list, label_patch_list,second_chan_patch_list, cpts


class ImageDataset(Dataset):

    def __init__(self, csv_file, transform = None):

        self.image_list = pd.read_csv(csv_file, header = None)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
    
        if torch.is_tensor(idx):
            idx = idx.tolist()

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

## Includes distance map as an additional output
class ImageDataset_withDMap(Dataset):

    def __init__(self, csv_file, transform = None):

        self.image_list = pd.read_csv(csv_file, header = None)
        self.transform = transform
 
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
    
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        image_id = self.image_list.iloc[idx,2]
        img_type = self.image_list.iloc[idx,4] #train or val
    
        #Alternative in nibabel
        #image = nib.Nifti1Image.from_filename(self.image_list.iloc[idx,0])
        image = nib.load(self.image_list.iloc[idx,0])
        affine = image.affine
        image = image.get_fdata().astype(np.float32)
        
        seg = nib.load(self.image_list.iloc[idx,1])
        seg = seg.get_fdata().astype(np.float32)
        
        second_chan = nib.load(self.image_list.iloc[idx,5])
        second_chan = second_chan.get_fdata().astype(np.float32)
        
        sample = {'image':image, 'seg':seg, 'second_chan':second_chan, 'id':image_id, 'affine':affine, 'type':img_type}
    
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class PatchDataset(Dataset):
    
    def __init__(self, csv_file, include_second_chan = False):
    
        self.image_list = pd.read_csv(csv_file, header = None)
        self.include_second_chan = include_second_chan
    
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
        
        if self.include_second_chan:            
            dmap = nib.load(self.image_list.iloc[idx,3])
            dmap = dmap.get_fdata().astype(np.float32)
        else:
            dmap = None
    
        sample = {'image':image, 'seg':seg, 'affine':affine, 'cpt':cpt, 'dmap':dmap}
    
        return sample

    
class GeneratePatches(Dataset):

    def __init__(self, sample, patch_size, is_training = False,transform = False, multires = False, include_second_chan = False ):
    
        self.sample = sample            
    
        self.transform = transform
    
        img = sample['image']
        seg = sample['seg']           
        self.affine = sample['affine']        
        if include_second_chan:
            second_chan = sample['second_chan']
    
        if is_training:
            all_patches = False
            spacing = None
            
        else:
            all_patches = True
            spacing = c.test_patch_spacing
            # Pad test image to take care of boundary condition
            pad_size = c.half_patch[0]
            img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode = "reflect")
            seg = np.pad(seg, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode = "constant", constant_values = 0)
            if include_second_chan:
                second_chan = np.pad(second_chan, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)), mode = "constant", constant_values = 0)
            
            
        ## Standardize and normalize full image
        img_std = p.standardize_image(img)
        img_norm = p.Normalize(img_std)   
    
        #Returns center points of patches - useful when reconstructing test image
        if multires:
            second_chan_patches = None
            img_patches, seg_patches, cpts = get_multi_resolution_patches(img_norm,seg,c.patchsize_multi_res, 
                                                                          num_pos = c.num_pos, num_neg = c.num_neg, 
                                                                          all_patches=all_patches,spacing=spacing)
        elif include_second_chan:
             img_patches, seg_patches,second_chan_patches, cpts = get_patches_withsecondchan(img_norm, seg, second_chan, patch_size = patch_size[0],
                                                     patch_shape = patch_size, num_pos = c.num_pos, num_neg = c.num_neg, 
                                                     all_patches = all_patches, spacing = spacing)
             
        else:
            second_chan_patches = None
            img_patches, seg_patches, cpts = get_patches(img_norm, seg,patch_size = patch_size[0],
                                                     patch_shape = patch_size, num_pos = c.num_pos, num_neg = c.num_neg, 
                                                     all_patches = all_patches, spacing = spacing)
                         
        if self.transform:

            #Apply the transformation to a random subset of patches
            rnd_idx = random.sample(range(0,len(img_patches)), k = c.aug)
 
            for r in range(len(rnd_idx)):
                
                i = rnd_idx[r]

                if include_second_chan:
                    img_elastic, seg_elastic, second_chan_elastic = p.elastic_deformation(img_patches[i], seg_patches[i],second_chan_patches[i])
                    img_rot, seg_rot, second_chan_rot = p.RandomRotate90(img_elastic, seg_elastic, second_chan_elastic,c.segsize[0])
                    second_chan_patches.append(second_chan_rot)
                   
                    
                else:        
                     img_elastic, seg_elastic = p.elastic_deformation(img_patches[i], seg_patches[i])
                     img_rot, seg_rot = p.RandomRotate90(img_elastic, seg_elastic,None, c.segsize[0])
                
                ## adjuts brightness and contarst of patch (gamma augmentation)
                mu = 0
                sigma = 0.5
#                img_rot[img_rot == 0] = 1e-8
                img_rot = p.Normalize(img_rot)
                img_int = np.power(img_rot,np.exp(rnd.normal(mu,sigma)))  #gamma augmentation
                    
                img_patches.append(img_int)
                seg_patches.append(seg_rot)
                cpts = np.column_stack([cpts,cpts[:,i]])
    
        if is_training:
            write_patch_to_file(img_patches,seg_patches, sample, cpts, second_chan_patch_list = second_chan_patches)

        else:
            
            for idx in range(cpts.shape[1]):
        
                    image = img_patches[idx]
                    img_patches[idx] = p.standardize_image(image)
                
                
    
        self.image_patches = img_patches
        self.seg_patches = seg_patches
        self.second_chan_patches = second_chan_patches
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
