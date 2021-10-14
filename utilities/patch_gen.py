#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:12:14 2020

@author: sadhana-ravikumar
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import random

# This function generate center points in order of image. Just to keep the API consistent
def grid_center_points(shape, space):
	#patch_width = int(patch_size/2)
	x = np.arange(start = 0, stop = shape[0], step = space[0] )
	y = np.arange(start = 0, stop = shape[1], step = space[1] )
	z = np.arange(start = 0, stop = shape[2], step = space[2])
	x_t, y_t, z_t = np.meshgrid(x, y, z, indexing = "ij")
    
	idx = np.stack([x_t.flatten(), y_t.flatten(), z_t.flatten()], axis = 0)
    
	return idx

def sample_center_points(label,num_pos,num_neg, patch_size, res = 1):
    
    image_shape = np.shape(label)
    sg_size = image_shape[0]
    cr_size = image_shape[1]
    ax_size = image_shape[2]
    
    idx_pos = np.stack(np.where(label > 0))
		
    min_boundary = np.ceil((patch_size/2)/res)
	# Only include points not near boundary
    sg_idx = np.where((min_boundary <= idx_pos[0]) & (idx_pos[0] <= (sg_size - min_boundary)))
    idx_pos = idx_pos[:,sg_idx[0]]
    cr_idx = np.where((min_boundary <= idx_pos[1]) & (idx_pos[1] <= (cr_size - min_boundary)))
    idx_pos = idx_pos[:, cr_idx[0]]
    ax_idx = np.where((min_boundary <= idx_pos[2]) & (idx_pos[2] <= (ax_size - min_boundary)))
    idx_pos = idx_pos[:, ax_idx[0]]

    if idx_pos[0].shape[0]<num_pos:
        cpts_pos_sampled = idx_pos
    else:
        idx_rand = np.random.choice(idx_pos[0].shape[0], num_pos, replace = False)
        cpts_pos_sampled = idx_pos[:, idx_rand] 
    
    # For negative points
    idx_neg = np.stack(np.where(label==0), axis = 0)
		
    #Only include points not near boundary
    sg_idx = np.where((min_boundary <= idx_neg[0]) & (idx_neg[0] <= (sg_size - min_boundary)))
    idx_neg = idx_neg[:,sg_idx[0]]
    cr_idx = np.where((min_boundary <= idx_neg[1]) & (idx_neg[1] <= (cr_size - min_boundary)))
    idx_neg = idx_neg[:, cr_idx[0]]
    ax_idx = np.where((min_boundary <= idx_neg[2]) & (idx_neg[2] <= (ax_size - min_boundary)))
    idx_neg = idx_neg[:, ax_idx[0]]
	
    if idx_neg[0].shape[0]<num_neg:
        cpts_neg_sampled = idx_neg
    else:	
        idx_rand = np.random.choice(idx_neg[0].shape[0], num_neg, replace = False)
        cpts_neg_sampled = idx_neg[:, idx_rand] 
        
    return cpts_pos_sampled,cpts_neg_sampled


#This function define standardization to image
def standardize_image(image_np):
    
    eps = 1e-6
    image_voxels = image_np[image_np>0] # Get rid of the background
    if image_voxels.size == 0:
        mean = 0
        std = 0
    else:
        mean = np.mean(image_voxels)
        std = np.std(image_voxels)
        
    image_np = (image_np - mean) / np.clip(std,a_min = eps,a_max = None)
    if np.isnan(np.sum(image_np)):
        print("break")
    return image_np

def Normalize(image, min_value=0, max_value=1):
	"""
	chnage the intensity range
	"""
	value_range = max_value - min_value
	normalized_image = (image - np.min(image)) * (value_range) / (np.max(image) - np.min(image))
	normalized_image = normalized_image + min_value
	return normalized_image


# Random transform
def RandomRotate90(image,seg, second_chan = None, patch_size=64):
    """
    Randomly rotate an image
    """
    image = np.reshape(image, (patch_size, patch_size, patch_size))
    seg = np.reshape(seg, (patch_size, patch_size, patch_size))
    k = random.randint(0, 4)
    image_rot = np.rot90(image, k, (1, 2))
    seg_rot = np.rot90(seg, k, (1, 2))
    
    if second_chan is None:
        return image_rot, seg_rot
    else:
        second_chan_rot = np.rot90(second_chan, k, (1, 2))
        return image_rot, seg_rot, second_chan_rot
        

def elastic_deformation(img, seg = None, second_chan = None, alpha=15, sigma=3):
	"""
	Elastic deformation of 2D or 3D images on a pixelwise basis
	X: image
	Y: segmentation of the image
	alpha = scaling factor the deformation
	sigma = smooting factor
	inspired by: https://gist.github.com/fmder/e28813c1e8721830ff9c which inspired imgaug through https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
	based on [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
	   Convolutional Neural Networks applied to Visual Document Analysis", in
	   Proc. of the International Conference on Document Analysis and
	   Recognition, 2003.
	First a random displacement field (sampled from a gaussian distribution) is created,
	it's then convolved with a gaussian standard deviation, σ determines the field : very small if σ is large,
		like a completely random field if σ is small,
		looks like elastic deformation with σ the elastic coefficent for values in between.
	Then the field is added to an array of coordinates, which is then mapped to the original image.
	"""
	shape = img.shape
	dx = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha #originally with random_state.rand * 2 - 1
	dy = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
	if len(shape)==2:
		x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
		indices = x+dx, y+dy

	elif len(shape)==3:
		dz = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
		x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
		indices = x+dx, y+dy, z+dz

	else:
		raise ValueError("can't deform because the image is not either 2D or 3D")

	if seg is None:
		if second_chan is None:
 			return map_coordinates(img, indices, order=3).reshape(shape), None
	else:
		if second_chan is None:
			return map_coordinates(img, indices, order=3).reshape(shape), map_coordinates(seg, indices, order=0).reshape(shape)
		else:
			return map_coordinates(img, indices, order=3).reshape(shape), map_coordinates(seg, indices, order=0).reshape(shape), map_coordinates(second_chan, indices, order=3).reshape(shape)


def RandomFlip(image, image_label):
	"""
	Randomly flips the image across the given axes.
	Note from original repo: When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
	otherwise the models won't converge.
	"""
	axes = (0, 1, 2)
	image_rot = np.flip(image, axes[1])
	label_rot = np.flip(image_label, axes[1])
	return image_rot, label_rot
