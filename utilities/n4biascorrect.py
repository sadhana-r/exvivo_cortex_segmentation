#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:34:37 2020

@author: sadhana-ravikumar
"""
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd


def bias_correct(input_image):
	"""
	N4BiasFieldCorrection: PICSL UPenn
	"""
	bias_free_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
	return bias_free_image
    
csv_file = '/home/sadhana-ravikumar/Documents/Sadhana/unet3d_srlm/data_csv/split.csv'
output_folder = '/home/sadhana-ravikumar/Documents/Sadhana/unet3d_srlm/inputs/'
image_list = pd.read_csv(csv_file, header = None)


for idx in range(len(image_list)):
    file_name = image_list.iloc[idx,0]
    image = nib.load(image_list.iloc[idx,0])
    affine = image.affine
#    image = image.get_fdata().astype(np.float32)
    
#    image = sitk.GetArrayFromImage(sitk.ReadImage(image_list.iloc[idx,0]))
    
    #n4 bias correction
    image_n4 = bias_correct(image)
    
    output_filename =  file_name[:-7] + '_n4.nii.gz' 
    nib.save(nib.Nifti1Image(image_n4, affine), output_filename)
    

    