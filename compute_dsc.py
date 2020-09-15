#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:33:36 2020

@author: sadhana-ravikumar
"""
import SimpleITK as sitk

def computeGeneralizedDSC(gt, seg):
    
     gt_seg = gt[gt > 0]
     myseg = seg[gt > 0]
     
     gdsc = 100*(sum(gt_seg == myseg)/ len(gt_seg))
     
     return gdsc

val_dir = '/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/validation_output/'
#exp_dir = 'Experiment_23072020_pulks/'
exp_dir = 'Experiment 5/'
#exp_dir = 'Experiment_14082020_MGNet/'
predicted_seg = 'seg_3918'
input_dir = '/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/inputs/'
#gt_seg = 'HNL39_18-R/HNL39_18-R_trimmed_phg.nii.gz'
#gt = sitk.GetArrayFromImage(sitk.ReadImage(input_dir+gt_seg))
#seg = sitk.GetArrayFromImage(sitk.ReadImage(val_dir + exp_dir + predicted_seg))
#
#dsc = computeGeneralizedDSC(gt,seg)
#print(dsc)

img =input_dir+'HNL39_18-R/HNL39_18-R_trimmed_img.nii.gz'
image1=sitk.GetArrayFromImage(sitk.ReadImage(img))