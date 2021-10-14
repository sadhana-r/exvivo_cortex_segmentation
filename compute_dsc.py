#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:33:36 2020

@author: sadhana-ravikumar
"""
import nibabel as nib
import sys
sys.path.append('./utilities')
import preprocess_data as p
import numpy as np


def computeGeneralizedDSC(gt, seg):
    
     gt_seg = gt[gt > 0]
     myseg = seg[gt > 0]
     
     gdsc = 100*(sum(gt_seg == myseg)/ len(gt_seg))
     
     return gdsc

root_dir = "/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet"
train_val_csv = root_dir + "/data_csv/split.csv" 
exp_dir = 'Experiment_060120201_updateddata/'
val_dir = root_dir + '/validation_output/' + exp_dir

input_dir = root_dir + '/inputs/'


image_dataset = p.ImageDataset(csv_file = train_val_csv)

dsc_list = []
for i in range(1,len(image_dataset)):
        sample = image_dataset[i]
        if(sample['type'] == 'test'):
        
            image_id = sample['id']
            seg = sample['seg']
            print(image_id)
            predicted_segfile = val_dir + 'seg_' + str(image_id) + ".0.nii.gz" 
            pred_seg =  nib.load(predicted_segfile)
            pred_seg = pred_seg.get_fdata().astype(np.float32)
            
            dsc = computeGeneralizedDSC(seg,pred_seg)
            dsc_list.append(dsc)
 
print("Average srlm validation accuracy is ", sum(dsc_list)/len(dsc_list))
print(dsc_list)
print("Standard deviation is ", np.std(dsc_list))
