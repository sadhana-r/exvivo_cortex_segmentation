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
import medpy.metric as metric
import csv
import os

def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)
    
class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))

def computeGeneralizedDSC(gt, seg):
    
     gt_seg = gt[gt > 0]
     myseg = seg[gt > 0]
     
     numerator = sum(gt_seg == myseg)
     denominator = 2*len(gt_seg)
     gdsc = 2*numerator/denominator
     
     return gdsc
 
def computeIndividualDSC(gt, seg, labels):
    
    gt_seg = gt[gt > 0]
    myseg = seg[gt > 0]
     
#    dsc = np.zeros(len(labels),1)
    dsc = []
    for i in labels:
    
        gt_label = (gt_seg == i).astype(float)
        pred_label = (myseg == i).astype(float)
        dsc.append(metric.binary.dc(pred_label, gt_label))
        
    
    return dsc

def ComputeHausdorffDistance(gt,seg, voxelspacing=None, connectivity=1):
    
    #Restrict to area where I have segmentations
    seg[gt == 0] = 0
    
    #Only keep gray matter
    seg[seg != 1] = 0
    gt[gt != 1] = 0
    
    hd = metric.binary.hd95(seg, gt, voxelspacing, connectivity)
    
    return hd

    
root_dir = "/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet"
train_val_csv = root_dir + "/data_csv/split.csv" 
exp_dir = 'Experiment_02112021_fourlabels_removedHF'
#exp_dir = 'nnUNET_Task508_ExvivoMTL'
val_dir = root_dir + '/validation_output/' + exp_dir
input_dir = root_dir + '/inputs/'
metrics_csv = val_dir + '/eval_metrics.csv'

labels = [1,2,3,4]

with open(metrics_csv, 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ID','Experiment',1,2,3,4,'symmetric HD 95'])

image_dataset = p.ImageDataset(csv_file = train_val_csv)

dsc_list = []
for i in range(0,len(image_dataset)):
    
        sample = image_dataset[i]
        if(sample['type'] == 'test'):
        
            image_id = sample['id']
            seg = sample['seg']
            seg[seg==5] = 1
            
            subj_metrics = [image_id, exp_dir]
            print(image_id)
            for fname in os.listdir(val_dir):
                fname_edit = fname.replace("_","")
                if str(image_id) in fname_edit:
                    predicted_segfile = fname
                    print(predicted_segfile)
        
#            predicted_segfile = val_dir + '/' + predicted_segfile
            predicted_segfile = val_dir + '/seg_' + str(image_id) + ".nii.gz" 
            pred_seg =  nib.load(predicted_segfile)
            pred_seg = pred_seg.get_fdata().astype(np.float32)
            pred_seg[seg == 0] = 0
            
            for i in labels:
                print("Label: ",i)
                test = pred_seg == i
                reference = seg == i
                subj_metrics.append(dice(test,reference))
                
                
    #            dsc = computeGeneralizedDSC(seg,pred_seg)
    #            subj_metrics.append(dsc)
    #            
                # To double check other approach. Gives same result!
    #            dsc = computeIndividualDSC(seg, pred_seg, labels)
    #            
            hd = ComputeHausdorffDistance(seg,pred_seg, voxelspacing = 0.2)
            subj_metrics.append(hd)
                
            with open(metrics_csv, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(subj_metrics)
 
#print("Average srlm validation accuracy is ", sum(dsc_list)/len(dsc_list))
#print(dsc_list)
#print("Standard deviation is ", np.std(dsc_list))