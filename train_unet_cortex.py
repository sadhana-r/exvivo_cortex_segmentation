#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:05:15 2020

@author: sadhana-ravikumar
"""
import sys
sys.path.append('./utilities')
sys.path.append('./utilities/pulkit')

from unet_model import UNet, MGNet, UNet_wDeepSupervision
import numpy as np
import config_cortex as config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch
import preprocess_data as p
import nibabel as nib
import os.path as osp
import os
import loss as l
import math
import shutil

#Try Pulkit's code
#import Unet3D_meta_learning

def Normalize(image, min_value=0, max_value=1):
	"""
	change the intensity range
	"""
	value_range = max_value - min_value
	normalized_image = (image - torch.min(image)) * (value_range) / (torch.max(image) - torch.min(image))
	normalized_image = normalized_image + min_value
	return normalized_image

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:

    def __init__(self, n=4):
        
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def computeGeneralizedDSC_patch(probability, seg):
    
     seg = seg.cpu().numpy()
     probability = probability.cpu().numpy()
     preds = np.argmax(probability, 1)
     
     gt = seg[seg > 0]
     myseg = preds[seg > 0]
     
     gdsc = sum(gt == myseg)/ len(gt)
     
     return gdsc

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
    
    return preds_tensor, probability
    
def plot_images_to_tfboard(img, seg, output, step, is_training = True, num_image_to_show = 3):
    
    preds, probability = generate_prediction(output)
    
    img = Normalize(img, max_value = 255)
#    i = 0
    if is_training:
       for i in range(num_image_to_show):
            writer.add_image('Training/Intensity images/'+str(i), img[i,:,:,:,24], global_step = step)
            writer.add_image('Training/Ground Truth seg/'+ str(i), color_transform(seg[i,None,:,:,24]), global_step = step)
            writer.add_image('Training/Predicted seg/'+ str(i), color_transform(preds[i,None,:,:,24]), global_step = step)
    else:
        for i in range(num_image_to_show):
            writer.add_image('Validation/Intensity images/'+str(i), img[i,:,:,:,24], global_step = step)
            writer.add_image('Validation/Ground Truth seg/'+ str(i), color_transform(seg[i,None,:,:,24]), global_step = step)
            writer.add_image('Validation/Predicted seg/'+ str(i), color_transform(preds[i,None,:,:,24]), global_step = step)

def center_crop(layer, target_size):
    # only four elements since channels is 1
    _, layer_height, layer_width, layer_depth = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    diff_z = (layer_width - target_size[2]) // 2
    return layer[
        :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1]),  diff_z : (diff_z + target_size[2])
    ]
        
def load_ckp(ckp_path, net, optimizer):
    checkpoint = torch.load(ckp_path)
    net.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    return net, optimizer, epoch
    
c = config.Config_BaselineUnet()
dir_names = config.Setup_Directories()

# Set up GPU if available    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Set up directories
root_dir = dir_names.root_dir
experiment_name = 'Experiment_02112021_fourlabels_removedHF_test'
tfboard_dir = dir_names.tfboard_dir + '/' + experiment_name
model_dir = dir_names.model_dir + '/' + experiment_name + '/'
output_dir = dir_names.valout_dir + '/' + experiment_name + '/'
model_file = model_dir + 'model.pth'
 
#Network properties - parameters to set (in future make these argparse)
num_class = 5
generate_uncertainty = False
num_groups = 4
unet_num_levels = 3
include_second_chan = True   ## Prior flag - include a second input channel for UNet w/ prior or distance map
unet_in_channels = 1
unet_init_feature_numbers = 32
resume_training = False
load_model = False

# Load model or not. If load model, the modelDir and tfboardDir should have existed. Otherwise they
# will be created forcefully, wiping off the old one.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not load_model:
    c.force_create(model_dir)
    c.force_create(tfboard_dir)
    
# load unet model
if not load_model:
    #net=Unet3D_meta_learning.Net(num_classes = num_class)
    if generate_uncertainty:
        net = MGNet(num_class = num_class, num_groups = num_groups)
        net = net.to(device)
    else:
#        net = UNet(num_class = num_class, padding = False, num_levels = 3, init_feature_number = 64)
        net = UNet_wDeepSupervision(num_class = num_class, patch_size = c.segsize, in_channels = unet_in_channels,
                                    num_levels = unet_num_levels,init_feature_number = unet_init_feature_numbers, padding = False)
        net = net.to(device)
else:
    #net=Unet3D_meta_learning.Net(num_classes = num_class)
    net = UNet_wDeepSupervision(num_class = num_class, patch_size = c.segsize,in_channels = unet_in_channels,
                                num_levels = unet_num_levels,init_feature_number = unet_init_feature_numbers, padding = False)
    
    ckp_path = model_dir + 'best_model.pth'
    net, _, start_epoch = load_ckp(ckp_path, net,None)
#    net = UNet(num_class = num_class, padding = False, num_levels = 4, init_feature_number = 64)
    net = net.to(device)
    net.eval()

print(net)

#Initialize class to convert labels to color images
color_transform = Colorize(n = num_class)
    
# Set up tensor board
writer = SummaryWriter(tfboard_dir)
  
alpha = torch.ones(unet_num_levels - 1)
alpha[0] = 0.4 # [0.4,1]
alpha = alpha.to(device)

# Define a loss function and optimizer
weights = torch.ones(num_class)
weights[0] = 0
weights = weights.to(device)

#criterion = l.GeneralizedDiceLoss(num_classes=num_class, weight = weights)
criterion = l.DSCLoss_deepsupervision(weights, alpha = alpha, num_classes = num_class)
#criterion = torch.nn.CrossEntropyLoss(weights)
#criterion = l.CELoss_deepsupervision(weights)

# Optimizer and learning rate
optimizer = optim.Adam(net.parameters(), lr = c.learning_rate, weight_decay = c.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = c.step_size, gamma = 0.5)

#Set up data
#Define image dataset (reads in full images and segmentations)
image_dataset = p.ImageDataset_withDMap(csv_file = c.train_val_csv)

"""
DO ONCE UNLESS I NEED TO UPDATE PATCHES
"""
# Save and genearte patches for datasets
tp_dir = dir_names.patch_dir + "/training_data_dmap"
vp_dir = dir_names.patch_dir + "/validation_data_dmap"

c.force_create(tp_dir)
c.force_create(vp_dir)
if os.path.exists(dir_names.train_patch_csv):
    os.remove(dir_names.train_patch_csv)
if os.path.exists(dir_names.val_patch_csv):
    os.remove(dir_names.val_patch_csv)
  
##Train/val split determined in the split_data.csv len(image_dataset)
for i in range(len(image_dataset)):
    print(i)
    sample = image_dataset[i]
     
    if(sample['type'] == 'train'):
        print(sample['type'])
        patches = p.GeneratePatches(sample, patch_size = c.segsize, is_training = True, transform = True, include_second_chan= include_second_chan)
    else:
        patches = p.GeneratePatches(sample, patch_size = c.segsize, is_training = True, transform = False, include_second_chan = include_second_chan)
"""
PATCHH GENERATION UP TO HERE

 
###Now create a dataset with the patches
train_dataset = p.PatchDataset(dir_names.train_patch_csv, include_second_chan)
val_dataset = p.PatchDataset(dir_names.val_patch_csv, include_second_chan)

if resume_training:
    ckp_path = model_dir + 'checkpoint.pt'
    net, optimizer, start_epoch = load_ckp(ckp_path, net, optimizer)
else:
    start_epoch = 0


# Training loop
training_loss = 0.0

for epoch in range(start_epoch, c.num_epochs):
    
    
    trainloader = DataLoader(train_dataset, batch_size = c.batch_size, shuffle = True, num_workers = c.num_thread)
    net.train()
    for j, patch_batched in enumerate(trainloader,0):
        
        img = patch_batched['image'][:,None,...].to(device)
        
        seg = patch_batched['seg'].to(device)
        
        #Combine hippocampus with cortex 
        if num_class ==5:
            seg[seg==5] = 1
        elif num_class == 4:
            seg[seg==4] = 1
                
        #Zero the parameter gradients
        optimizer.zero_grad()

        if generate_uncertainty:
            output_list = net(img)
            
            loss_uncertain = 0
            for output in output_list:
                loss_uncertain += criterion(output, seg.long())
            loss = loss_uncertain/num_groups
                
            ouput = torch.mean(torch.stack(output_list), axis = 0)
            uncertainty =  (torch.std(torch.stack(output_list), axis = 0))**2
        else:
             # forward + backward + optimize
#             output,_ = net(img)
             output,_,ds_outputs = net(img)
             
             ## With no padding, need to crop label
             seg = center_crop(seg,output.shape[2:])
             loss = criterion(output,ds_outputs, seg.long())
#             loss = criterion(output, seg.long())
             
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()

        if j % 5 == 4: #print every 5 batches
            #Plot images
            plot_images_to_tfboard(img, seg, output, epoch*len(trainloader) + j,
                                   is_training = True, num_image_to_show = c.num_image_to_show)   
            print('Training loss: [epoch %d,  iter %5d] loss: %.3f lr: %.5f' %(epoch +1, j+1, training_loss/5, scheduler.get_lr()[0]))
            writer.add_scalar('training_loss', training_loss/5, epoch*len(trainloader) + j)
            training_loss = 0.0
            
    
    ## Validation   
    best_val_loss = math.inf
    validation_loss = 0
    validation_dsc = 0
    count = 0
    with torch.no_grad():
        
        valloader = DataLoader(val_dataset, batch_size = c.batch_size, shuffle = True, num_workers = c.num_thread)
        net.eval()        
        
        for j, patch_batched in enumerate(valloader):
                           
            img = patch_batched['image'][:,None,...].to(device)
            
            #img = patch_batched['image'].permute(0,4,1,2,3).to(device) #- with generatedeepmedic patches
            seg = patch_batched['seg'].to(device)

            #Combine hippocampus with cortex 
            if num_class ==5:
                seg[seg==5] = 1
            elif num_class == 4:
                seg[seg==4] = 1
#            output,_ = net(img) # Pulkit's code outputs w and /wo soft max
            
            if generate_uncertainty:
                
                output_list = net(img)                
                loss_uncertain = 0
                for output in output_list:
                    loss_uncertain += criterion(output, seg.long())
                loss = loss_uncertain/num_groups
                    
                ouput = torch.mean(torch.stack(output_list), axis = 0)
                uncertainty =  (torch.std(torch.stack(output_list), axis = 0))**2
            else:
                 # forward + backward + optimize
#                 output,_,ds_outputs = net(img)
#                 loss = criterion(output,ds_outputs, seg.long())
#                 output,_= net(img)
                 output,_,ds_outputs = net(img)
                 seg = center_crop(seg,output.shape[2:])
#                 loss = criterion(output,seg.long())
                 loss = criterion(output,ds_outputs, seg.long())
             
            
            pred, probability = generate_prediction(output)
            
            gdsc = computeGeneralizedDSC_patch(probability, seg)
            
            validation_loss += loss.item()
            validation_dsc += gdsc
            
            count += 1
            
            if j % 5 == 4: #print every 5 batches
            #Plot images
                plot_images_to_tfboard(img, seg, output, epoch*len(valloader) + j, 
                                       is_training = False,num_image_to_show = c.num_image_to_show)            

     
        print('Validation loss: epoch %d loss: %.3f' %(epoch +1, validation_loss/count))
        writer.add_scalar('validation_loss', validation_loss/count, epoch + 1)
        writer.add_scalar('validation_accuracy', validation_dsc/count, epoch + 1)
        
    scheduler.step()
    
    #Save the model at the end of every epoch
    checkpoint = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
            }
    
    f_path = model_dir + 'most_recent_checkpoint.pt'
    torch.save(checkpoint, f_path)
    if (validation_loss/count) < best_val_loss:
        best_model_file = model_dir + 'best_model.pth'
        shutil.copyfile(f_path, best_model_file)
        best_val_loss = validation_loss/count
    
        
# when predicting, I need to do softmax and argmax
                
print('Finished Training')
writer.close()


# Run network on validation set and save outputs. DO a dense sampling of patches for the final validation DSC score
pad_size = c.half_patch[0]
gdsc_val = []
with torch.no_grad():
    for i in range(len(image_dataset)):
        print(i)
        sample = image_dataset[i]
        if(sample['type'] == 'test'):
        
            image_id = sample['id']
            print("Generating test patches for ", image_id )
              
            test_patches = p.GeneratePatches(sample, patch_size = c.test_patch_size, is_training = False, transform =False, include_second_chan = include_second_chan)

            testloader = DataLoader(test_patches, batch_size = c.batch_size, shuffle = False, num_workers = c.num_thread)    
            
            image_shape = sample['image'].shape
            affine = sample['affine']
            
            ## For assembling image
            im_shape_pad = [x + pad_size*2 for x in image_shape]
            prob = np.zeros([num_class] + list(im_shape_pad))
            rep = np.zeros([num_class] + list(im_shape_pad))
            uncertainty_map = np.zeros([num_class] + list(im_shape_pad))
            
            pred_list = []
            for j, patch_batched in enumerate(testloader):
                
                    print("batch", j)                
                    img = patch_batched['image'][:,None,...].to(device)
                    seg = patch_batched['seg'].to(device)
                    cpts = patch_batched['cpt']
                    
                    
                    if generate_uncertainty:
                
                        output_list = net(img)                
                        loss_uncertain = 0
                        for output in output_list:
                            loss_uncertain += criterion(output, seg.long())
                        loss = loss_uncertain/num_groups
                            
                        output = torch.stack(output_list)
                        probability_group = F.softmax(output, dim = 2)
                        probability = torch.mean(probability_group, axis = 0).cpu().numpy()
                        uncertainty =  ((torch.std(probability_group, axis = 0))**2).cpu().numpy()
                       # uncertainty = uncertainty.cpu().numpy()
                    else:
                         output, predictions, ds_outputs = net(img)
#                         loss = criterion(output, ds_outputs, seg.long())        
#                         output, predictions = net(img)
#                         seg = center_crop(seg,output.shape[2:])
#                         loss = criterion(output, seg.long()) 
                         probability = predictions.cpu().numpy()
                    
                    #Crop the patch to only use the center part
#                    probability = probability[:,:,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size]
                    #uncertainty = uncertainty[:,:,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size]
                                    
                    ## Assemble image in loop!
                    n, C, hp, wp, dp = probability.shape
        
#                    print(probability.shape)
                    half_shape = torch.tensor([hp, wp,dp])/2
                    half_shape = half_shape.int()
                    
                    hs, ws, ds = half_shape
                    
                    
                    #for cpt, pred, uncert in zip(list(cpts), list(probability), list(uncertainty)):
                  
                    for cpt, pred in zip(list(cpts), list(probability)):
                        #if np.sum(pred)/hs/ws/ds < 0.1:
                        prob[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred
                        rep[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += 1
                        #uncertainty_map[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += uncert
                                    
    #                pred_list.append((probability, cpts))
                    
             #Crop the image since we added padding when generating patches
            prob = prob[:,pad_size:-pad_size, pad_size:-pad_size,pad_size:-pad_size]
            rep = rep[:,pad_size:-pad_size,pad_size:-pad_size,pad_size:-pad_size]
            #uncertainty_map = uncertainty_map[:,pad_size:-pad_size, pad_size:-pad_size,pad_size:-pad_size]
            #rep[rep==0] = 1e-6
        
            # Normalized by repetition
            prob = prob/rep
            #uncertainty_map = uncertainty_map/rep
            
            seg_pred = np.argmax(prob, axis = 0).astype('float')
            prob = np.moveaxis(prob,0,-1)
            #uncertainty_map = np.moveaxis(uncertainty_map,0,-1)
            
            gdsc = computeGeneralizedDSC(sample['seg'], seg_pred)
            print("Prediction accuracy", gdsc)
            gdsc_val.append(gdsc)
            
            nib.save(nib.Nifti1Image(prob, affine), osp.join(output_dir, "prob_" + str(image_id) + ".nii.gz"))
            nib.save(nib.Nifti1Image(seg_pred, affine), osp.join(output_dir, "seg_" + str(image_id)+".nii.gz" ))
    #nib.save(nib.Nifti1Image(uncertainty_map, affine), osp.join(output_dir, "uncertanity_" + str(image_id) + ".nii.gz" ))
            
            print("Done!")
    
    print("Average validation accuracy is ", sum(gdsc_val)/len(gdsc_val))
    print(gdsc_val)
    print("Standard deviation is ", np.std(gdsc_val))
"""
