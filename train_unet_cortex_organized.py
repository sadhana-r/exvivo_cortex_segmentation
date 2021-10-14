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
import util as util

#Try Pulkit's code
#import Unet3D_meta_learning


class BaselineModel:

	def __init__(self):
		
		self.experiment_name = 'Experiment_10012021_dmap_secondchan_250_run2'
		
		self.num_class = 4
		self.generate_uncertainty = False
		self.num_groups = 4 # for MGNet
		self.unet_num_levels = 3
		self.include_second_chan = True   ## Prior flag - include a second input channel for UNet w/ prior
		self.load_model = False
		self.c = config.Config_Unet()
		
		
		# Set up GPU if available    
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.writer = SummaryWriter(tfboard_dir)
		
		
		#Set up data
		#Define image dataset (reads in full images and segmentations)
		image_dataset = p.ImageDataset_withDMap(csv_file = c.train_val_csv)		
		
	def setup_directories(self):
	
		self.dir_names = config.Setup_Directories()
		self.root_dir = dir_names.root_dir
		self.tfboard_dir = dir_names.tfboard_dir + '/' + self.experiment_name
		self.model_dir = dir_names.model_dir + '/' + self.experiment_name + '/'
		self.output_dir = dir_names.valout_dir + '/' + self.experiment_name + '/'
		self.model_file = model_dir + 'model.pth'
		
		# Load model or not. If load model, the modelDir and tfboardDir should have existed. Otherwise they
		# will be created forcefully, wiping off the old one.
		if not os.path.exists(output_dir):
		    os.makedirs(output_dir)

		if not self.load_model:
		    self.c.force_create(model_dir)
		    self.c.force_create(tfboard_dir)
		    
	def configure(self):
		
		alpha = torch.ones(self.unet_num_levels - 1)
		alpha[0] = 0.4 # [0.4,1]
		self.alpha = alpha.cuda()
		
		# Define a loss function and optimizer
		weights = torch.ones(num_class)
		weights[0] = 0
		self.loss_weights = weights.cuda()
		
		#criterion = l.GeneralizedDiceLoss(num_classes=num_class, weight = weights)
		self.criterion = l.DSCLoss_deepsupervision(self.loss_weights, alpha = self.alpha, num_classes = num_class)
		#criterion = torch.nn.CrossEntropyLoss(weights)
		#criterion = l.CELoss_deepsupervision(weights)

		# Optimizer and learning rate
		optimizer = optim.Adam(net.parameters(), lr = c.learning_rate, weight_decay = c.weight_decay)
		scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = c.step_size, gamma = 0.5)
		
		# load unet model
		if not self.load_model:
		
		    #net=Unet3D_meta_learning.Net(num_classes = num_class)
		    if generate_uncertainty:
			net = MGNet(num_class = num_class, num_groups = num_groups)
			net = net.to(self.device)
		    else:
		#        net = UNet(num_class = num_class, padding = False, num_levels = 3, init_feature_number = 64)
			net = UNet_wDeepSupervision(num_class = num_class, patch_size = c.segsize, in_channels = 2,
				                    num_levels = unet_num_levels,init_feature_number = 32, padding = False)
			net = net.to(self.device)
		else:
		    #net=Unet3D_meta_learning.Net(num_classes = num_class)
		    net = UNet_wDeepSupervision(num_class = num_class, patch_size = c.segsize,in_channels = 2,
				                num_levels = unet_num_levels,init_feature_number = 32, padding = False)
		#    net = UNet(num_class = num_class, padding = False, num_levels = 4, init_feature_number = 64)
		    net.load_state_dict(torch.load(model_file, map_location = torch.device(device)))
		    net = net.to(self.device)
		    net.eval()

		print(net)
	
	def generate_patches(self):
	
		# Save and genearte patches for datasets
		tp_dir = self.dir_names.patch_dir + "/training_data"
		vp_dir = self.dir_names.patch_dir + "/validation_data"

		self.c.force_create(tp_dir)
		self.c.force_create(vp_dir)
		if os.path.exists(dir_names.train_patch_csv):
		    os.remove(dir_names.train_patch_csv)
		if os.path.exists(dir_names.val_patch_csv):
		    os.remove(dir_names.val_patch_csv)
		  
		##Train/val split determined in the split_data.csv
		for i in range(len(image_dataset)):
		    sample = image_dataset[i]
		    
		     # If I want to include a prior
		    if include_second_chan:
			dmap = sample['second_chan']
			# randomly distort segmentation to simulate "warped template"       
		    else:
			dmap = None
		     
		    if(sample['type'] == 'train'):
			print(sample['type'])
			_ = p.GeneratePatches(sample, patch_size = self.c.segsize, is_training = True, transform = True, second_chan = dmap)
		    else:
			_ = p.GeneratePatches(sample, patch_size = self.c.segsize, is_training = True, transform = False, second_chan = dmap)

	def train(self):
	
		###Now create a dataset with the patches
		train_dataset = p.PatchDataset(csv_file = dir_names.train_patch_csv)
		val_dataset = p.PatchDataset(csv_file = dir_names.val_patch_csv)
		
		# Training loop
		training_loss = 0.0

		for epoch in range(self.c.num_epochs):
		    
		    
		    trainloader = DataLoader(train_dataset, batch_size = self.c.batch_size, shuffle = True, num_workers = c.num_thread)
		    net.train()
		    for j, patch_batched in enumerate(trainloader,0):
		
			img = patch_batched['image'][:,None,...].to(device)
			img = patch_batched['image'].permute(0,4,1,2,3).to(device)
			seg = patch_batched['seg'].to(device)
		    
		
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
			    util.plot_images_to_tfboard(img, seg, output, epoch*len(trainloader) + j,
				                   is_training = True, num_image_to_show = c.num_image_to_show)            
			    print('Training loss: [epoch %d,  iter %5d] loss: %.3f lr: %.5f' %(epoch +1, j+1, training_loss/5, scheduler.get_lr()[0]))
			    self.writer.add_scalar('training_loss', training_loss/5, epoch*len(trainloader) + j)
			    training_loss = 0.0
			    
		    
		    # At the end of the epoch, run valudation
		    ## Validation    
		    validation_loss = 0
		    validation_dsc = 0
		    count = 0
		    with torch.no_grad():
		
			valloader = DataLoader(val_dataset, batch_size = c.batch_size, shuffle = True, num_workers = c.num_thread)
			net.eval()        
		
			for j, patch_batched in enumerate(valloader):
				           
			    img = patch_batched['image'][:,None,...].to(device)
			    img = patch_batched['image'].permute(0,4,1,2,3).to(device) #- with generatedeepmedic patches
			    seg = patch_batched['seg'].to(device)

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
			     
			    
			    pred, probability = util.generate_prediction(output)
			    
			    gdsc = util.computeGeneralizedDSC_patch(probability, seg)
			    
			    validation_loss += loss.item()
			    validation_dsc += gdsc
			    
			    count += 1
			    
			    if j % 5 == 4: #print every 5 batches
			    #Plot images
				util.plot_images_to_tfboard(img, seg, output, epoch*len(valloader) + j, 
				                       is_training = False,num_image_to_show = c.num_image_to_show)            

		     
			print('Validation loss: epoch %d loss: %.3f' %(epoch +1, validation_loss/count))
			self.writer.add_scalar('validation_loss', validation_loss/count, epoch + 1)
			self.writer.add_scalar('validation_accuracy', validation_dsc/count, epoch + 1)
		
		    scheduler.step()
		    
		    #Save the model at the end of every epoch
		    model_file = model_dir + 'model_' + str(epoch + 1) + '.pth'
		    torch.save(net.state_dict(), model_file)
		
		# when predicting, I need to do softmax and argmax
				
		print('Finished Training')

		#Save the model
		model_file = model_dir + 'model.pth'
		torch.save(net.state_dict(), model_file)

		self.writer.close()

	
	def test(self):
		  
		# Run network on validation set and save outputs. Do a dense sampling of patches for the final validation DSC score
		pad_size = c.half_patch[0]
		gdsc_val = []
		with torch.no_grad():
		    for i in range(len(image_dataset)):
			print(i)
			sample = image_dataset[i]
			if(sample['type'] == 'test'):
		
			    image_id = sample['id']
			    print("Generating test patches for ", image_id )
			    
			    if include_second_chan:
				dmap = sample['second_chan']
				test_patches = p.GeneratePatches(sample, patch_size = c.test_patch_size, is_training = False, transform =False, second_chan = dmap)
			    else:
				test_patches = p.GeneratePatches(sample, is_training = False, transform =False, second_chan = None)     
			    
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
				    img = patch_batched['image'][:,None,...].to(device).squeeze(1)
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
		                    probability = probability[:,:,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size]
				    #uncertainty = uncertainty[:,:,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size,c.patch_crop_size:-c.patch_crop_size]
				                    
				    ## Assemble image in loop!
				    n, C, hp, wp, dp = probability.shape
		#                    print(probability.shape)
				    half_shape = torch.tensor([hp, wp,dp])/2
		    #                half_shape = half_shape.astype(int)
				    hs, ws, ds = half_shape
				    
				    #for cpt, pred, uncert in zip(list(cpts), list(probability), list(uncertainty)):
				    for cpt, pred in zip(list(cpts), list(probability)):
		#                        print(cpt)
		#                        print(pred.shape)
				        #if np.sum(pred)/hs/ws/ds < 0.1:
				        prob[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred
				        rep[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += 1
				        #uncertainty_map[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += uncert
				                    
		    #                pred_list.append((probability, cpts))
				    
			     #Crop the image since we added padding when generating patches
			    prob = prob[:,pad_size:-pad_size, pad_size:-pad_size,pad_size:-pad_size]
			    rep = rep[:,pad_size:-pad_size,pad_size:-pad_size,pad_size:-pad_size]
			    #uncertainty_map = uncertainty_map[:,pad_size:-pad_size, pad_size:-pad_size,pad_size:-pad_size]
			    rep[rep==0] = 1e-6
		
			    # Normalized by repetition
			    prob = prob/rep
			    #uncertainty_map = uncertainty_map/rep
			    
			    seg_pred = np.argmax(prob, axis = 0).astype('float')
			    prob = np.moveaxis(prob,0,-1)
			    #uncertainty_map = np.moveaxis(uncertainty_map,0,-1)
			    
			    gdsc = util.computeGeneralizedDSC(sample['seg'], seg_pred)
			    print("Prediction accuracy", gdsc)
			    gdsc_val.append(gdsc)
			    
			    nib.save(nib.Nifti1Image(prob, affine), osp.join(output_dir, "prob_" + str(image_id) + ".nii.gz"))
			    nib.save(nib.Nifti1Image(seg_pred, affine), osp.join(output_dir, "seg_" + str(image_id)+".nii.gz" ))
		    	    #nib.save(nib.Nifti1Image(uncertainty_map, affine), osp.join(output_dir, "uncertanity_" + str(image_id) + ".nii.gz" ))
			    
			    print("Done!")
		    
		    print("Average validation accuracy is ", sum(gdsc_val)/len(gdsc_val))
		    print(gdsc_val)
		    print("Standard deviation is ", np.std(gdsc_val))
