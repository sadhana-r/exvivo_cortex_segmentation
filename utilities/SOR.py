#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:35:47 2021

@author: sadhana-ravikumar
"""
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib

class SuccessiveOverRelaxation(nn.Module):

    def __init__(self,source, sink,threshold = 0.05, w = 1.5):
        super(SuccessiveOverRelaxation, self).__init__()

        """
        w is the over-relaxation parameter
        """
        self.wopt = w
        self.thresh = threshold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.source_label = source
        self.sink_label = sink

    def ravel_index(self,index, shape):

        [dim1,dim2,dim3] = shape
        out = []
        for i in range(index.shape[1]):
            out.append((dim2*dim3)*index[0,i] +
            dim3*index[1,i] + index[2,i])

        return torch.stack(out)

    def forward(self, image):

        """
        image should be of size NxCxHxDxW where C is 1 and N can be greater than 1
        solver iterates through each image in the batch - is there a more efficient implementation? Concatenating between batches
        may result in computations across batches?
        """
        
        # Concatenate batches
        img_list = list(image.squeeze())
        bs = len(img_list)
        img_list_padded = []
        
        # Pad each image for boundaries
        for img in img_list:
            im_shape_pad = [x + 2 for x in img.shape]
            image_pad = torch.zeros(im_shape_pad).to(self.device)
            image_pad[1:-1,1:-1,1:-1] = img
            img_list_padded.append(image_pad)
            
        image = torch.cat(img_list_padded, axis = 0)
        
        h,w,d = image.shape

        init = torch.zeros(image.shape).to(self.device)

        #source
        init[image == self.source_label ] = 0
        #sink
        init[image == self.sink_label] = 1
        init_ravel = torch.flatten(init)


        #over-relaxation parameter
        min_dim = torch.min(torch.tensor(image.shape)).type(torch.float32)
        self.wopt = 2/(1+(np.pi/min_dim))

        # Black-red coordinates
        black = torch.zeros([h,w,d])
        red = torch.zeros([h,w,d])

        xx,yy,zz = torch.meshgrid( torch.arange(0,h), torch.arange(0,w), torch.arange(0,d))
        coords = xx + yy + zz
        coords = coords.to(self.device)
        black[(torch.fmod(coords,2) == 0) * (image == 1)] = 1
        red[(torch.fmod(coords,2) == 1) * (image == 1)] = 1

        idx_black_gm = torch.stack(torch.where(black == 1))
        idx_red_gm = torch.stack(torch.where(red == 1))


        if ((idx_black_gm.numel() > 10) and (idx_red_gm.numel() > 10)):

            black_ravel = self.ravel_index(idx_black_gm, image.shape)
            red_ravel = self.ravel_index(idx_red_gm, image.shape)

            gauss_seidel = torch.zeros(init_ravel.shape).to(self.device)
            sor_adjustment = torch.zeros(init_ravel.shape).to(self.device)
            iterations = 0
            delta_v = 0

            while(True):

                #black
                gauss_seidel[black_ravel] = (init_ravel[black_ravel - 1]+ init_ravel[black_ravel + 1] +  init_ravel[black_ravel + d] +  init_ravel[black_ravel - d] +  init_ravel[black_ravel + d*w] +  init_ravel[black_ravel - d*w] - 6*init_ravel[black_ravel])/6
                sor_adjustment[black_ravel] =self.wopt * gauss_seidel[black_ravel]
                init_ravel[black_ravel] = sor_adjustment[black_ravel] + init_ravel[black_ravel]

#                print("black ravek adj")
#                print(torch.sum(torch.abs(sor_adjustment[black_ravel])))
                delta_v += torch.sum(torch.abs(sor_adjustment[black_ravel]))

                #red
                gauss_seidel[red_ravel] = (init_ravel[red_ravel - 1]+ init_ravel[red_ravel + 1] +  init_ravel[red_ravel + d] +
                             init_ravel[red_ravel - d] +  init_ravel[red_ravel + d*w] +  init_ravel[red_ravel - d*w] - 6*init_ravel[red_ravel])/6
                sor_adjustment[red_ravel] = self.wopt * gauss_seidel[red_ravel]
                init_ravel[red_ravel] = sor_adjustment[red_ravel] + init_ravel[red_ravel]

                delta_v += torch.sum(torch.abs(sor_adjustment[red_ravel]))

                iterations += 1

                if delta_v < self.thresh:
#                    print(delta_v)
#                    print('breaking below thresh')
                    break
                elif iterations > 500:
#                    print('breaking num iter')
                    break
                else:
                    delta_v = 0  # Restart counting delta_v for the next iteration

                laplace_sol = init_ravel.view(image.shape)
                
                #Convert back into original input shape of N x C x H x D x W and remove padding
                chunk_size = (int(h/bs),)*bs
                laplace_list = torch.split(laplace_sol,chunk_size, 0)
                laplace_cropped = [torch.unsqueeze(sol[1:-1,1:-1,1:-1],0) for sol in laplace_list]
                laplace_cropped = torch.stack(laplace_cropped)

            return laplace_cropped, iterations, delta_v

        else:
#            self.iterations = 0
            init = init[1:-1,1:-1,1:-1]
            return init,None, None


sor = SuccessiveOverRelaxation(source = 2, sink = 2)
file = nib.load('/Users/sravikumar/Box Sync/PennPhD/Research/PICSL/exvivo_mtl_unet/inputs/HNL29_18-L/seg_patch.nii.gz')
img = file.get_fdata().astype(np.float32)
img = img[np.newaxis,:]
img_batch = np.stack([img, img, img], axis = 0)

img_batch = torch.tensor(img_batch)
laplace_cropped,_,_ = sor(img_batch)
laplace_sol = laplace_cropped[0,0,:,:,:]
nib.save(nib.Nifti1Image(laplace_sol.numpy(), file.affine),'/Users/sravikumar/Box Sync/PennPhD/Research/PICSL/exvivo_mtl_unet/test_SOR_torch_batch.nii.gz')

