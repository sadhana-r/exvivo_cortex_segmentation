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

class Successive_over_relaxation(nn.Module):

    def __init__(self,threshold = 0.05, w = 1.5):
        super(Successive_over_relaxation, self).__init__()
        
        """
        w is the over-relaxation parameter
        """
        self.wopt = w
        self.thresh = threshold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def ravel_index(self,index, shape):
        
        [dim1,dim2,dim3] = shape
        out = []
        for i in range(index.shape[1]):
            out.append((dim2*dim3)*index[0,i] + 
            dim3*index[1,i] + index[2,i])
        
        return torch.stack(out)
    
    def forward(self, image, source_label, sink_label):
                
        #reflection_pad = nn.ReflectionPad3d(2)
        im_shape_pad = [x + 2 for x in image.shape]
        image_pad = torch.zeros(im_shape_pad).to(self.device)
        image_pad[1:-1,1:-1,1:-1] = image
        image = image_pad
        #image = reflection_pad(image)
        
        h,w,d = image.shape
        
        init = torch.zeros(image.shape).to(self.device)

        #source
        init[image == source_label ] = 0
        #sink
        init[image == sink_label] = 1
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
#        
#        print(idx_black_gm.numel())
#        print(idx_red_gm.numel())
        
        if ((idx_black_gm.numel() > 10) and (idx_red_gm.numel() > 10)):
            
#            print(idx_black_gm.numel())
#            print(idx_red_gm.numel())
            black_ravel = self.ravel_index(idx_black_gm, image.shape)
            red_ravel = self.ravel_index(idx_red_gm, image.shape)

            gauss_seidel = torch.zeros(init_ravel.shape).to(self.device)
            sor_adjustment = torch.zeros(init_ravel.shape).to(self.device)
            iterations = 0
            delta_v = 0
            
            while(True):
                
                #black
                gauss_seidel[black_ravel] = (init_ravel[black_ravel - 1]+ init_ravel[black_ravel + 1] +  init_ravel[black_ravel + h] +  init_ravel[black_ravel - h] +  init_ravel[black_ravel + h*w] +  init_ravel[black_ravel - h*w] - 6*init_ravel[black_ravel])/6
                sor_adjustment[black_ravel] =self.wopt * gauss_seidel[black_ravel]
                init_ravel[black_ravel] = sor_adjustment[black_ravel] + init_ravel[black_ravel]
             
#                print("black ravek adj")
#                print(torch.sum(torch.abs(sor_adjustment[black_ravel])))
                delta_v += torch.sum(torch.abs(sor_adjustment[black_ravel]))
                
                #red
                gauss_seidel[red_ravel] = (init_ravel[red_ravel - 1]+ init_ravel[red_ravel + 1] +  init_ravel[red_ravel + h] +  
                             init_ravel[red_ravel - h] +  init_ravel[red_ravel + h*w] +  init_ravel[red_ravel - h*w] - 6*init_ravel[red_ravel])/6
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
                laplace_sol = laplace_sol[1:-1,1:-1,1:-1]   

            return laplace_sol, iterations, delta_v
        
        else:
#            self.iterations = 0
            init = init[2:-2,2:-2,2:-2] 
            init = init[1:-1,1:-1,1:-1] 
            return init,None, None
            
        
sor = Successive_over_relaxation()
file = nib.load('/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/patch_data/training_data/seg_1115_34.nii.gz')
img = file.get_fdata().astype(np.float32)
img = torch.tensor(img)
laplace_sol,_,_ = sor(img,2,3)
nib.save(nib.Nifti1Image(laplace_sol.cpu().numpy(), file.affine),'/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/inputs/test115_34_SOR_torch_reflect.nii.gz')

            
              