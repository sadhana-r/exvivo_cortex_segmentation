#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:13:33 2021

@author: sadhana-ravikumar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:13:30 2021

@author: sadhana-ravikumar
"""

import nibabel as nib
import numpy as np

file = nib.load('/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/inputs/HNL29_18-L/seg_patch.nii.gz')
img = file.get_fdata().astype(np.float32)

#test github commit

#Image shape
img = np.pad(img, [(1,1),(1,1),(1,1)])
h,w,d = img.shape
init = np.zeros(img.shape)
#init[:] = np.nan
#source
init[img == 2 ] = 0
#sink
init[img ==3] = 1

init_ravel = np.ravel(init)

#indices of gray matter
#idxgm = np.where(img == 1)

w_opt = 2/(1+(np.pi/np.min([h,w,d])))
print(w_opt)
error_threshold = 0.001
gauss_seidel = np.zeros(img.shape)
iterations = 0
delta_v = 0

black = np.zeros([h,w,d])
red = np.zeros([h,w,d])
xdim = np.arange(0,h)
ydim = np.arange(0,w)
zdim = np.arange(0,d)
xx,yy,zz = np.meshgrid(xdim, ydim, zdim)
coords = xx + yy + zz
black[(np.mod(coords,2) == 0) * (img == 1)] = 1
red[(np.mod(coords,2) == 1) * (img == 1)] = 1

idx_black_gm = np.where(black == 1)
idx_red_gm = np.where(red == 1)
            
black_ravel = np.ravel_multi_index(idx_black_gm, img.shape)
red_ravel = np.ravel_multi_index(idx_red_gm, img.shape)

gauss_seidel = np.zeros(init_ravel.shape)
sor_adjustment = np.zeros(init_ravel.shape)         
  
while(True):
    print(iterations)
    
    #black
    gauss_seidel[black_ravel] = (init_ravel[black_ravel - 1]+ init_ravel[black_ravel + 1] +  init_ravel[black_ravel + h] +  init_ravel[black_ravel - h] +  init_ravel[black_ravel + h*w] +  init_ravel[black_ravel - h*w] - 6*init_ravel[black_ravel])/6
    sor_adjustment[black_ravel] = w_opt * gauss_seidel[black_ravel]
    init_ravel[black_ravel] = sor_adjustment[black_ravel] + init_ravel[black_ravel]
 
    delta_v += np.sum(abs(sor_adjustment[black_ravel]))
    
    #red
    gauss_seidel[red_ravel] = (init_ravel[red_ravel - 1]+ init_ravel[red_ravel + 1] +  init_ravel[red_ravel + h] +  
                 init_ravel[red_ravel - h] +  init_ravel[red_ravel + h*w] +  init_ravel[red_ravel - h*w] - 6*init_ravel[red_ravel])/6
    sor_adjustment[red_ravel] = w_opt * gauss_seidel[red_ravel]
    init_ravel[red_ravel] = sor_adjustment[red_ravel] + init_ravel[red_ravel]
    delta_v += np.sum(abs(sor_adjustment[red_ravel]))        
    
    iterations += 1
    if delta_v < error_threshold:
        break
    elif iterations > 1000:
        break
    else:
        delta_v = 0  # Restart counting delta_v for the next iteration 

nib.save(nib.Nifti1Image(np.reshape(init_ravel, img.shape), file.affine),'/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet/inputs/HNL29_18-L/test_SOR.nii.gz')
                
