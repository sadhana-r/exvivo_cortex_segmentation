# This scripts houses some usual config and constants used in the network
import os
import shutil
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Setup_Directories(object):
    def __init__(self):
        ## Directories and training image generation
        # Root directory that house the image
        self.root_dir = "/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet"
        
        # Directory that contain CNN related files
        self.code = "/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_segmentation"
                
        # csv file that store train/test split
        self.train_val_csv = self.root_dir + "/data_csv/split.csv"
#        self.final_test_csv = self.root_dir + "/data_csv/split_test.csv"
        #self.final_test_csv = "/home/sadhana-ravikumar/Documents/Sadhana/n4bias_dots/dots_csv.csv"
        #self.final_test_csv = self.root_dir + "/data_csv/test_pulkit.csv"
        
        # Patch directories
        
        self.patch_dir = self.root_dir + "/patch_data"
        self.train_patch_csv = self.root_dir + "/data_csv/train_patch.csv"
        self.val_patch_csv = self.root_dir + "/data_csv/val_patch.csv"
        
        # Directories that contain the model
        self.model_dir = self.root_dir + "/model"
        
        # Directories that contain the tensorboard output
        self.tfboard_dir = self.root_dir + "/tfboard"
        
        # Directories that store the validation output
        self.valout_dir = self.root_dir + "/validation_output"
        
        # Directories that store the validation output
        self.test_dir = self.root_dir + "/mtl_pulkit"
        #self.test_dir = "/home/sadhana-ravikumar/Documents/Sadhana/n4bias_dots/initial_test_set/cortex_segmentation"
                
    def force_create(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)    

        
class Config_DistanceMapUnet(Setup_Directories):
    def __init__(self):
        # Set base init
        super().__init__()
        
        # Number image per tfrecord in train and test set
#        self.nTrainPerTfrecord = 10
#        self.nTestPerTfrecord = 10
        
        # Training patch params
        self.num_pos = 250
        self.num_neg = 0 #30 # try switching number of pos and neg
        self.aug = 50
        self.num_thread = 8
        
        # Multi resolution patch size and spacing setting
        self.patchsize_multi_res = [(1, (48, 48, 48))]
        self.segsize = (96,96,96) 
        self.test_patch_size = (96,96,96)
        self.half_patch = np.ceil((np.array(self.segsize) - 1) / 2).astype(np.int32)
        self.test_patch_spacing = (32,32,32) #was 16 and 32 for 96 (for 96 input, must be less than 56 output)
        self.patch_crop_size = 4
        
        ## Learning parameters
        self.batch_size = 3 #15 for training. Make it 10 for testing
        self.shuffle_buffer = 100
        self.learning_rate = 1e-4#1e-2?
        self.step_size = 10
        self.weight_decay = 0.0005
        
        ## Training parameters 
        self.num_epochs = 15 
        
        # Visualization params
        self.num_image_to_show = 2
        
        
        # num batch for validation
        self.batch_validation = 1
        
class Config_BaselineUnet(Setup_Directories):
    def __init__(self):
        # Set base init
        super().__init__()
        
        # Training patch params
        self.num_pos = 250
        self.num_neg = 0 #30 # try switching number of pos and neg
        self.aug = 50
        self.num_thread = 8
        
        # Multi resolution patch size and spacing setting
        self.patchsize_multi_res = [(1, (48, 48, 48))]
        self.segsize = (96,96,96) 
        self.test_patch_size = (96,96,96)
        self.half_patch = np.ceil((np.array(self.segsize) - 1) / 2).astype(np.int32)
        self.test_patch_spacing = (32,32,32) #was 16 and 32 for 96 (for 96 input, must be less than 56 output)
        self.patch_crop_size = 4
        
        ## Learning parameters
        self.batch_size = 4
        self.shuffle_buffer = 100
        self.learning_rate = 1e-4
        self.step_size = 10
        self.weight_decay = 0.0005
        
        ## Training parameters 
        self.num_epochs =65
        
        # Visualization params
        self.num_image_to_show = 3
        
        
        # num batch for validation
        self.batch_validation = 1
