#%% This script splits the dataset and saves the split into a csv file. This should be run before the 
# preprocessing script get run so that the split be read by the preprocessing script
import numpy as np
import csv
import os
import config
c = config.Config()

#%%
# This function gives a list of train and test directory, and optionally save the 
# train and test into a csv file
def divide_train_test(rootDir, trainRatio = 0.7, outDir = None):
    # Divide train and test set
    trainDirs = []
    testDirs = []
    
    diseases = os.listdir(rootDir)
    
    for d in diseases:
        diseaseDir = rootDir + d + "/"
        patientDirs = [ (diseaseDir + p + "/", p, d) for p in os.listdir(diseaseDir)]
        
        num_patients = len(patientDirs)
        idx = np.random.permutation(num_patients)
        
        trainDirs.extend([patientDirs[i] for i in idx[:int(trainRatio*num_patients)]])
        testDirs.extend([patientDirs[i] for i in idx[int(trainRatio*num_patients):]])
    
    # Write out the train test split
    if outDir is not None:
        with open(outDir, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for tr in trainDirs:
                writer.writerow(list(tr) + ['train'])
            for tr in testDirs:
                writer.writerow(list(tr) + ['test'])
    return trainDirs, testDirs

# This function read in the split file created by divide_train_test
def read_split(splitDir = c.train_val_csv):
    trainDirs = []
    testDirs = []
    
    with open(splitDir, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            if row[4] == "train":
                trainDirs.append(row[:4])
            elif row[4] == "test":
                testDirs.append(row[:4])
    return trainDirs, testDirs

# This function read in the split file created by divide_train_test and create one output
def read_split_one(splitDir = c.train_val_csv):
    allDirs = []
    
    with open(splitDir, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            allDirs.append(row[:5])
    return allDirs


def main():
    divide_train_test(c.rootDir, trainRatio = c.trainRatio, outDir = c.train_test_csv)
    trainDirs, testDirs = read_split(splitDir = c.train_test_csv)

if __name__ == "__main__":
    main()
