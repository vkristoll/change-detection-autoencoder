#Creating training data

#Import libraries
import os
import numpy as np
from osgeo import gdal

'''Define path to 4 folders that contain first date patches (224x224x4 px) for the 4 study areas (Granada, Rhodes, Venice, Tønsberg).
The original 16 bit images have been previously converted to 8 bit. They contain R, G, B, NIR bands'''
#Create list with the names of the patches
dir1_path="folder path for Granada patches"
dir1_list=os.listdir(dir1_path)

dir2_path="folder path for Rhodes patches"
dir2_list=os.listdir(dir2_path)

dir3_path="folder path for Venice patches"
dir3_list=os.listdir(dir3_path)

dir4_path="folder path for Tønsberg patches"
dir4_list=os.listdir(dir4_path)

#Create list that contains the paths of the 4 folders
dir_path_list=[dir1_path,dir2_path,dir3_path,dir4_path]

#Create list that contains the 4 lists of the names of the patches
dir_sorted_list=[sorted(dir1_list),sorted(dir2_list), sorted(dir3_list), sorted(dir4_list)]

#Create list that stores the number of files (patches) in each folder
number_files_list=[len(dir1_list),len(dir2_list),len(dir3_list),len(dir4_list)]

#Create list that contains float arrays to store the patches for each study area
train_list=[np.float16(np.zeros((len(dir1_list),224,224,4))),np.float16(np.zeros((len(dir2_list),224,224,4))),
            np.float16(np.zeros((len(dir3_list),224,224,4))),np.float16(np.zeros((len(dir4_list),224,224,4)))]

#Add the patches to the train_list
for i in range (4):
    os.chdir(dir_path_list[i])
    for k in range(0, number_files_list[i],1):        
        #Print the index of the patch
        print(f"x_train {i}: k is {k}")
        #Read the patch 
        x_1 = gdal.Open(dir_sorted_list[i][k])
        #Convert it to an array
        x_1 = np.array(x_1.ReadAsArray())
        #Change the order of the axes to width,height,channels
        x_1 = x_1.transpose(1,2,0)    
        #Convert values to [0,1]
        x_1 = x_1/255 
        #Add the patch to the train_list
        train_list[i][k,:,:,:]=x_1