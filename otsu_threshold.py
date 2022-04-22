#Applying Otsu segmentation

#Import libraries
import cv2
import numpy as np
import os
from osgeo import gdal
from skimage import filters

#Define the folder path that contains the change maps with the float values
#Create list of the files
dir1_path="folder path with change maps (float values) "
dir1_list=os.listdir(dir1_path)
#Define folder path to store the binary change maps
dir2_path="folder path to the binary change maps"

#Count the number of the files
number_files=len(dir1_list)

#Iterate over the files and create the binary change maps
for i in range (number_files):
    
    #Print the index of the file
    print(" The file index is %s" %i)
    
    #Access the directory of the change maps (float)
    os.chdir(dir1_path)   
    #Read image   
    im1 = gdal.Open(dir1_list[i])
    im1=np.array(im1.ReadAsArray())
    
    #Store the file name and define the name of the binary change map
    filename=os.path.basename(dir1_list[i])
    filename2=filename.split('.')[0] + "_otsu.png"
    
    #Access the directory of the binary change maps
    os.chdir(dir2_path)
    
    #Calculate the Otsu threshold
    if im1.min() == 0 and im1.max()==0:
        val=0
    else:
        val = filters.threshold_otsu(im1[:,:])  
        
    #Create array to store the binary values   
    A=np.uint8(np.zeros((1120,1120)))

    for i1 in range(1120):
        for i2 in range(1120):
            if im1[i1,i2]>val:
                A[i1,i2]=255
            else:
                A[i1,i2]=0
                
    cv2.imwrite(filename2, A)
    
    

    