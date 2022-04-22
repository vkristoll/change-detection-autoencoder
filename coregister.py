#Co-registration of images
#This script can be used to co-register images when the x,y coordinates of at least 3 matching points are previously known

#Importing libraries
import numpy as np
import os
import affine6p
import tifffile as tiff
from skimage import transform

#Define path to the folder with images of first or the second date 
#Create a sorted list of the files
dir1_path="folder with images of first or second date"
dir1_list=os.listdir(dir1_path)
dir1_sorted_list=sorted(dir1_list)

#Define path to the folder with .txt files containing the x,y coordinates of matching points
#Create a sorted list of the files
dir2_path="folder with .txt files containing the x,y coordinates of matching points (e.g. date1 (x,y) \t date2 (x,y) -- 4 columns). One file for each image"
dir2_list=os.listdir(dir2_path)
dir2_sorted_list=sorted(dir2_list)

#Count the number of files
number_files=len(dir1_path)

#Iterate over the folders of images and points and create a new co-registered image
for i in range(number_files):
    
    #Print the index of the file
    print(" The file index is %s" %i) 
    #Access the directory of the images    
    os.chdir(dir1_path)  
    #Read image    
    im=tiff.imread(dir1_sorted_list[i])
    
    #Store the file name and define the name of the co-registered image
    filename=os.path.basename(dir1_sorted_list[i])
    filename2=filename.split('.')[0] + "_affine.tiff"
    
    #Access the directory of the .txt coordinate files
    os.chdir(dir2_path)
    #Open and read the file
    f = open(dir2_sorted_list[i], "r")
    f2=f.readlines() 
    #Count the number of matching points
    length=len(f2)
    
    #Create array with rows equal to number of points and 4 columns
    A=np.zeros((length,4)).astype(int)
    #Fill the array with the x,y coordinates read from the .txt file
    for i in range(length):
       for j in range (4):    
          f2b=f2[i].split()
          f2c=f2b[j].split(".")     
          A[i,j]= int(f2c[0])

    #Define 2 lists to store the x.y cordinates for the 2 dates
    base=[]
    warp=[]

    for i in range(length):
        base.append(list(A[i,0:2]))
        warp.append(list(A[i,2:4])) 

    #Calculate the parameters of the affine transformation 
    trans = affine6p.estimate(base, warp)    
    transmatrix=np.array(trans.get_matrix())
    tform=transform.AffineTransform(transmatrix)

    #Apply the transformation on the image
    tf_img = transform.warp(im, tform.inverse, preserve_range=True)

    tiff.imsave(filename2,tf_img)

       
       



    


