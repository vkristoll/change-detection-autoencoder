#Creating the change maps

#Import libraries
import os
import numpy as np
import tifffile as tiff
from osgeo import gdal
import tensorflow as tf
from keras.models import Model

#Import required variables from previous steps
#The autoencoder model is  defined in "autoencoder_model.py"
from autoencoder_model import model

#Load the weights
#The autoencoder model is  defined in "autoencoder_training.py"
model.load_weights('weights.h5')  

#Run the line below to find the indexes of the layers 
layer_names = [layer.name for layer in model.layers]

#Define submodels that calculate the model predictions for the selected indexes
model1 = Model(inputs=model.inputs, outputs=model.layers[3].output) 
model2 = Model(inputs=model.inputs, outputs=model.layers[6].output) 
model3 = Model(inputs=model.inputs, outputs=model.layers[10].output) 
model4 = Model(inputs=model.inputs, outputs=model.layers[12].output) 

#Define function that resizes the multilevel feature maps and concatenates them
def feat(x_train):   	
    
    feat1 = model1.predict(x_train)
    feat2 = model2.predict(x_train)
    feat3 = model3.predict(x_train)
    feat4 = model4.predict(x_train)
    
    x1 = tf.image.resize(feat1,[128,128])
    x2 = tf.image.resize(feat2,[128,128])
    x3 = tf.image.resize(feat3,[128,128])
    x4 = tf.image.resize(feat4,[128,128])    
    
    F = tf.concat([x2,x1,x4,x3],3) 
    return F

#Define paths to 2 folders that contain the patches of the 1st and 2nd date (224x224x4 px)
#Create sorted list of the files
dir1_path="folder path for the 1st date patches"
dir1_list=os.listdir(dir1_path)
dir1_sorted_list=sorted(dir1_list)

dir2_path="folder path for the 2nd date patches"
dir2_list=os.listdir(dir2_path)
dir2_sorted_list=sorted(dir2_list)

#Define path to folder to store the change maps
dir3_path="folder path for the change maps"

#Count number of files
number_files=len(dir1_list)

#Iterate over the files and calculate the change map. The output is a file contains float values denoting the distance of the feature values.
# The Otsu threshold can be later applied for a binary output
for i in range(0,number_files,1):   
    
    #Print index of the file
    print(f" i is {i}") 
    
    #Access the first date folder
    os.chdir(dir1_path) 
    #Read the first date image
    im1 = gdal.Open(dir1_sorted_list[i])
    im1=np.array(im1.ReadAsArray()) 
    #Change the order of the axes to width,height,channels
    im1=im1.transpose(1,2,0) 
    #Convert values to [0,1]
    im1=im1/255 
    #Expand the dimensions so that the model can read it (batch_size=1)
    im1=np.expand_dims(im1,0)    
    
    #Store the file name and define name of the change map
    filename=os.path.basename(dir1_sorted_list[i])
    filename2=filename.split('.')[0] + "change.tiff" 
    
    os.chdir(dir2_path)
    im2 = gdal.Open(dir2_sorted_list[i])
    im2=np.array(im2.ReadAsArray()) 
    im2=im2.transpose(1,2,0)
    im2= im2/255 
    im2=np.expand_dims(im2,0)    

    F1=feat(im1) #Features from image patch 1
    #Calculate the square value of the feature maps
    F1=tf.square(F1)
    F2=feat(im2) #Features from image patch 2
    F2=tf.square(F2)
    #Subtract the feature maps from the 2 dates
    d=tf.subtract(F1,F2)
    d=tf.square(d) 
    #Create the change map by summing values from the feature maps 
    d=tf.reduce_sum(d,axis=3) 
    
    #Resize the change map
    dis=d.numpy()    
    dis=dis.transpose(1,2,0)        
    dis = tf.image.resize(dis,[224,224], method="nearest")
    dis=dis.numpy()    
    dis=np.resize(dis,[224,224])  

    #Save the change map
    os.chdir(dir3_path)    
    tiff.imsave(filename2,dis)
    
    
