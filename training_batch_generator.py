#create a function that generates batches of training data

#Import libraries
import numpy as np
from numpy.random import randint
import random

#Import required variables from previous steps
#dir1_list, dir2_list, dir3_list, dir4_list, train_list are defined in "create_training_data.py"

from create_training_data import dir1_list, dir2_list, dir3_list, dir4_list, train_list

def batch_generator():
    
    #Define batch size
    batch_size=8
    #Select randomly one of the 4 study areas        
    randomindexim=randint(4)  
    
    #Select random sample of indexes equal to the batch size from the randomly selected study area
    if randomindexim==0:
       randomindexsample=random.sample(range(0,len(dir1_list)),batch_size)
       
    elif  randomindexim==1:
       randomindexsample=random.sample(range(0,len(dir2_list)),batch_size) 
       
    elif  randomindexim==2:
       randomindexsample=random.sample(range(0,len(dir3_list)),batch_size) 
       
    elif  randomindexim==3:
       randomindexsample=random.sample(range(0,len(dir4_list)),batch_size) 
       
    #Create array with patches equal to the batch size      
    X=np.float16(np.zeros((batch_size,224,224,4)))    
    c=0
    for i in range(0,batch_size,1):       
        c=c+1
        X[c-1,:,:,:]=train_list[randomindexim][randomindexsample[i],:,:,:]  
        
    # Return training batch. It's the same for the input and the output because the model is an autoencoder.
    return X,X   