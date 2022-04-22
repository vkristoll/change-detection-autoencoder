#Training the autoencoder

#Import libraries
import tensorflow as tf
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics

#Import required variables from previous steps
#dir1_list, dir2_list, dir3_list, dir4_list are defined in "create_training_data.py"
#batch_generator is defined in "training_batch_generator.py"
#model is defined in "autoencoder_model.py"
from create_training_data import dir1_list, dir2_list, dir3_list, dir4_list
from training_batch_generator import batch_generator
from autoencoder_model import model

#Define the number of epochs
epochs=400

#Define the number of training steps
#dir1_list, dir2_list, dir3_list, dir4_list are defined in "create_training_data.py"
train_steps=int((len(dir1_list)+ len(dir2_list) + len(dir3_list) + len(dir4_list))/16)

for i in range (epochs):
    for j in range (train_steps):
        #batch_generator is defined in "training_batch_generator.py"
        X,y = batch_generator()  
        loss=model.train_on_batch(X,y)

        print('>%d, %d/%d, d=%.3f' % (i+1, j+1, train_steps, loss))

#Save the weights   
model.save_weights('weights.h5')

      








