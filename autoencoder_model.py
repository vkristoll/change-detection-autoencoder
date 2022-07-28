#Define the autoencoder model

#Import libraries
import tensorflow as tf
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam

#Create function to define the autoencoder model
def autoencoder():
    #Define the patch size
    original_img_size = (224, 224, 4)   
    
    #encoder
    input_img = Input(shape=original_img_size)
    x = Conv2D(64, (3, 3), padding='same')(input_img) 
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)        
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    encoded = Activation('relu')(x)
    
    #decoder
    x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(encoded)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(x)
    x = Activation('relu')(x)
    x= Conv2D(4, (3, 3), padding='same')(x) 
    decoded = Activation('sigmoid')(x)

    #Define the model
    model = Model(input_img, decoded)   
    
    #Compile the model    
    model.compile(optimizer='adam', loss='mean_squared_error') 
    
    return model

#Define the autoencoder model
model=autoencoder()
