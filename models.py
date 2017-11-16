import math
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, Conv2D, Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

# Common hyper paremeters
LENGTH_OF_SIDE = 488

# Genarator hyper parameters
G_FRAMES = 4

# Discriminator hyper parameters
D_FRAMES = 5


def generator():
    model = Sequential([
        Conv3D(64,          #filters
               kernel_size=(4, 4, 4),
               strides=(2, 2, 2),
               padding='same',
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        Dense(units=10, input_dim=28),
        Activation('softmax')
    ])

    return model

def discriminator():
    model = Sequential([
        Conv3D(64,  # filters
               kernel_size=(4, 4, 5),
               strides=(2, 2, 2),
               padding='same',
               input_shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        LeakyReLU(0.2),


    ])

    return model