import math
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv3D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

# Common hyper paremeters
LENGTH_OF_SIDE = 488

# Genarator hyper parameters
G_FRAMES = 4

# Discriminator hyper parameters
D_FRAMES = 5


def make_generator():
    model = Sequential([
        # The 5 following convolutions are from
        # https://www.semanticscholar.org/paper/Generating-the-Future-with-Adversarial-Transformer-Vondrick-Torralba/263d82ed38ddf2940c64ef74de3cdcaf76f1082d
        Conv3D(32,          #filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 1, 1),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        Conv3D(64,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 2, 2),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        Conv3D(128,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 4, 4),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        Conv3D(256,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 8, 8),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        Conv3D(32,  # filters
               kernel_size=(G_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='same',
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),

        # followings are original
        LeakyReLU(0.2),
        Flatten(),
        Dense(61*61*128),
        BatchNormalization(),
        Activation('relu'),
        Reshape((61, 61, 128), input_shape=(61*61*128)),
        UpSampling2D((2, 2)),
        Conv2D(64, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D((2, 2)),
        Conv2D(32, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D((2, 2)),
        Conv2D(16, (5, 5), padding='same'),
        Activation('tanh')
    ])

    return model

def make_discriminator():
    model = Sequential([
        Conv3D(64,  # filters
               kernel_size=(4, 4, 4),
               strides=(2, 2, 2),
               padding='same',
               input_shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        LeakyReLU(0.2),
        Conv3D(128,  # filters
               kernel_size=(4, 4, 4),
               strides=(2, 2, 2),
               padding='same',),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv3D(256,  # filters
               kernel_size=(4, 4, 4),
               strides=(2, 2, 2),
               padding='same', ),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv3D(512,  # filters
               kernel_size=(4, 4, 4),
               strides=(2, 2, 2),
               padding='same', ),
        BatchNormalization(),
        LeakyReLU(0.2),
        Flatten(),
        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ])

    return model