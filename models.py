import math
import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Input, concatenate
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


def make_generator(summary = False):
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
               dilation_rate=(1, 2, 2)),
        Conv3D(128,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 4, 4)),
        Conv3D(256,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 8, 8)),
        Conv3D(32,  # filters
               kernel_size=(G_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid'),

        # followings are original
        LeakyReLU(0.2),
        Flatten(),
        Dense(61*61*128),
        BatchNormalization(),
        Activation('relu'),
        Reshape((61, 61, 128), input_shape=(61*61*128,)),
        UpSampling2D((2, 2)),
        Conv2D(64, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D((2, 2)),
        Conv2D(32, (5, 5), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D((2, 2)),
        Conv2D(1, (5, 5), padding='same'),
        Activation('tanh')
    ])

    if summary:
        print("Generator is as follows:")
        print(model.summary())
    return model

def make_functional_generator(summary = False):
    inputs = Input(shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    x = Conv3D(32,          #filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 1, 1),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    x = Conv3D(64,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 2, 2))(x)
    x = Conv3D(128,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 4, 4))(x)
    x = Conv3D(256,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 8, 8))(x)
    x = Conv3D(32,  # filters
               kernel_size=(G_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid')(x)

    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(61*61*128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((61, 61, 128), input_shape=(61*61*128,))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (5, 5), padding='same')(x)
    generated_image = Activation('tanh')(x)

    #cat generated_image with inputs -> images
    reshaped_gen_image = Reshape((1, 488, 488, 1), input_shape=(488, 488, 1))(generated_image)
    images = concatenate([inputs, reshaped_gen_image], axis=1)
    print(images.shape)

    model = Model(inputs=inputs, outputs=images)

    if summary:
        print("Generator is as follows:")
        print(model.summary())

    return model


def make_discriminator(summary = False):
    model = Sequential([
        Conv3D(64,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same',
               input_shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)),
        LeakyReLU(0.2),
        Conv3D(128,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same',),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv3D(256,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same', ),
        BatchNormalization(),
        LeakyReLU(0.2),
        Conv3D(512,  # filters
               kernel_size=(D_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid', ),
        BatchNormalization(),
        LeakyReLU(0.2),
        Flatten(),
        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ])

    if summary:
        print("Discriminator is as follows:")
        print(model.summary())
    return model


def make_functional_discriminator(summary = False):
    inputs = Input(shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    x = Conv3D(64,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same',
               input_shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(128,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(256,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(512,  # filters
               kernel_size=(D_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)

    likelihood = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=likelihood)

    if summary:
        print("Discriminator is as follows:")
        print(model.summary())

    return model