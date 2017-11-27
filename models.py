import math
import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Input, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv3D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

# Common hyper paremeters
LENGTH_OF_SIDE = 122

# Genarator hyper parameters
G_FRAMES = 4

# Discriminator hyper parameters
D_FRAMES = 5


def make_generator(summary = False):
    inputs = Input(shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    # x = Conv3D(32,          #filters
    #            kernel_size=(1, 3, 3),
    #            strides=(1, 1, 1),
    #            padding='valid',
    #            dilation_rate=(1, 1, 1),
    #            input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    # x = LeakyReLU(0.2)(x)
    # x = Conv3D(64,  # filters
    #            kernel_size=(1, 3, 3),
    #            strides=(1, 1, 1),
    #            padding='valid',
    #            dilation_rate=(1, 1, 1))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = Conv3D(64,  # filters
    #            kernel_size=(1, 3, 3),
    #            strides=(1, 1, 1),
    #            padding='same',
    #            dilation_rate=(1, 4, 4))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = Conv3D(32,  # filters
    #            kernel_size=(1, 3, 3),
    #            strides=(1, 1, 1),
    #            padding='same',
    #            dilation_rate=(1, 8, 8))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    x = Conv3D(64,  # filters
               kernel_size=(G_FRAMES, 3, 3),
               strides=(1, 1, 1),
               padding='valid',
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(128,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='valid',
               dilation_rate=(1, 1, 1))(x)
    x = BatchNormalization()(x)

    #x = MaxPooling3D(pool_size=(1, 7, 7), strides=None, padding='valid')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(200)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(61*61*1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((61, 61, 1), input_shape=(61*61*1,))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(2, (5, 5), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (5, 5), padding='same')(x)
    generated_image = Activation('tanh')(x)

    #cat generated_image with inputs -> images
    reshaped_gen_image = Reshape((1, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1), input_shape=(LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(generated_image)
    images = concatenate([inputs, reshaped_gen_image], axis=1)

    model = Model(inputs=inputs, outputs=images)

    if summary:
        print("Generator is as follows:")
        print(model.summary())

    return model


def make_sequential_generator(summary = False):
    inputs = Input(shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    x = Conv3D(32,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='valid',
               dilation_rate=(1, 1, 1),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(64,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='valid',
               dilation_rate=(1, 1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv3D(128,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='valid',
               dilation_rate=(1, 1, 1))(x)


    x = Conv2D(1, (5, 5), padding='same')(x)
    generated_image = Activation('tanh')(x)

    # cat generated_image with inputs -> images
    reshaped_gen_image = Reshape((1, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1),
                                 input_shape=(LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(generated_image)
    images = concatenate([inputs, reshaped_gen_image], axis=1)

    model = Model(inputs=inputs, outputs=images)

    if summary:
        print("Sequential generator is as follows:")
        print(model.summary())

    return model



def make_discriminator(summary = False):
    inputs = Input(shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    x = Conv3D(32,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same',
               input_shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(64,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # x = Conv3D(32,  # filters
    #            kernel_size=(1, 4, 4),
    #            strides=(1, 1, 1),
    #            padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)

    x = Conv3D(16,  # filters
               kernel_size=(D_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid')(x)
    x = MaxPooling3D(pool_size=(1, 7, 7), strides=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)

    likelihood = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=likelihood)

    if summary:
        print("Discriminator is as follows:")
        print(model.summary())

    return model