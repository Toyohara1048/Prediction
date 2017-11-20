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
    conv_1 = Conv3D(32,          #filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 1, 1),
               input_shape=(G_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    conv_2 = Conv3D(64,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 2, 2))(conv_1)
    conv_3 = Conv3D(128,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 4, 4))(conv_2)
    conv_4 = Conv3D(256,  # filters
               kernel_size=(1, 3, 3),
               strides=(1, 1, 1),
               padding='same',
               dilation_rate=(1, 8, 8))(conv_3)
    conv_5 = Conv3D(32,  # filters
               kernel_size=(G_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid')(conv_4)

    act_1           = LeakyReLU(0.2)(conv_5)
    activated_vec   = Flatten()(act_1)
    fc_1            = Dense(61*61*128)(activated_vec)
    bn_1            = BatchNormalization()(fc_1)
    act_2           = Activation('relu')(bn_1)
    img_61          = Reshape((61, 61, 128), input_shape=(61*61*128,))(act_2)
    img_122         = UpSampling2D((2, 2))(img_61)
    conv_122        = Conv2D(64, (5, 5), padding='same')(img_122)
    bn_conv_122     = BatchNormalization()(conv_122)
    activated_122   = Activation('relu')(bn_conv_122)
    img_244         = UpSampling2D((2, 2))(activated_122)
    conv_244        = Conv2D(32, (5, 5), padding='same')(img_244)
    bn_conv_244     = BatchNormalization()(conv_244)
    activated_244   = Activation('relu')(bn_conv_244)
    img_488         = UpSampling2D((2, 2))(activated_244)
    conv_488        = Conv2D(1, (5, 5), padding='same')(img_488)
    generated_image = Activation('tanh')(conv_488)

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
    conv_1 = Conv3D(64,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same',
               input_shape=(D_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))(inputs)
    act_conv1 = LeakyReLU(0.2)(conv_1)

    conv_2 = Conv3D(128,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same')(act_conv1)
    bn_conv_2 = BatchNormalization()(conv_2)
    act_conv2 = LeakyReLU(0.2)(bn_conv_2)

    conv_3 = Conv3D(256,  # filters
               kernel_size=(1, 4, 4),
               strides=(1, 1, 1),
               padding='same')(act_conv2)
    bn_conv_3 = BatchNormalization()(conv_3)
    act_conv3 = LeakyReLU(0.2)(bn_conv_3)

    conv_4 = Conv3D(512,  # filters
               kernel_size=(D_FRAMES, 1, 1),
               strides=(1, 1, 1),
               padding='valid')(act_conv3)
    bn_conv_4 = BatchNormalization()(conv_4)
    act_conv4 = LeakyReLU(0.2)(bn_conv_4)

    vec = Flatten()(act_conv4)
    fc_1 = Dense(256)(vec)
    act_fc_1 = LeakyReLU(0.2)(fc_1)
    drop_fc_1 = Dropout(0.5)(act_fc_1)
    fc_2 = Dense(1)(drop_fc_1)

    likelihood = Activation('sigmoid')(fc_2)

    model = Model(inputs=inputs, outputs=likelihood)

    if summary:
        print("Discriminator is as follows:")
        print(model.summary())

    return model