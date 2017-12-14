from keras.models import Model
from keras.layers import Input
import keras.callbacks
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF


import numpy as np
import math
import os
import tensorflow as tf
from PIL import Image

from load_video import load_data
from models import make_simple_discriminator, make_simple_generator

# Hyper paremeters
NUM_OF_DATA = 1427
BATCH_SIZE = 3
NUM_EPOCH = 100
NUM_FRAME = 5
LENTH_OF_SIDE = 122

# Image saving
local = True    #Work on local mac or linux with GPU?
GENERATED_IMAGE_PATH = '/media/hdd/Toyohara/PredictNextPose/simple_generated_image/'
LOCAL_GENERATED_IMAGE_PATH = 'simple_generated_image/'

# Weight saving
WEIGHT_PATH = '/media/hdd/Toyohara/PredictNextPose/simple_weight/'
LOCAL_WEIGHT_PATH = 'simple_weight/'


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width = generated_images.shape[1]
    height = generated_images.shape[2]
    combined_image = np.zeros((int(height*rows), int(width*cols)),
                              dtype=generated_images.dtype)


    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image


def train():
    # about path for saving images
    if local:
        path = LOCAL_GENERATED_IMAGE_PATH
    else:
        path = GENERATED_IMAGE_PATH

    # configuration of GPU usage
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    KTF.set_session(session)

    data = load_data(NUM_OF_DATA)
    data = (data.astype(np.float32) - 127.5) / 127.5
    print("Data.shape is:")
    print(data.shape)

    # Make discriminator
    discriminator = make_simple_discriminator(summary=True)
    d_optimizer = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    discriminator.trainable = False
    generator = make_simple_generator(summary=True)
    input = Input(shape=(128,))
    generated_image = generator(input)
    likelihood = discriminator(generated_image)
    dcgan = Model(inputs=input, outputs=likelihood)
    g_optimizer = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    num_of_batch = int(data.shape[0] / BATCH_SIZE)
    print("Number of batch: %d" % num_of_batch)
    for epoch in range(NUM_EPOCH):
        for index in range(num_of_batch):
            five_frame_image_batch = data[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

            # generate image (4 frames -> next 1 frame)
            noise = np.array([np.random.uniform(-1, 1, 128) for _ in range(BATCH_SIZE)])
            generated_frames = generator.predict(noise)

            if index % 20 == 0:
                if not os.path.exists(path):
                    os.mkdir(path)
                image = combine_images(generated_frames)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)) \
                    .save(path + '%04d_%04d.png' % (epoch, index))

            # update discriminator
            X = np.concatenate((five_frame_image_batch[0:BATCH_SIZE, 0], generated_frames))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # update generator
            noise = np.array([np.random.uniform(-1, 1, 128) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1] * BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

            # save weights
            if local:
                weight_path = LOCAL_WEIGHT_PATH
            else:
                weight_path = WEIGHT_PATH

            if not os.path.exists(weight_path):
                os.mkdir(weight_path)
            generator.save_weights(weight_path + 'generator.h5')
            discriminator.save_weights(weight_path + 'discriminator.h5')


if __name__ == "__main__":
    train()