import keras
from keras.models import Model
from keras.layers import Input
import keras.callbacks
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF


import numpy as np
import time
import os
import tensorflow as tf
from PIL import Image

from load_video import load_data
from models import make_functional_generator, make_functional_discriminator

# Hyper paremeters
NUM_OF_DATA = 22
BATCH_SIZE = 2
NUM_EPOCH = 100
NUM_FRAME = 5
GENERATED_IMAGE_PATH = '/media/hdd/Toyohara/PredictNextPose/generated_image/'

def combine_images(generated_image):
    print(generated_image.shape)
    total = generated_image.shape[0]
    width = generated_image.shape[1]
    height = generated_image.shape[2]
    combine_image = np.zeros((height*BATCH_SIZE, width*5), dtype= generated_image.dtype)

    for index, image in enumerate(generated_image):
        i = int(index / NUM_FRAME)
        j = index % NUM_FRAME
        combine_image[width*i:width*(i+1), height*j:height*(j+1)] = image[0:488, 0:488]

    return combine_image

def train():
    # configuration of GPU usage
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    KTF.set_session(session)

    data = load_data(NUM_OF_DATA)
    print(data.shape)

    # Make discriminator
    discriminator = make_functional_discriminator(summary=True)
    d_optimizer =Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    discriminator.trainable = False
    generator = make_functional_generator(summary=True)
    input = Input(shape=(4, 488, 488, 1))
    generated_image = generator(input)
    likelihood = discriminator(generated_image)
    dcgan = Model(inputs=input, outputs=likelihood)
    g_optimizer = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    # tensorboardの出力
    # save_tb_cb = keras.callbacks.TensorBoard(
    #     log_dir='./tf_log/' +
    #             str(int(time.time() * 1000)),
    #     write_images=True,
    #     histogram_freq=1)

    num_of_batch = int(data.shape[0] / BATCH_SIZE)
    print("Number of batch: %d" % num_of_batch)
    for epoch in range(NUM_EPOCH):
        for index in range(num_of_batch):
            five_frame_image_batch = data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # generate image (4 frames -> next 1 frame)
            generated_frames = generator.predict(five_frame_image_batch[0:BATCH_SIZE, 0:4])
            print(five_frame_image_batch[0:BATCH_SIZE, 0:4].shape)

            if index == 0:
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                image = combine_images(generated_frames.reshape(5 * BATCH_SIZE, 488, 488))
                Image.fromarray(image.astype(np.uint8)) \
                    .save(GENERATED_IMAGE_PATH + '%04d_%04d.png' % (epoch, index))


            # update discriminator
            X = np.concatenate((five_frame_image_batch, generated_frames))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # update generator
            g_loss = dcgan.train_on_batch(five_frame_image_batch[0:BATCH_SIZE, 0:4], [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))


if __name__ == "__main__":
    train()