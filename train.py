import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras.callbacks
from keras.optimizers import Adam

import numpy as np

import time

from load_video import load_data

from models import make_generator, make_discriminator

# Hyper paremeters
NUM_OF_DATA = 22
BATCH_SIZE = 2
NUM_EPOCH = 2

def train():
    data = load_data(NUM_OF_DATA)
    print(data.shape)

    # Make discriminator
    discriminator = make_discriminator(summery=True)
    d_optimizer =Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    # discriminator.trainable = False
    # generator = make_generator()
    # dcgan = Sequential([generator, discriminator])
    # g_optimizer = Adam(lr=2e-4, beta_1=0.5)
    # dcgan = compile(loss='binary_crossentropy', optimizer=g_optimizer)

    #test
    g = make_generator(summery=True)


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

            # make data to be processed by discriminator

            # update discriminator
            #X = np.concatenate((five_frame_image_batch, generated_frames))
            #y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(five_frame_image_batch, [1]*BATCH_SIZE)

            #print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))
            print("epoch: %d, batch: %d, g_loss: , d_loss: %f" % (epoch, index, d_loss))

if __name__ == "__main__":
    train()