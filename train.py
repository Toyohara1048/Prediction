import keras
from keras.models import Model
from keras.optimizers import RMSprop
import keras.callbacks
from keras.optimizers import Adam

import numpy as np
import time
import os
from PIL import Image

from load_video import load_data
from models import make_functional_generator, make_functional_discriminator

# Hyper paremeters
NUM_OF_DATA = 22
BATCH_SIZE = 1
NUM_EPOCH = 5
GENERATED_IMAGE_PATH = 'generated_image/'

def train():
    data = load_data(NUM_OF_DATA)
    print(data.shape)

    # Make discriminator
    discriminator = make_functional_discriminator(summary=True)
    d_optimizer =Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    discriminator.trainable = False
    generator = make_functional_generator(summary=True)
    generated_image = generator(generator.input)
    likelihood = discriminator(generated_image)
    dcgan = Model(inputs=generator.get_input_at(0), outputs=likelihood)
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

            if not os.path.exists(GENERATED_IMAGE_PATH):
                os.mkdir(GENERATED_IMAGE_PATH)
            Image.fromarray(generated_frames.astype(np.uint8))\
                .save(GENERATED_IMAGE_PATH+'%04d_%04d.png' % (epoch, index))

            # update discriminator
            X = np.concatenate((five_frame_image_batch, generated_frames))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # update generator
            g_loss = dcgan.train_on_batch(five_frame_image_batch[0:BATCH_SIZE, 0:4], [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

if __name__ == "__main__":
    train()