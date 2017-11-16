import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras.callbacks
from keras.optimizers import Adam

import time

from load_video import load_data

from models import make_generator, make_discriminator

# Hyper paremeters
NUM_OF_DATA = 22
BATCH_SIZE = 32
NUM_EPOCH = 20

def train():
    data = load_data(NUM_OF_DATA)
    print(data.shape)

    # Make discriminator
    discriminator = make_discriminator()
    d_optimizer =Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    discriminator.trainable = False
    generator = make_generator()
    dcgan = Sequential([generator, discriminator])
    g_optimizer = Adam(lr=2e-4, beta_1=0.5)
    dcgan = compile(loss='binary_crossentropy', optimizer=g_optimizer)


    # tensorboardの出力
    save_tb_cb = keras.callbacks.TensorBoard(
        log_dir='./tf_log/' +
                str(int(time.time() * 1000)),
        write_images=True,
        histogram_freq=1)

if __name__ == "__main__":
    train()