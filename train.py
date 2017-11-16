import keras
from keras.optimizers import RMSprop
import keras.callbacks

import time

from load_video import load_data


NUM_OF_DATA = 22
BATCH_SIZE = 32
NUM_EPOCH = 20

def train():
    data = load_data(NUM_OF_DATA)
    print(data.shape)

    # tensorboardの出力
    save_tb_cb = keras.callbacks.TensorBoard(
        log_dir='./tf_log/' +
                str(int(time.time() * 1000)),
        write_images=True,
        histogram_freq=1)

if __name__ == "__main__":
    train()