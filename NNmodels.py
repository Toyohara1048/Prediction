from keras.models import Model
from keras.layers import Dense, Activation, Reshape, Input, concatenate
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

# Common hyper paremeters
LENGTH_OF_SIDE = 122

# Genarator hyper parameters
G_FRAMES = 4

# Discriminator hyper parameters
D_FRAMES = 5

def make_predictor(summary=False):
    input_center = Input(shape=(G_FRAMES*2,), name='center_in')
    input_tan = Input(shape=(G_FRAMES,), name='tan_in')

    # Center of object
    x_center = Dense(256, input_dim=G_FRAMES*2)(input_center)
    x_center = Activation('relu')(x_center)
    x_center = Dropout(0.25)(x_center)
    x_center = Dense(128)(x_center)
    x_center = Activation('relu')(x_center)
    x_center = Dropout(0.25)(x_center)
    x_center = Dense(2)(x_center)
    output_center = Activation('relu', name='center_out')(x_center)

    # Angle
    x_tan = Dense(256, input_dim=G_FRAMES)(input_tan)
    x_tan = Activation('relu')(x_tan)
    x_tan = Dropout(0.25)(x_tan)
    x_tan = Dense(128)(x_tan)
    x_tan = Activation('relu')(x_tan)
    x_tan = Dropout(0.25)(x_tan)
    x_tan = Dense(1)(x_tan)
    output_tan = Activation('relu', name='tan_out')(x_tan)

    model = Model(inputs=[input_center, input_tan], outputs=[output_center, output_tan])

    if summary:
        print("NN is as follows:")
        print(model.summary())

    return model


def make_recurrent_predictor(summary=False):
    input_center = Input(shape=(G_FRAMES * 2,), name='center_in')
    input_tan = Input(shape=(G_FRAMES,), name='tan_in')

    # Center of object
    x_center = Reshape((G_FRAMES, 2), input_shape=(G_FRAMES*2,))(input_center)
    x_center = SimpleRNN(128, input_shape=(G_FRAMES, 2))(x_center)
    x_center = Dense(2)(x_center)
    output_center = Activation('relu', name='center_out')(x_center)

    # Angle
    x_tan = Reshape((G_FRAMES, 1), input_shape=(G_FRAMES,))(input_tan)
    x_tan = SimpleRNN(128, input_shape=(G_FRAMES, 1))(x_tan)
    x_tan = Dense(1)(x_tan)
    output_tan = Activation('relu', name='tan_out')(x_tan)

    model = Model(inputs=[input_center, input_tan], outputs=[output_center, output_tan])

    if summary:
        print("RNN is as follows:")
        print(model.summary())

    return model


