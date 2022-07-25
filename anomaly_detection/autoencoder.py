
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import Model


class Autoencoder:
    """
    Autoencoder being build with n_bottleneck neurons in bottleneck.
    Encoder and decoder contain n_layers each.
    size of layers starts at 2**(log2(n_bottleneck) + 1) near bottleneck and increases with 2**(last+1)
    """
    def __init__(self, n_inputs, cpus=0, n_layers=3, n_bottleneck=2**3, seed=0, **params):
        # resetting session, setting number of threads for parallelization
        clear_session()
        tf.random.set_seed(seed)
        if cpus > 0:
            tf.config.threading.set_intra_op_parallelism_threads(cpus * 2)
            tf.config.threading.set_inter_op_parallelism_threads(cpus * 2)

        bottleneck_exp = (np.log2(n_bottleneck))

        model = Sequential()

        # Encoder
        model.add(Dense(2**(bottleneck_exp + n_layers), activation="relu", input_shape=(n_inputs,)))  # input layer
        for i in range(n_layers - 1, 0, -1):  # layers from bottleneck: 8, 16, 32, 64, ...
            model.add(Dense(2**(bottleneck_exp + i), activation="relu"))

        # Bottleneck
        model.add(Dense(n_bottleneck))

        # Decoder
        for i in range(1, n_layers + 1):  # layers from bottleneck: 8, 16, 32, 64, ...
            model.add(Dense(2**(bottleneck_exp + i), activation="relu"))

        def custom_one_hot_regularizer(outputs):
            """Forces categorical outputs towards 0 or 1"""
            categorical_mask = np.ones(200)
            # Need to exclude numerical outputs (here last alternating with 0-columns)
            categorical_mask[list(range(-2, -21, -2))] = 0
            categorical_mask = categorical_mask.reshape(1, categorical_mask.shape[0])
            return tf.reduce_sum(0.001 * tf.boolean_mask(tensor=tf.minimum(tf.abs(outputs), tf.abs(outputs - 1)),
                                                       mask=categorical_mask))

        model.add(Dense(n_inputs))  # output layer
        # model.add(Dense(n_inputs, activity_regularizer=custom_one_hot_regularizer))  # output layer with regularizer

        if 'learning_rate' in params:
            optimizer = Adam(learning_rate=params.pop('learning_rate'))
        else:
            optimizer = Adam()
        model.compile(metrics=['mse'], loss='mean_squared_error', optimizer=optimizer)

        self.model = model
        self.params = params

    def score_samples(self, x):
        return -MSE(x, self.model.predict(x)).numpy()  # invert to mimic log_likelihood outputs of sklearn

    def save(self, save_path):
        self.model.save(save_path)
        return self

    def load(self, load_path):
        self.model = load_model(load_path)
        return self

    def fit(self, data):
        self.model.fit(x=data, y=data, **self.params)
        return self

    def embed(self, data):
        # bottleneck as output
        embed_model = Model(inputs=self.model.input,
                            outputs=self.model.get_layer('dense_{}'.format(str(int(len(self.model.layers)/2-1)))).output)
        return embed_model.predict(data)
