
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import Model
from anomaly_detection.nalu_architectures import *


class NaluAE:
    """
        Mixed Layer based "Autoencoder" with n_layers mixed layers (plus linear input and output layer)
    """

    def __init__(self, n_inputs, cpus=0, n_layers=3, n_bottleneck=2 ** 3, seed=0, **params):
        clear_session()
        tf.random.set_seed(seed)
        if cpus > 0:
            tf.config.threading.set_intra_op_parallelism_threads(cpus * 2)
            tf.config.threading.set_inter_op_parallelism_threads(cpus * 2)

        # Encoder
        input = Input(shape=(n_inputs,))
        layer = Dense(n_inputs)(input)  # linear input layer ("Routing / Mixing Layer")

        # Mixed Layers "Bottleneck"
        for i in range(n_layers):
            nalu = NaluiLayer(n_bottleneck // 2)(layer)
            dense = Dense(n_bottleneck // 2, activation="relu")(layer)
            layer = Concatenate(axis=1)([nalu, dense])

        out = Dense(n_inputs)(layer)  # output layer ("Routing / Mixing Layer")

        model = Model(inputs=input, outputs=out)
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
                            outputs=self.model.get_layer(
                                'dense_{}'.format(str(int(len(self.model.layers) / 2 - 1)))).output)
        return embed_model.predict(data)
