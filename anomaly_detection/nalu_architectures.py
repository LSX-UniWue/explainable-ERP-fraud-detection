import numpy as np
import tensorflow as tf

ig = 0.0
im = 0.5
iw = 0.88

isg = 0.2
ism = 0.2
isw = 0.2

from tensorflow.keras.layers import Layer

class NaluiLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NaluiLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        w_initialization = tf.random_normal_initializer(mean=iw, stddev=isw)
        m_initialization = tf.random_normal_initializer(mean=im, stddev=ism)
        g_initialization = tf.random_normal_initializer(mean=ig, stddev=isg)

        self.w_hat1 = tf.Variable(w_initialization(shape=(input_shape[1], self.output_dim), dtype="float32"), name="what", trainable=True)
        self.m_hat1 = tf.Variable(m_initialization(shape=(input_shape[1], self.output_dim), dtype="float32"), name="mhat", trainable=True)
        self.w_hat2 = tf.Variable(w_initialization(shape=(input_shape[1], self.output_dim), dtype="float32"), name="whatm", trainable=True)
        self.m_hat2 = tf.Variable(m_initialization(shape=(input_shape[1], self.output_dim), dtype="float32"), name="mhatm", trainable=True)
        self.G = tf.Variable(g_initialization(shape=(1, self.output_dim), dtype="float32"), name="g", trainable=True)

    def call(self, input):
        W1 = tf.tanh(self.w_hat1) * tf.sigmoid(self.m_hat1)
        W2 = tf.tanh(self.w_hat2) * tf.sigmoid(self.m_hat2)
        a1 = tf.matmul(input, W1)

        m1 = tf.math.exp(tf.minimum(tf.matmul(tf.math.log(tf.maximum(tf.abs(input), 1e-7)), W2), 20)) # clipping

        ### sign
        W1s = tf.reshape(W2, [-1]) # flatten W1s to (200)
        W1s = tf.abs(W1s)
        Xs = tf.concat([input] * W1.shape[1], axis=1)
        Xs = tf.reshape(Xs, shape=[-1,W1.shape[0] * W1.shape[1]])
        sgn = tf.sign(Xs) * W1s + (1 - W1s)
        sgn = tf.reshape(sgn, shape=[-1, W1.shape[1], W1.shape[0]])
        ms1 = tf.reduce_prod(sgn, axis=2)
        g = tf.sigmoid(self.G)
        out = g * a1 + (1 - g) * m1 * tf.clip_by_value(ms1, -1, 1)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


