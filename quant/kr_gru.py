import sys

import time

import numpy as np
import tensorflow as tf
import keras as kr
from keras.layers import *
from keras.models import Model

sys.path.append('../lib')

import stock_data


def data_type():
    return tf.float32


def print_table(tb):
    for i in range(len(tb)):
        print(tb[i])


class RnnEval(object):
    def __init__(self, X, Y, time_step = 450):
        self._learn_rate = 0.0002
        self.batch_size = 100
        self.epoch = 350
        self.rnn_units = 30
        self.num_y = 1
        input_shape = np.shape(X[0])
        self.num_step = input_shape[0]
        self.emb_size = input_shape[1]
        self.x = X
        self.y = Y
        self.build_net()

    def shape(self, tensor):
        print(tensor.get_shape().as_list())

    def build_net(self):
        x = Input([self.num_step, self.emb_size])
        # cnn = Conv1D(50, kernel_size=10, strides=2)(x)
        # cnn = Conv1D(20, kernel_size=10)(cnn)
        gru = CuDNNGRU2(self.rnn_units, return_sequences=True)(x)
        self.shape(gru)
        den = Flatten()(gru)
        # den = Flatten()(cnn)
        # den = gru
        out = Dense(1, name='output')(den)
        # den = Reshape([])(den)
        model = Model(x, out)
        sgd = kr.optimizers.Adam(lr=self._learn_rate, decay=1e-6)
        model.compile(sgd, loss='mean_squared_error', metrics=['mse'])
        self.model = model

    def train(self):
        early = kr.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        self.model.fit(self.x, self.y, batch_size=self.batch_size, epochs=self.epoch, verbose=2, validation_split=0.1, callbacks=[early])

    def predict(self, x):
        return self.model.predict(x)

    def test(self):
        y = self.predict(self.x[-20:])
        for i in range(len(y)):
            print(y[i], self.y[i])


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T


class CuDNNGRU2(CuDNNGRU):
    def __init__(self, units, return_sequences = False, return_state = False, stateful = False, **kwargs):
        self.units = units
        super(CuDNNGRU2, self).__init__(units=units, return_sequences=return_sequences, return_state=return_state, stateful=stateful, **kwargs)

    @property
    def cell(self):
        Cell = namedtuple('cell', 'state_size')
        cell = Cell(state_size=self.units)
        return cell

    def build(self, input_shape):
        super(CuDNNGRU2, self).build(input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_dim = input_shape[-1]

        from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
        self._cudnn_gru = cudnn_rnn_ops.CudnnGRU(num_layers=2, num_units=self.units, input_size=input_dim, input_mode='linear_input')

        self.kernel = self.add_weight(shape=(input_dim, self.units * 3), name='kernel', initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3), name='recurrent_kernel', initializer=self.recurrent_initializer, regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.bias = self.add_weight(shape=(self.units * 6,), name='bias', initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units:
        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        self.bias_z_i = self.bias[:self.units]
        self.bias_r_i = self.bias[self.units: self.units * 2]
        self.bias_h_i = self.bias[self.units * 2: self.units * 3]
        self.bias_z = self.bias[self.units * 3: self.units * 4]
        self.bias_r = self.bias[self.units * 4: self.units * 5]
        self.bias_h = self.bias[self.units * 5:]

        self.built = True

    def _process_batch(self, inputs, initial_state):
        import tensorflow as tf
        inputs = tf.transpose(inputs, (1, 0, 2))
        input_h = initial_state[0]
        input_h = tf.expand_dims(input_h, axis=0)

        params = self._canonical_to_params(weights=[self.kernel_r, self.kernel_z, self.kernel_h, self.recurrent_kernel_r, self.recurrent_kernel_z, self.recurrent_kernel_h, ], biases=[self.bias_r_i, self.bias_z_i, self.bias_h_i, self.bias_r, self.bias_z, self.bias_h, ], )
        outputs, h = self._cudnn_gru(inputs, input_h=input_h, params=params, is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
        if self.return_sequences:
            output = tf.transpose(outputs, (1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h]


if __name__ == "__main__":
    import logging as log
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')

    X, Y = stock_data.prepare_list()

    # obses.append(stock_data.prepare_single(2))
    # obses.append(stock_data.prepare_single(5))
    enet = RnnEval(X, Y)
    enet.train()
    enet.test()
