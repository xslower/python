import sys

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

sys.path.append('../../lib')

from qhead import *


def data_type():
    return tf.float32


class RnnEval(object):
    def __init__(self, obses):
        self._learn_rate = 0.01
        self.obses = obses
        self.batch_size = 7
        self.epoch = 10
        self.rnn_units = 30
        input_shape = np.shape(obses[0].train_x[0])
        self.input_size = input_shape[0] * input_shape[1]
        for i in range(len(obses)):
            obs = obses[i]
            num_step = len(obs.train_x) // self.batch_size
            max = num_step * self.batch_size
            obses[i].train_x = np.reshape(obs.train_x[-max:], [num_step, self.batch_size, self.input_size])
            obses[i].test_x = np.reshape(obs.test_x, [-1, 1, self.input_size])
        self._define_train()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_rnn_net(self, inputs, is_train = True):
        cells = []
        for i in range(2):
            c = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_units, activation=tf.nn.relu)
            if is_train:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=0.6)
            cells.append(c)
        multi_rnn = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
        self._initial_state = multi_rnn.zero_state(self.batch_size, data_type())
        # inputs = tf.transpose(inputs, [1, 0, 2])

        outputs = []
        state = self._initial_state
        for step in range(inputs.get_shape().as_list()[0]):
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            cell_output, state = multi_rnn(inputs[step], state)
            outputs.append(cell_output)
        # output = tf.reshape(tf.concat(outputs, 0), [-1, config.hidden_size])
        # outputs, final_state = tf.nn.static_rnn(multi_rnn, inputs, initial_state=self._initial_state)
        outputs = tf.reshape(outputs, [-1, self.rnn_units])
        weights = tf.get_variable('output-weights', [self.rnn_units, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        bias = tf.get_variable('output-bias', [1], initializer=tf.constant_initializer(0.1))
        out = tf.nn.xw_plus_b(outputs, weights, bias)
        return out, state

    def _define_train(self):
        self.x = tf.placeholder(tf.float32, [None, self.batch_size, self.input_size])
        self.y = tf.placeholder(tf.float32, [None])
        self.eval, self.stage_state = self._build_rnn_net(self.x)
        self.cost = tf.reduce_mean(tf.squared_difference(x=self.eval, y=self.y))
        self.train_op = tf.train.AdamOptimizer(self._learn_rate).minimize(self.cost)

    def train(self):
        # max_step =
        for i in range(self.epoch):
            for j in range(len(self.obses)):
                obs = self.obses[i]
                _, cost = self.sess.run([self.train_op, self.cost], feed_dict={self.x: obs.train_x, self.y: obs.train_y})
                log.info('cost: %5,3f', cost)

    def test(self):
        log.info('test:')
        for j in range(len(self.obses)):
            obs = self.obses[i]
            _, cost = self.sess.run([self.cost], feed_dict={self.x: obs.test_x, self.y: obs.test_x})
            log.info('cost: %5,3f', cost)


if __name__ == "__main__":
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    # print(len(obs_x))
    obses = []
    for i in range(20):
        try:
            obs = stock_data.prepare_single(i, 7)
            obses.append(obs)
        except:
            continue
    # obses.append(stock_data.prepare_single(2))
    # obses.append(stock_data.prepare_single(5))
    enet = RnnEval(obses)
    enet.train()
    enet.test()
    # enet.continue_test()
    enet.sess.close()
