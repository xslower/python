import sys

import time

import numpy as np
import tensorflow as tf

sys.path.append('../../lib')

from qhead import *


def data_type():
    return tf.float32


def print_table(tb):
    for i in range(len(tb)):
        print(tb[i])


class RnnEval(object):
    def __init__(self, obses):
        self._learn_rate = 0.03
        self.batch_size = 1
        self.epoch = 80
        self.rnn_units = 12
        self.num_y = 2
        input_shape = np.shape(obses[0].train_x[0])
        self.input_size = input_shape[0] * input_shape[1]
        for i in range(len(obses)):
            obs = obses[i]
            num_step = len(obs.train_x) // self.batch_size
            max = num_step * self.batch_size
            obses[i].train_x = np.reshape(obs.train_x[-max:], [self.batch_size, num_step, self.input_size])
            obses[i].train_y = obs.train_y[-max:]
            num_step_t = len(obs.test_x) // self.batch_size
            max_t = num_step_t * self.batch_size
            obses[i].test_x = np.reshape(obs.test_x[:max_t], [self.batch_size, num_step_t, self.input_size])
            obses[i].test_y = obs.test_y[:max_t]
            # print_table(obs.test_x)
            # print_table(obs.test_y)
        self.obses = obses
        self._define_train()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_rnn_net(self, inputs, is_train = True):
        cells = []
        for i in range(2):
            c = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_units, activation=tf.nn.tanh)
            # if is_train:
            #     c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=0.6)
            cells.append(c)
        multi_rnn = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
        self._initial_state = multi_rnn.zero_state(self.batch_size, data_type())
        # inputs = tf.transpose(inputs, [1, 0, 2])

        outputs, state = tf.nn.dynamic_rnn(multi_rnn, inputs, initial_state=self._initial_state)
        outputs = tf.reshape(outputs, [-1, self.rnn_units])
        # weights = tf.get_variable('output-weights', [self.rnn_units, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        # bias = tf.get_variable('output-bias', [1], initializer=tf.constant_initializer(0.1))
        # out = tf.nn.xw_plus_b(outputs, weights, bias)
        out = tf.layers.dense(outputs, self.num_y)
        if self.num_y == 1:
            out = tf.squeeze(out, axis=1)
        return out, state

    def _define_train(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.num_y])
        self.eval, self.stage_state = self._build_rnn_net(self.x)

        self.cost = tf.reduce_mean(tf.squared_difference(x=self.eval, y=self.y))
        self.train_op = tf.train.AdamOptimizer(self._learn_rate).minimize(self.cost)

    def train(self):
        # max_step =
        for i in range(self.epoch):
            for j in range(len(self.obses)):
                obs = self.obses[j]
                _, cost = self.sess.run([self.train_op, self.cost], feed_dict={self.x: obs.train_x, self.y: obs.train_y})
                log.info('%d cost: %s', j, str(cost))

    def test(self):
        log.info('test:')
        states = []
        for j in range(len(self.obses)):
            obs = self.obses[j]
            eval, state, cost = self.sess.run([self.eval, self.stage_state, self.cost], feed_dict={self.x: obs.train_x, self.y: obs.train_y})
            states.append(state)
            # for i in range(len(eval)):
            #     log.info('eval:%s y:%s', eval[i], obs.train_y[i])
            log.info('%d cost: %s', j, str(cost))
        return states

    def predict(self, states):
        log.info('predict')
        for j in range(len(self.obses)):
            obs = self.obses[j]
            pred, cost = self.sess.run([self.eval, self.cost], feed_dict={self.x: obs.test_x, self.y: obs.test_y, self._initial_state: states[j]})
            for i in range(len(pred)):
                log.info('pred:%s y:%s', pred[i], obs.test_y[i])
            log.info('%d cost: %s', j, str(cost))
            sq = np.square(np.mean(obs.test_y, axis=0)-obs.test_y)
            cost = np.mean(sq)
            log.info(' mean cost: %s', cost)


if __name__ == "__main__":
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    # print(len(obs_x))
    obses = []
    for i in range(2):
        try:
            obs = stock_data.prepare_single(i, 7)
            obses.append(obs)
        except:
            continue
    # obses.append(stock_data.prepare_single(2))
    # obses.append(stock_data.prepare_single(5))
    enet = RnnEval(obses)
    enet.train()
    states = enet.test()
    enet.predict(states)
    # enet.continue_test()
    enet.sess.close()
