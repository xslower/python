import sys

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

sys.path.append('../../lib')

flags = tf.flags
logging = tf.logging


def data_type():
    return tf.float32


class StockModel(object):

    def __init__(self, config, input_, is_training=True):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        size = config.hidden_size
        out_size = config.out_size

        # 根据输入的词选取这些词对应的训练参数
        # with tf.device("/cpu:0"):
        inputs = input_.inputs
        # if is_training and config.keep_prob < 1:
        #     inputs = tf.nn.dropout(inputs, config.keep_prob)
        output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

        softmax_w = tf.get_variable(
            "softmax_w", [size, out_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [out_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(
            logits, [self.batch_size, self.num_steps, out_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(logits, input_.labels, tf.ones(
            [self.batch_size, self.num_steps], dtype=data_type()), average_across_timesteps=False, average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(
            self._cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            # return tf.contrib.rnn.BasicLSTMCell(
            return tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""

        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # print('state shape', len(state), config.num_layers, len(state[0]), config.batch_size, state[0])
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})



class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 2
    out_size = 10

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost, "final_state": model.final_state, }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("step: %.3f, cost: %.3f, speed: %.0f wps" % (
                step, costs, iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


class RnnEval(object):
    def __init__(self, obses):
        self._learn_rate = 0.01
        self.obses = obses
        self.batch_size = 2
        self.input_shape = np.shape(obses[0].train_x[0])

    def _build_rnn_net(self, inputs, is_train = True):
        cells = []
        for i in range(3):
            c = tf.nn.rnn_cell.GRUCell(num_units=20, activation=tf.nn.relu)
            if is_train:
                c = tf.contrib.rnn.DropoutWrapper(c, output_keep_prob=0.6)
            cells.append(c)
        multi_rnn = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
        self._initial_state = multi_rnn.zero_state(self.batch_size, data_type())

        outputs, state = tf.nn.static_rnn(multi_rnn, inputs, initial_state=self._initial_state)
        return outputs

    def _define_train(self):
        self.x = tf.placeholder(tf.float32, [None, *self.input_shape])



if __name__ == "__main__":
    tf.app.run()
