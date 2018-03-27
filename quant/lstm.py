import sys

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

sys.path.append('../../lib')

flags = tf.flags
logging = tf.logging


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class StockInput():
    def __init__(self, config, x, y, is_train=True, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(x) // batch_size) - 1)
        # self.batch_len = 0
        self.inputs, self.labels = fw.rnn_producer(x, y, batch_size, num_steps)
        # if is_train:
        # else:
        #     vx = tf.convert_to_tensor(x, dtype=tf.float32)
        #     vy = tf.convert_to_tensor(y, dtype=tf.int32)
        #     shape = vx.get_shape().as_list()
        #     self.data_len = data_len = shape[0]
        #     dims = shape[1]
        #     batch_len = data_len // batch_size
        #     self.inputs = tf.reshape(vx[0:batch_size * batch_len], [batch_size, batch_len, dims])
        #     self.labels = tf.reshape(vy[0:batch_size * batch_len], [batch_size, batch_len])


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

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {self._name + "/cost": self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr,
                       lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = self._name + "/initial"
        self._final_state_name = self._name + "/final"
        export_state_tuples(self._initial_state, self._initial_state_name)
        export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell, self._cell.params_to_canonical, self._cell.canonical_to_params, rnn_params, base_variable_scope="Model/RNN")
                tf.add_to_collection(
                    tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(self._name + "/cost")[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)


def export_state_tuples(state_tuples, name):
    for state_tuple in state_tuples:
        tf.add_to_collection(name, state_tuple.c)
        tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
    restored = []
    for i in range(len(state_tuples) * num_replicas):
        c = tf.get_collection_ref(name)[2 * i + 0]
        h = tf.get_collection_ref(name)[2 * i + 1]
        restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
    return tuple(restored)


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

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

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


import stock_data
import tf_framework as fw


class RnnEval(object):
    def __init__(self, obses):
        self._learn_rate = 0.01
        self.obses = obses

    def _build_cell(self):


def main():
    x, y = stock_data.prepare_single('000001.XSHE')
    train_x, valid_x, train_y, valid_y = split_data(x, y)

    # eval_config.batch_size = 1
    # eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                _input = StockInput(
                    config, train_x, train_y, name='train input')
                m = StockModel(is_training=True, config=config, input_=_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            v_input = StockInput(config, valid_x, valid_y)
            # eval_config.batch_size = v_input.batch_size
            # eval_config.num_steps = v_input.batch_len
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = StockModel(is_training=False,
                                    config=config, input_=v_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        models = {"Train": m, "Valid": mvalid}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto()
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i +
                                                  1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" %
                      (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(
                    session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" %
                      (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" %
                      (i + 1, valid_perplexity))

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path,
                              global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
