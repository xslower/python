'''dqn '''
import tensorflow as tf
from stock_simulator import Simulator


def dtype():
    return tf.float16


stock = []
sim = Simulator(stock)


class Dqn(object):
    def __init__(self):
        self.decay = 0.9
        self.input_shape = [300, 6]

    def _dtype(self):
        return tf.float16

    def _build_net(self):
        samples = tf.placeholder(dtype(), [None, *self.input_shape], name='samples')
        q_target = tf.placeholder(dtype(), [None, sim.num_acts], name='q_target')

        cnn1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, input_shape=self.input_shape, padding='valid')(samples)
        pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(cnn1)
        cnn2 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='valid')(pool1)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(cnn2)

