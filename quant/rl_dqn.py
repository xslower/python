'''dqn '''
import tensorflow as tf
import numpy as np
from stock_simulator import Simulator


def dtype():
    return tf.float16


stock = []
sim = Simulator(stock)


class Dqn(object):
    def __init__(self, n_act, obs_shape):
        self.learn_rate = 0.1
        self.decay = 0.9
        self.input_shape = obs_shape
        self.num_act = n_act
        self.rand_gate = 0.5
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size = 5
        self.step = 1
        self.l_obs = []
        self.l_act = []
        self.l_reward = []
        self.l_next_obs = []

    def _dtype(self):
        return tf.float16

    def _build_net(self, obs, scope):
        with tf.variable_scope(scope):
            cnn1 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, input_shape=self.input_shape, padding='valid')(obs)
            pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(cnn1)
            cnn2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='valid')(pool1)
            pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(cnn2)
            dnn = tf.keras.layers.Dense(units=256, activation='relu')(pool2)
            out = tf.keras.layers.Dense(units=sim.num_acts)(dnn)
            return out

    def _build_dqn(self):
        self.samples = tf.placeholder(dtype(), [None, *self.input_shape], name='samples')
        self.q_target = tf.placeholder(dtype(), [None, sim.num_acts], name='q_target')
        self.q_eval = self._build_net(self.samples, 'eval')
        self.loss = tf.reduce_mean(tf.squared_difference(x=self.q_eval, y=self.q_target))
        self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)
        self.q_pred = self._build_net(self.samples, 'pred')
        eval_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        pred_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred')
        move_decay = 0.5
        self.replace = [tf.assign(e, (1 - move_decay) * e + move_decay * p) for e, p in zip(eval_para, pred_para)]

    def store_transition(self, obs, act, reward, next_obs):
        self.l_obs.append(obs)
        self.l_act.append(act)
        self.l_reward.append(reward)
        self.l_next_obs.append(next_obs)

    def choose_action(self, obs):
        if np.random.uniform() < self.rand_gate:
            obs = obs[np.newaxis, :]
            acts = self.sess.run(self.q_eval, feed_dict={self.samples: obs})
            act = np.argmax(acts)
        else:
            act = np.random.randint(0, self.num_act)
        return act

    def learn(self):
        if self.step % 10 == 0:
            self.sess.run(self.replace)

        q_predict = self.sess.run(self.q_pred, feed_dict={self.samples: self.l_next_obs})
        q_target = self.sess.run(self.q_eval, feed_dict={self.samples: self.l_obs})
        num = len(self.l_reward)
        batch_indexes = np.arange(num, dtype=np.int32)
        q_target[batch_indexes, self.l_act] = self.l_reward + self.decay * np.max(q_predict, axis=1)

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.samples: self.l_obs, self.q_target: q_target})
        print(cost)
        self.step += 1
