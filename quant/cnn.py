# coding=utf-8
import sys

sys.path.append('../lib')
from header import *
import tensorflow as tf
import pickle
from sklearn import preprocessing


def dt():
    return tf.float32


class EvalNet(object):
    def __init__(self, obs, labels, d_line, train_rate = 5):
        self.learn_rate = 0.5
        self.lr_decay = 0.9
        self.obs = obs
        self.labels = labels
        self.d_line = d_line
        self.input_shape = np.shape(obs[0])
        self.batch_size = 50
        self.epoch = 10
        self.train_split = len(obs) // 10 * train_rate
        self.step = tf.Variable(0, trainable=False)
        self._define_train()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _infer(self, batch_x, scope = 'cnn'):
        bias_initer = tf.constant_initializer(0.1)
        kernel_initer = tf.random_normal_initializer(0, 0.5)
        with tf.variable_scope(scope):
            cnn1 = tf.layers.conv1d(batch_x, filters=16, kernel_size=5, strides=1, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            pool1 = tf.layers.max_pooling1d(cnn1, pool_size=2, strides=1)
            # pool1 = cnn1
            cnn2 = tf.layers.conv1d(pool1, filters=32, kernel_size=5, strides=1, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            pool2 = tf.layers.max_pooling1d(cnn2, pool_size=2, strides=1)
            # x = cnn2
            x = pool2
            # d1 = tf.layers.dense(batch_x, units=256)
            shape_x = x.get_shape().as_list()
            if len(shape_x) > 2:  # 自动把输入转为扁平
                num = 1
                for i in range(1, len(shape_x)):
                    num *= shape_x[i]
                x = tf.reshape(x, [-1, num])

            dn1 = tf.layers.dense(x, 128, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            dn2 = tf.layers.dense(dn1, 128, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            q = tf.layers.dense(dn1, 1)
        return q

    def _define_train(self):
        self.batch_x = tf.placeholder(dt(), [None, *self.input_shape], name='batch_x')
        self.batch_y = tf.placeholder(dt(), [None], name='label_y')
        self.eval = self._infer(self.batch_x)
        sd = tf.squared_difference(x=self.eval, y=self.batch_y)
        # self.loss = tf.reduce_mean()
        self.loss = tf.reduce_mean(tf.where(tf.greater_equal(self.batch_y, 0),
            tf.where(tf.greater_equal(self.eval, self.batch_y), sd/2, sd * 100),
            tf.where(tf.less_equal(self.eval, self.batch_y), sd/2, sd * 100)))
        lr = tf.train.exponential_decay(self.learn_rate, global_step=self.step, decay_steps=10, decay_rate=self.lr_decay)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self):
        max_step = 100 * self.epoch
        for i in range(max_step):
            start = (i * self.batch_size) % self.train_split
            end = min(start + self.batch_size, self.train_split)
            batch_x = self.obs[start:end]
            batch_y = self.labels[start:end]
            self.sess.run([self.train_op], feed_dict={self.batch_x: batch_x, self.batch_y: batch_y})
            if i % 10 == 0:
                loss = self.sess.run(self.loss, feed_dict={self.batch_x: batch_x, self.batch_y: batch_y})
                log.info('cost: %5.4f', loss)

    def test(self):
        log.info('test:')
        pred = self.sess.run(self.eval, feed_dict={self.batch_x: self.obs[:self.train_split]})
        for i in range(len(pred)):
            log.info('%s pred:%.4f y:%.4f', self.d_line[i], pred[i], self.labels[i])

    def predict(self):
        log.info('predict:')
        pred = self.sess.run(self.eval, feed_dict={self.batch_x: self.obs[self.train_split:]})
        d_line = self.d_line[self.train_split:]
        labels = self.labels[self.train_split:]
        for i in range(len(pred)):
            log.info('%s pred:%.4f y:%.4f', d_line[i], pred[i], labels[i])
        p_cost = np.mean(np.square(labels - pred))
        log.info('pred cost: %.4f', p_cost)


if __name__ == '__main__':
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    d_line, k_line, obs_x = stock_data.prepare_single(5)
    label = stock_data.Label(k_line=k_line, d_line=d_line)
    label.calc_up()
    enet = EvalNet(obs_x, labels=label.up_table, d_line=d_line)
    enet.train()
    enet.test()
    enet.predict()
    enet.sess.close()
