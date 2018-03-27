# coding=utf-8
import sys

sys.path.append('../lib')
from qhead import *
import tensorflow as tf
import pickle
from sklearn import preprocessing


class EvalNet(object):
    def __init__(self, obses, train_rate = 5):
        self.learn_rate = 0.001
        self.lr_decay = 0.9
        self.obses = obses
        self.input_shape = np.shape(obses[0].obs_x[0])
        self.num_y = len(stock_data.Label.spliter) - 1
        self.batch_size = 100
        self.epoch = 10
        self.train_split = len(obses[0].obs_x) // 10 * train_rate
        # self.train_split = 50
        self.step = tf.Variable(0, trainable=False)
        self._define_train()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _flatten(self, inp):
        shape_x = inp.get_shape().as_list()
        # if len(shape_x) > 2:  # 自动把输入转为扁平
        num = 1
        for i in range(1, len(shape_x)):
            num *= shape_x[i]
        out = tf.reshape(inp, [-1, num])
        return out

    def _infer(self, batch_x, scope, is_train = False):
        bias_initer = tf.constant_initializer(0.01)
        kernel_initer = tf.random_normal_initializer(0, 0.1)

        # offset = tf.Variable(np.zeros(self.input_shape), dtype=tf.float32)
        # scale = tf.Variable(np.ones(self.input_shape), dtype=tf.float32)
        # mean, var = tf.nn.moments(batch_x, axes=[0])
        # norm_x = tf.nn.batch_normalization(batch_x, mean, var, offset=offset, scale=scale, variance_epsilon=1e-9)
        norm_x = batch_x
        with tf.variable_scope(scope):
            cnn1 = tf.layers.conv1d(norm_x, filters=16, kernel_size=32, strides=1, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            pool1 = tf.layers.max_pooling1d(cnn1, pool_size=2, strides=1)
            # pool1 = cnn1
            cnn2 = tf.layers.conv1d(pool1, filters=16, kernel_size=32, strides=1, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            pool2 = tf.layers.max_pooling1d(cnn2, pool_size=2, strides=1)
            # x = cnn2
            x2 = self._flatten(pool2)
            dn21 = tf.layers.dense(x2, 64, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            dn22 = tf.layers.dense(dn21, self.num_y)
            # if is_train:
            #     dn22 = tf.nn.dropout(dn22, keep_prob=0.6)

            x3 = tf.slice(norm_x, [0, 0, 0], [-1, 20, -1])
            # d1 = tf.layers.dense(batch_x, units=256)
            x3 = self._flatten(x3)
            dn31 = tf.layers.dense(x3, 16, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            dn32 = tf.layers.dense(dn31, self.num_y)
            q = dn32 + dn22
        return q

    def _define_train(self):
        self.batch_x = tf.placeholder(dt(), [None, *self.input_shape], name='batch_x')
        self.batch_y = tf.placeholder(tf.int32, [None], name='label_y')
        self.eval = self._infer(self.batch_x, 'eval')
        self.pred = self._infer(self.batch_x, 'pred', True)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_y, logits=self.eval))
        # sd = tf.squared_difference(x=self.eval, y=self.batch_y)
        # self.loss = tf.reduce_mean(sd)
        # self.loss = tf.reduce_mean(tf.where(tf.greater_equal(self.batch_y, 0), tf.where(tf.greater_equal(self.eval, self.batch_y), sd, sd * 5), tf.where(tf.less_equal(self.eval, self.batch_y), sd, sd * 2)))
        lr = tf.train.exponential_decay(self.learn_rate, global_step=self.step, decay_steps=10, decay_rate=self.lr_decay)
        mmt = tf.Variable(1)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self):
        max_step = 100 * self.epoch
        for i in range(max_step):
            oi = 0
            if len(self.obses) > 1:
                oi = np.random.randint(0, len(self.obses))
            obs = self.obses[oi]
            randidxes = np.random.randint(0, self.train_split, self.batch_size)
            # start = (i * self.batch_size) % self.train_split
            # end = min(start + self.batch_size, self.train_split)
            batch_x = obs.obs_x[randidxes]
            batch_y = obs.obs_y[randidxes]
            self.sess.run([self.train_op], feed_dict={self.batch_x: batch_x, self.batch_y: batch_y})
            if i % 10 == 0:
                loss = self.sess.run(self.loss, feed_dict={self.batch_x: batch_x, self.batch_y: batch_y})
                log.info('cost: %5.4f', loss)

    def test(self):
        log.info('test:')
        for i in range(len(self.obses)):
            obs = self.obses[i]
            x = obs.obs_x[:self.train_split]
            # log.info(x)
            pred = self.sess.run(self.eval, feed_dict={self.batch_x: x})
            pred = np.argmax(pred, 1)
            # for i in range(len(pred)):
            #     log.info('%s eval:%d y:%d', self.d_line[i], pred[i], self.labels[i])
            precise(obs.obs_y[:self.train_split], pred, self.num_y)

    def predict(self):
        log.info('predict:')
        for i in range(len(self.obses)):
            obs = self.obses[i]
            x = obs.obs_x[self.train_split:]
            pred = self.sess.run(self.pred, feed_dict={self.batch_x: x})
            pred = np.argmax(pred, 1)
            d_line = obs.dates[self.train_split:]
            labels = obs.obs_y[self.train_split:]
            for i in range(len(pred)):
                log.info('%s pred:%d y:%d', d_line[i], pred[i], labels[i])
            precise(labels, pred, self.num_y)
            # p_cost = np.mean(np.square(labels - pred))
            # log.info('pred cost: %.4f', p_cost)


if __name__ == '__main__':
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    # print(len(obs_x))
    obses = []
    obses.append(stock_data.prepare_single(1))
    obses.append(stock_data.prepare_single(5))
    enet = EvalNet(obses)
    enet.train()
    enet.test()
    enet.predict()
    enet.sess.close()
