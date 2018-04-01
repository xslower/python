# coding=utf-8
import sys

sys.path.append('../lib')
from qhead import *
import tensorflow as tf
import pickle
from sklearn import preprocessing


class EvalNet(object):
    def __init__(self, obses):
        self.learn_rate = 0.001
        self.lr_decay = 0.9
        self.obses = obses
        print(len(obses))
        print(len(obses[0].train_x))
        self.input_shape = np.shape(obses[0].train_x[0])
        self.num_y = 2
        self.batch_size = 100
        self.epoch = 150
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

    def _infer(self, batch_x, scope, is_train = True):
        bias_initer = tf.constant_initializer(0.01)
        kernel_initer = tf.random_normal_initializer(0, 0.1)

        # offset = tf.Variable(np.zeros(self.input_shape), dtype=tf.float32)
        # scale = tf.Variable(np.ones(self.input_shape), dtype=tf.float32)
        # mean, var = tf.nn.moments(batch_x, axes=[0])
        # norm_x = tf.nn.batch_normalization(batch_x, mean, var, offset=offset, scale=scale, variance_epsilon=1e-9)
        norm_x = batch_x
        with tf.variable_scope(scope):
            cnn1 = tf.layers.conv1d(norm_x, filters=16, kernel_size=24, strides=2, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            cnn1 = tf.layers.max_pooling1d(cnn1, pool_size=2, strides=1)
            # pool1 = cnn1
            cnn2 = tf.layers.conv1d(cnn1, filters=16, kernel_size=24, strides=2, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            cnn2 = tf.layers.max_pooling1d(cnn2, pool_size=2, strides=1)
            # x = cnn2
            x2 = self._flatten(cnn2)
            dn21 = tf.layers.dense(x2, 64, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            dn22 = tf.layers.dense(dn21, self.num_y)
            # if is_train:
            #     dn22 = tf.nn.dropout(dn22, keep_prob=0.6)

            x3 = tf.slice(norm_x, [0, 0, 0], [-1, 15, -1])
            # d1 = tf.layers.dense(batch_x, units=256)
            x3 = self._flatten(x3)
            dn31 = tf.layers.dense(x3, 8, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            dn32 = tf.layers.dense(dn31, self.num_y)
            q = dn32 + dn22
            if self.num_y == 1:
                q = tf.squeeze(q, axis=1)
        return q

    def _define_train(self):
        self.batch_x = tf.placeholder(dt(), [None, *self.input_shape], name='batch_x')
        self.batch_y = tf.placeholder(tf.float32, [None, self.num_y], name='label_y')
        self.eval = self._infer(self.batch_x, 'eval')
        self.pred = self._infer(self.batch_x, 'pred', False)
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_y, logits=self.eval))
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.eval, labels=self.batch_y))
        # self.p_lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.batch_y, logits=self.pred))
        # self.eval = tf.squeeze(self.eval, axis=1)
        # self.pred = tf.squeeze(self.pred)
        self.loss = tf.reduce_mean(tf.squared_difference(x=self.eval, y=self.batch_y))
        self.p_loss = tf.reduce_mean(tf.squared_difference(x=self.pred, y=self.batch_y))
        # self.loss = tf.reduce_mean(tf.where(tf.greater_equal(self.batch_y, 0), tf.where(tf.greater_equal(self.eval, self.batch_y), sd, sd * 5), tf.where(tf.less_equal(self.eval, self.batch_y), sd, sd * 2)))
        lr = tf.train.exponential_decay(self.learn_rate, global_step=self.step, decay_steps=10, decay_rate=self.lr_decay)
        mmt = tf.Variable(1)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self):
        max_step = 5 * self.epoch
        for i in range(max_step):
            for j in range(len(self.obses)):
                obs = self.obses[j]
                ln = len(obs.train_x)
                start = (i * self.batch_size) % ln
                end = min(start + self.batch_size, ln)
                batch_x = obs.train_x[start:end]
                batch_y = obs.train_y[start:end]
                # randidxes = np.random.randint(0, ln, self.batch_size)
                # batch_x = obs.train_x[randidxes]
                # batch_y = obs.train_y[randidxes]
                self.sess.run([self.train_op], feed_dict={self.batch_x: batch_x, self.batch_y: batch_y})
                if i % 10 == 0:
                    loss = self.sess.run(self.loss, feed_dict={self.batch_x: batch_x, self.batch_y: batch_y})
                    log.info('cost: %5.3f', loss)

    def test(self):
        log.info('test:')
        for i in range(len(self.obses)):
            obs = self.obses[i]
            x = obs.train_x
            d_line = obs.train_date
            labels = obs.train_y
            # log.info(x)
            pred = self.sess.run(self.eval, feed_dict={self.batch_x: x})
            # pred = np.argmax(pred, 1)
            for i in range(len(pred)):
                log.info('%s eval:%s y:%s', d_line[i], pred[i], labels[i])
                # precise(obs.obs_y[:self.train_split], pred, self.num_y)

    def predict(self):
        log.info('predict:')
        total_cost = 0.0
        for j in range(len(self.obses)):
            obs = self.obses[j]
            x = obs.test_x
            labels = obs.test_y
            dates = obs.test_date
            pred, p_cost = self.sess.run([self.eval, self.loss], feed_dict={self.batch_x: x, self.batch_y: labels})
            total_cost += p_cost
            # pred_class = np.argmax(pred, 1)
            # labl_class = np.argmax(labels, 1)
            for j in range(len(pred)):
                log.info('%s pred:%s y:%s', dates[j], pred[j], labels[j])
            # p_cost = self.sess.run(self.p_lost, feed_dict={self.pred:pred, self.batch_y:labels})
            # precise(labels, pred, self.num_y)
            # p_cost = np.mean(np.square(labels - pred))
            log.info('pred cost: %.4f', p_cost)
        log.info('total cost: %.4f', total_cost)

    # 应该用渐近式预测，就是预测少量数据，对比cost，然后训练之，再循环
    def continue_test(self):
        log.info('continue test:')
        batch = 5
        all_cost = []
        days = stock_data.Label.days
        for i in range(len(self.obses)):
            obs = self.obses[i]
            tx = obs.test_x
            step = (len(tx) - days) // batch
            if step < 1:
                continue
            total_cost = []
            for j in range(step + 1):
                start = j * batch
                end = min(start + batch, len(tx))
                x = tx[start + days:end + days]
                labels = obs.test_y[start + days:end + days]
                dates = obs.test_date[start + days:end + days]
                # 用后面没有被未来数据污染的数据预测
                pred, loss = self.sess.run([self.eval, self.loss], feed_dict={self.batch_x: x, self.batch_y: labels})
                total_cost.append(loss)
                # 用前面的数据训练
                for _ in range(1):
                    self.sess.run(self.train_op, feed_dict={self.batch_x: tx[start:end], self.batch_y: obs.test_y[start:end]})
                for t in range(len(pred)):
                    log.info('%s ctest:%s y:%s', dates[t], pred[t], labels[t])
            log.info('pred cost: %.4f', np.mean(total_cost))
            all_cost.append(np.mean(total_cost))
        log.info('total cost: %.4f', np.mean(all_cost))


if __name__ == '__main__':
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    # print(len(obs_x))
    obses = []
    for i in range(1, 20):
        try:
            obs = stock_data.prepare_single(i)
            obses.append(obs)
        except Exception as e:
            print(e)
            continue
    # obses.append(stock_data.prepare_single(2))
    # obses.append(stock_data.prepare_single(5))
    enet = EvalNet(obses)
    enet.train()
    enet.test()
    enet.predict()
    # enet.continue_test()
    enet.sess.close()
