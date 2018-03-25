from header import *


class AutoEncoder(object):
    def __init__(self, obs, labels, d_line, train_rate = 5):
        self.learn_rate = 0.008
        self.lr_decay = 0.9
        self.obs = obs
        self.labels = labels
        self.d_line = d_line
        self.input_shape = np.shape(obs[0])
        self.batch_size = 50
        self.epoch = 10
        self.train_split = len(obs) // 10 * train_rate
        # self.train_split = 100
        self.step = tf.Variable(0, trainable=False)
        self._define_train()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _encoder(self, flatten_x):
        bias_initer = tf.constant_initializer(0.1)
        kernel_initer = tf.random_normal_initializer(0, 0.5)
        scope = 'auto'
        # with tf.variable_scope(scope):
        # x = cnn2
        # x = batch_x
        # d1 = tf.layers.dense(batch_x, units=256)
        dn1 = tf.layers.dense(flatten_x, 16, activation=None, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
        # dn2 = tf.layers.dense(dn1, 512, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
        # dn3 = tf.layers.dense(dn2, 256)
        # q = tf.layers.dense(dn1, 1)
        return dn1

    def _decoder(self, enc):
        dec = tf.layers.dense(enc, self.num_enc_x, activation=tf.nn.relu)
        return dec

    def _define_train(self):
        self.batch_x = tf.placeholder(dt(), [None, *self.input_shape], name='batch_x')
        self.shape_x = self.batch_x.get_shape().as_list()
        self.num_enc_x = 1
        for i in range(1, len(self.shape_x)):
            self.num_enc_x *= self.shape_x[i]
        self.flatten_x = tf.reshape(self.batch_x, [-1, self.num_enc_x])
        self.enc = self._encoder(self.flatten_x)
        self.dec = self._decoder(self.enc)
        sd = tf.squared_difference(x=self.flatten_x, y=self.dec)*10
        self.loss = tf.reduce_mean(sd)
        lr = tf.train.exponential_decay(self.learn_rate, global_step=self.step, decay_steps=10, decay_rate=self.lr_decay)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self):
        max_step = 100 * self.epoch
        for i in range(max_step):
            randidx = np.random.randint(0, self.train_split, self.batch_size)
            # start = (i * self.batch_size) % self.train_split
            # end = min(start + self.batch_size, self.train_split)
            batch_x = self.obs[randidx]
            self.sess.run([self.train_op], feed_dict={self.batch_x: batch_x})
            if i % 10 == 0:
                loss = self.sess.run(self.loss, feed_dict={self.batch_x: batch_x})
                log.info('cost: %5.4f', loss)

    def test(self):
        log.info('test:')
        x = self.obs[:self.train_split]
        # log.info(x)
        # enc = self.sess.run(self.enc, feed_dict={self.batch_x: x})
        # for i in range(len(enc)):
        #     log.info(enc[i])

    def predict(self):
        log.info('predict:')
        flatten = self.sess.run(self.flatten_x, feed_dict={self.batch_x: self.obs[self.train_split:]})
        labels = self.sess.run(self.dec, feed_dict={self.batch_x: self.obs[self.train_split:]})
        d_line = self.d_line[self.train_split:]
        # for i in range(len(enc)):
        #     log.info(enc[i])
        p_cost = np.mean(np.square(labels - flatten))
        log.info('pred cost: %.4f', p_cost)


if __name__ == '__main__':

    d_line, k_line, obs_x = stock_data.prepare_single(5)
    label = stock_data.Label(k_line=k_line, d_line=d_line)
    label.calc_up()
    enet = AutoEncoder(obs_x, labels=label.up_table, d_line=d_line)
    enet.train()
    # enet.test()
    enet.predict()
    enet.sess.close()
