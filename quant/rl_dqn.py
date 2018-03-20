'''dqn '''
from header import *
from stock_simulator import Simulator


def dtype():
    return tf.float32


class Dqn(object):
    def __init__(self, n_act, obs_shape):
        self.learn_rate = 0.1
        self.decay = 0.9
        self.input_shape = obs_shape
        self.num_act = n_act
        self.rand_gate = 0.1
        self.rand_gate_max = 0.9
        self._build_dqn()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.batch_size = 5
        self.step = 1
        self.l_obs, self.l_stores = [], []
        self.l_act, self.l_reward = [], []
        self.l_next_obs, self.l_next_stores = [], []

    def _dtype(self):
        return tf.float16

    def _build_net_k(self, obs, scope):
        with tf.variable_scope(scope):
            cnn1 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, input_shape=self.input_shape, padding='valid')(obs)
            print(tf.shape(cnn1))
            pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(cnn1)
            cnn2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='valid')(pool1)
            print(tf.shape(cnn2))
            pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(cnn2)
            dnn = tf.keras.layers.Dense(units=256, activation='relu')(pool2)
            print(tf.shape(dnn))
            out = tf.keras.layers.Dense(units=self.num_act)(dnn)
            print(tf.shape(out))
        return out

    def _build_net(self, obs, store, scope):
        with tf.variable_scope(scope):
            cnn1 = tf.layers.conv1d(obs, filters=16, kernel_size=5)
            pool1 = tf.layers.max_pooling1d(cnn1, pool_size=2, strides=1)
            cnn2 = tf.layers.conv1d(pool1, filters=16, kernel_size=5)
            pool2 = tf.layers.max_pooling1d(cnn2, pool_size=2, strides=1)
            x = pool2
            shape_x = x.get_shape().as_list()
            if len(shape_x) > 2:  # 自动把输入转为扁平
                num = 1
                for i in range(1, len(shape_x)):
                    num *= shape_x[i]
                x = tf.reshape(x, [-1, num])

            x = tf.concat([x, tf.expand_dims(store, 1)], 1)
            # pool1 = tf.reshape(pool1, )
            dnn = tf.layers.dense(x, 128, activation=tf.nn.relu)
            out = tf.layers.dense(dnn, self.num_act)
        return out

    def _build_dqn(self):
        self.samples = tf.placeholder(dtype(), [None, *self.input_shape], name='samples')
        self.stores = tf.placeholder(dtype(), [None], name='stores')
        self.q_target = tf.placeholder(dtype(), [None, self.num_act], name='q_target')

        self.q_eval = self._build_net(self.samples, self.stores, 'eval')
        self.loss = tf.reduce_mean(tf.squared_difference(x=self.q_eval, y=self.q_target))
        self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)
        self.q_pred = self._build_net(self.samples, self.stores, 'pred')
        eval_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        pred_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred')
        move_decay = 0.5
        # self.replace = [tf.assign(e, (1 - move_decay) * e + move_decay * p) for e, p in zip(eval_para, pred_para)]
        self.replace = [tf.assign(p, e) for p, e in zip(pred_para, eval_para)]

    def store_transition(self, obs, store, act, reward, next_obs, next_store):
        self.l_obs.append(obs)
        self.l_stores.append(store)
        self.l_act.append(act)
        self.l_reward.append(reward)
        self.l_next_obs.append(next_obs)
        self.l_next_stores.append(next_store)

    def choose_action(self, obs, store):
        if np.random.uniform() < self.rand_gate:
            obs = obs[np.newaxis, :]
            store = store[np.newaxis, :]
            acts = self.sess.run(self.q_eval, feed_dict={self.samples: obs, self.stores: store})
            act = np.argmax(acts)  # 这里不指定纬度，[[0,1,2]]->2
        else:
            act = np.random.randint(0, self.num_act)
        self.increase_rand_gate()
        return act

    def increase_rand_gate(self):
        self.rand_gate += 0.0001
        if self.rand_gate > self.rand_gate_max:
            self.rand_gate = self.rand_gate_max

    def learn(self):
        if self.step % 20 == 0:
            self.sess.run(self.replace)

        q_predict = self.sess.run(self.q_pred, feed_dict={self.samples: self.l_next_obs, self.stores: self.l_next_stores})
        q_target = self.sess.run(self.q_eval, feed_dict={self.samples: self.l_obs, self.stores: self.l_stores})
        num = len(self.l_reward)
        batch_indexes = np.arange(num, dtype=np.int32)
        q_target[batch_indexes, self.l_act] = self.l_reward + self.decay * np.max(q_predict, axis=1)

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.samples: self.l_obs, self.stores: self.l_stores, self.q_target: q_target})

        self.step += 1
        return cost
