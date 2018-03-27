'''dqn '''
from qhead import *
from stock_simulator import Simulator


def dtype():
    return tf.float32


class Dqn(object):
    def __init__(self, n_act, obs):
        self.learn_rate = 0.001
        self.decay = 0.9
        self.input_shape = np.shape(obs)[1:]
        self.num_act = n_act
        self.rand_gate = 0.0
        self.rand_gate_max = 0.9
        # self.search_table = np.zeros([len(obs), 2, 2], dtype=np.int16)
        # self.reward_table = np.zeros([len(obs), 2, 2], dtype=np.float32)
        self.obs = obs
        # self.k_line = k_line
        self.step = 1
        self._build_dqn()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._reset_samples()
        # self.sim = Simulator(k_line)

    def _reset_samples(self):
        # 只是占个位置
        self.l_obs, self.l_next_obs = [], []
        self.l_store, self.l_next_store = [], []
        self.l_act, self.l_reward = [], []

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
        bias_initer = tf.constant_initializer(0.01)
        kernel_initer = tf.random_normal_initializer(0, 0.1)
        with tf.variable_scope(scope):
            cnn1 = tf.layers.conv1d(obs, filters=16, kernel_size=5, strides=1, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            pool1 = tf.layers.max_pooling1d(cnn1, pool_size=2, strides=1)
            # pool1 = cnn1
            cnn2 = tf.layers.conv1d(pool1, filters=32, kernel_size=5, strides=1, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            pool2 = tf.layers.max_pooling1d(cnn2, pool_size=2, strides=1)
            # x = cnn2
            x = pool2
            shape_x = x.get_shape().as_list()
            if len(shape_x) > 2:  # 自动把输入转为扁平
                num = 1
                for i in range(1, len(shape_x)):
                    num *= shape_x[i]
                x = tf.reshape(x, [-1, num])

            x = tf.concat([x, tf.expand_dims(store, 1)], 1)
            # pool1 = tf.reshape(pool1, )
            dnn = tf.layers.dense(x, 256, activation=tf.nn.relu, kernel_initializer=kernel_initer, bias_initializer=bias_initer)
            state = tf.layers.dense(dnn, 1)
            adv = tf.layers.dense(dnn, self.num_act)
            out = state + (adv - tf.reduce_mean(adv, axis=1, keep_dims=True))
        return out

    def _build_dqn(self):
        self.x = tf.placeholder(dtype(), [None, *self.input_shape], name='samples')
        # self.k_line = self.obs[:10]
        self.store = tf.placeholder(dtype(), [None], name='stores')
        self.q_target = tf.placeholder(dtype(), [None, self.num_act], name='q_target')

        self.q_eval = self._build_net(self.x, self.store, 'eval')
        self.loss = tf.reduce_mean(tf.squared_difference(x=self.q_eval, y=self.q_target))
        self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)
        self.q_pred = self._build_net(self.x, self.store, 'pred')
        eval_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        pred_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred')
        move_decay = 0.5
        # self.replace = [tf.assign(e, (1 - move_decay) * e + move_decay * p) for e, p in zip(eval_para, pred_para)]
        self.replace = [tf.assign(p, e) for p, e in zip(pred_para, eval_para)]

    def store_transition(self, idx, store, act, reward, next_store):
        # obs = tf.expand_dims(self.obs[idx], axis=0)
        # nxt_obs = tf.expand_dims(self.obs[idx + 1], axis=0)
        obs = self.obs[idx][np.newaxis, :]
        nxt_obs = self.obs[idx + 1][np.newaxis, :]
        if len(self.l_obs) == 0:
            self.l_obs = obs
            self.l_next_obs = obs
        else:
            self.l_obs = np.append(self.l_obs, obs, axis=0)
            self.l_next_obs = np.append(self.l_next_obs, nxt_obs, axis=0)

        self.l_store.append(store)
        self.l_act.append(act)
        self.l_reward.append(reward)
        self.l_next_store.append(next_store)

    def train_action(self, idx, store):
        # 在训练时搜索探索空间的选择
        if np.random.uniform() < self.rand_gate:
            obs = self.obs[idx]
            act = self.pred_action(obs, store)
            # # 如果两个act的reward差距过大，则不再探索
            # if abs(acts[0]-acts[1]) > 2.0:
        else:
            act = np.random.randint(0, self.num_act)

        return act

    def pred_action(self, obs, store):
        # 用于eval和pred的act选择
        # obs = self.obs[idx][np.newaxis, :]
        obs = obs[np.newaxis, :]
        store = [store]
        acts = self.sess.run(self.q_eval, feed_dict={self.x: obs, self.store: store})
        # print('acts:', np.squeeze(acts))
        act = np.argmax(acts)  # 这里不指定纬度，[[0,1,2]]->2
        return act

    def increase_rand_gate(self):
        self.rand_gate += 0.001
        if self.rand_gate > self.rand_gate_max:
            self.rand_gate = self.rand_gate_max

    def learn(self):
        if self.step % 100 == 0:
            self.sess.run(self.replace)

        q_predict = self.sess.run(self.q_pred, feed_dict={self.x: self.l_next_obs, self.store: self.l_next_store})
        q_target = self.sess.run(self.q_eval, feed_dict={self.x: self.l_obs, self.store: self.l_store})
        num = len(self.l_reward)
        batch_indexes = np.arange(num, dtype=np.int32)
        # q_target[batch_indexes, self.l_act] = self.l_reward + self.decay * np.clip(np.max(q_predict, axis=1), a_min=0, a_max=10)
        pred_val = np.max(q_predict, axis=1)
        # for i in range(num):
        #     if self.l_act[i] == 0:
        #         pred_val[i] = 0
        # q_target /= 2 # 防止q值无限增大
        q_target[batch_indexes, self.l_act] = self.l_reward + self.decay * pred_val
        if self.step % 10 == 0:
            print('q_target:', q_target)

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.x: self.l_obs, self.store: self.l_store, self.q_target: q_target})

        self.step += 1
        self._reset_samples()
        # self.increase_rand_gate()
        return cost
