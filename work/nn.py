import tensorflow as tf
import logging as log
import keras as kr
from keras.layers import Input, Dense, Embedding, Permute, multiply,Bidirectional
from keras.models import Model, Sequential

def attention_3d(inputs, time_step):
    # [bsz, time_step, emb]->[bsz, emb, time_step]
    a = Permute((2, 1))(inputs)
    # W=[time_step, time_step], a=[bsz, emb, time_step]
    a = Dense(time_step, activation='softmax')(a)
    # [bsz, time_step, emb]
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    # 逐元素相乘
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


class TitleRecKr(object):
    def __init__(self, cell_unit, max_len, emb_dim, learn_rate = 0.02, batch_size = 30, epoch = 100):
        self._cell_unit = cell_unit
        self._bsz = batch_size
        self._lr = learn_rate
        self._emb_dim = emb_dim
        self._max_len = max_len
        self._epoch = epoch
        self.build_model_kr()

    def build_model_kr(self):
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Dense(self._cell_unit, tf.nn.relu))
        # cell = tf.keras.layers.GRU(self._cell_unit)
        # bi_rnn = tf.keras.layers.Bidirectional(cell)
        # model.add(bi_rnn)
        # model.add(tf.keras.layers.Dense(1))
        #
        # output = model(self.x)
        # return output
        model = Sequential()
        model.add(Dense(self._cell_unit, input_dim=(self._max_len, self._emb_dim)))
        # gru = kr.layers.CuDNNGRU(self._cell_unit)
        gru = kr.layers.GRU(self._cell_unit)
        bi_rnn = Bidirectional(gru)
        model.add(bi_rnn)
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model

    def build_model_fn(self):
        x = Input([self._max_len, self._emb_dim])
        den = Dense(self._cell_unit)(x)
        rnn = Bidirectional(kr.layers.CuDNNGRU(self._cell_unit))(den)
        out = Dense(1)(rnn)
        model = Model(x, out)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model

    def train(self, x, y):
        self.model.fit(x, y, batch_size=self._bsz, epochs=self._epoch,
            verbose=2, validation_split=0.7)

    def predict(self, x):
        return self.model.predict(x)


class TitleRec(object):
    def __init__(self, cell_unit, max_len, emb_dim, learn_rate = 0.02, batch_size = 30, epoch = 100):
        self._cell_unit = cell_unit
        self._bsz = batch_size
        self._lr = learn_rate
        self._emb_dim = emb_dim
        self._max_len = max_len
        self._epoch = epoch
        self.build_input()
        self.y = self.build_tf()
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=self.y))
        self.train_op = tf.train.AdamOptimizer(self._lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_input(self):
        self.x = tf.placeholder(tf.float32, [None, self._max_len, self._emb_dim])

    def train(self, x, y):
        data_len = len(x)
        for i in range(self._epoch):
            start = (i * self._bsz) % data_len
            end = min(start + self._bsz, data_len)
            tx = x[start:end]
            ty = y[start:end]
            _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.x: tx, self.y: ty})
            log.info('%d cost: %s', i, str(cost))

    def predict(self, x):
        out = self.sess.run(self.y, feed_dict={self.x: x})
        return out

    def build_tf(self):
        # 转为cell_unit大小
        den = tf.layers.Dense(self._cell_unit, activation=tf.nn.relu)
        x = den(self.x)
        cell_fw = tf.nn.rnn_cell.GRUCell(self._cell_unit)
        init_state_fw = cell_fw.zero_state(self._bsz, tf.float32)
        cell_bw = tf.nn.rnn_cell.GRUCell(self._cell_unit)
        init_state_bw = cell_bw.zero_state(self._bsz, tf.float32)
        # mul_rnn = tf.nn.rnn_cell.MultiRNNCell()
        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length=self._bsz, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
        out = tf.concat(outputs, 2)
        y = tf.layers.Dense(1)(out)
        return y


