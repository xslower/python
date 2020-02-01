# coding=utf-8
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from .kr_plus import *

class FfmArgs(object):
    # number of latent factors
    k = 10
    # num of fields
    f = 3
    # num of features
    p = 100
    learning_rate = 0.1
    batch_size = 128
    l2_reg_rate = 0.001
    feature2field = None
    checkpoint_dir = '../ckpt/ffm/'
    is_training = True
    epoch = 100
    stop_key_val = 0.1 #loss or binary_accuracy value


class FFM(object):
    def __init__(self, args):
        self.k = args.k
        self.f = args.f
        self.p = args.p
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.l2_reg_rate = args.l2_reg_rate
        self.feature2field = args.feature2field
        self.checkpoint_dir = args.checkpoint_dir
        self.min_loss = args.min_loss
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        try:
            self.restore()
        except Exception as e:
            print(e)
        # self.save()

    def build_model(self):
        self.X = tf.placeholder('float32', [None, self.p])
        self.y = tf.placeholder('float32', [None, 1])

        # linear part
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[1],
                                initializer=tf.zeros_initializer())
            self.w1 = tf.get_variable('w1', shape=[self.p, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.linear_terms = tf.add(tf.matmul(self.X, self.w1), b)
            print('self.linear_terms:')
            print(self.linear_terms)

        with tf.variable_scope('nolinear_layer'):
            self.v = tf.get_variable('v', shape=[self.p, self.f, self.k], dtype='float32',
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # v:pxfxk
            self.field_cross_interaction = tf.constant(0, dtype='float32')
            # 每个特征, 这里完全是最开始的公式，与fm不同没有简化
            for i in range(self.p):
                # 寻找没有match过的特征，也就是论文中的j = i+1开始
                print('i:%s' % (i))
                for j in range(i + 1, self.p):
                    # vifj
                    vifj = self.v[i, self.feature2field[j]]
                    # vjfi
                    vjfi = self.v[j, self.feature2field[i]]
                    # vi · vj
                    vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
                    # xi · xj
                    xixj = tf.multiply(self.X[:, i], self.X[:, j])
                    self.field_cross_interaction += tf.multiply(vivj, xixj)
            self.field_cross_interaction = tf.reshape(self.field_cross_interaction, (-1, 1))
            print('self.field_cross_interaction:')
            print(self.field_cross_interaction)
        self.y_out = tf.add(self.linear_terms, self.field_cross_interaction)
        print('y_out_prob:')
        print(self.y_out)
        # -1/1情况下的logistic loss
        self.loss = tf.reduce_mean(tf.log(1 + tf.exp(-self.y * self.y_out)))

        # 正则：sum(w^2)/2*l2_reg_rate
        # 这边只加了weight，有需要的可以加上bias部分
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.w1)
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.v)
        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def _train(self, x, label):
        loss, _, step = self.sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.X: x,
            self.y: label
        })
        return loss, step

    def train(self, train_x, train_y):
        print(train_x.shape, train_y.shape)
        bsz = self.batch_size
        cnt = train_x.shape[0] // bsz
        # batch_size data
        for i in range(self.epoch):
            for j in range(cnt):
                x = get_batch(train_x, bsz, j)
                y = get_batch(train_y, bsz, j)
                y = np.reshape(y, (bsz, 1))
                print(x.shape, y.shape)
                # actual_batch_size = len(x)
                # batch_X = []
                # batch_y = []
                # for k in range(actual_batch_size):
                #     sample = x.iloc[k, :]
                #     array = transfer_data(sample, fields_dict, all_len)
                #     # 最后两位[0,-1]:label=0,[0,1]:label=1
                #     batch_X.append(array[:-2])
                #     # 最后一位即为label
                #     batch_y.append(array[-1])
                # batch_X = np.array(x)
                # batch_y = np.array(y)
                loss, step = self._train(x, y)
                if j % 100 == 0:
                    print('the times of training is %d, and the loss is %s' % (j, loss))
                    self.save()
                    if loss < self.min_loss:
                        break
                    # r1, r2 = model.cal(sess, batch_X, batch_y)
                    # print(r1)
                    # print(r2)

    def cal(self, x, label):
        y_out_prob_ = self.sess.run([self.y_out], feed_dict={
            self.X: x,
            self.y: label
        })
        return y_out_prob_, label

    def _predict(self, x):
        y = self.sess.run([self.y_out], feed_dict={
            self.X: x
        })
        y = np.reshape(y, (len(x)))
        return y

    def predict(self, x):
        bsz = self.batch_size*2
        cnt = x.shape[0] // bsz
        y = []
        for j in range(cnt+1):
            data = get_batch(x, bsz, j)
            result = self._predict(data)
            y.append(result)
        y = np.concatenate(y, axis=0)
        return y

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=self.checkpoint_dir)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path=self.checkpoint_dir)



class FMkr(object):
    def __init__(self, args):
        # self.k = args.k
        # self.f = args.f
        # self.p = args.p
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.l2_reg_rate = args.l2_reg_rate
        self.feature2field = args.feature2field
        self.checkpoint_dir = args.checkpoint_dir
        # self.stop_loss = args.min_loss
        self.stop_key_val = getattr(args, 'stop_key_val', None)
        self.input_type= args.input_type #['o', 'v', 'v', 'v']
        self.input_max= args.input_max #[100]
        self.emb_len = args.emb_len #10
        self.graph = None
        self.dtype = 'float32'
        self.c_onehot = 'o'
        self.c_float = 'v'
        self.build_model()
        sgd = optimizers.Adam(lr=self.learning_rate, decay=1e-6)
        # sgd = optimizers.RMSprop(lr=self._lr)
        # loss = kr.losses.binary_crossentropy()
        # self.model.compile(optimizer=sgd, loss='mae', metrics=['mae'])
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
        self.model.summary()
        try:
            self.load_weights()
        except:
            pass

    def build_model(self):
        x, fm, cc = self.build_fm()
        out = Dense(1, activation='sigmoid')(fm)
        self.model = krm.Model(x, out)

    def build_fm(self):
        x = [0] * len(self.input_type)
        emb2 = [0] * len(x)
        emb3 = [0] * len(x)
        for i in range(len(x)):
            x[i] = Input([1], name='input_%i' % i)
            if self.input_type[i] == self.c_onehot:
                # bsz * 1 * 1
                t2 = Embedding(self.input_max[i], 1, dtype=self.dtype)(x[i])
                emb2[i] = Flatten()(t2)
                # bsz * 1 * emb_len
                emb3[i] = Embedding(self.input_max[i], self.emb_len, dtype=self.dtype)(x[i])
            elif self.input_type[i] == self.c_float:
                # bsz * 1
                emb2[i] = Dense(1)(x[i])
                # 这里一个value映射到k个值上, bsz * emb_len
                t3 = Dense(self.emb_len)(x[i])
                # bsz * 1 * emb_len 保持一致
                emb3[i] = Reshape((1, -1))(t3)
        # bsz * 1
        part2 = Add(name='part2')(emb2)
        # bsz * f * emb_len
        cc = Concatenate(axis=1)(emb3)
        sqare = Multiply()([cc, cc])
        # bsz * emb_len
        sumed_sq = MySum(axis=1)(sqare)
        # bsz * emb_len
        sumed = MySum(axis=1)(cc)
        sqare_sm = Multiply()([sumed, sumed])
        part3 = Subtract()([sqare_sm, sumed_sq])
        part3 = Lambda(lambda x: x * 0.5)(part3)
        # bsz * 1
        part3 = MySum(axis=1, name='part3')(part3)
        fm = Add(name='add')([part2, part3])
        return x, fm, cc

    def parse_x(self, x):
        ln = len(self.input_type)
        nx = [0] * ln
        for i in range(ln):
            nx[i] = x[:, i]
        return nx

    def _train(self, x, y):
        x = self.parse_x(x)
        monitor_key = 'val_loss'
        # monitor_key = 'binary_accuracy'
        lm = LossMonitor(monitor_key)
        early_stop = kr.callbacks.EarlyStopping(monitor=monitor_key, patience=4, verbose=1, mode='auto', restore_best_weights=True)
        cbs = [early_stop, lm]
        # if self.stop_key_val is not None:
        #     target_stop = EStopping(monitor=monitor_key, verbose=1, baseline=self.stop_key_val)
        #     cbs.append(target_stop)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epoch, verbose=2, callbacks=cbs, shuffle=True, validation_split=0.1)
        return lm.get_loss()

    def train(self, train_x, train_y):
        print(train_x.shape, train_y.shape)
        return self._train(train_x, train_y)

    def _predict(self, x):
        ln = x.shape[0]
        x = self.parse_x(x)
        bsz = self.batch_size
        if ln > 100*bsz:
            bsz = 100*bsz

        y = self.model.predict(x, batch_size=ln)
        y = np.reshape(y, [ln])
        return y

    def predict(self, x):
        if self.graph is not None:
            with self.graph.as_default():
                return self._predict(x)
        else:
            return self._predict(x)

    def save_weights(self, ):
        f = self.checkpoint_dir+'fm.krw'
        self.model.save_weights(f)

    def save_model(self):
        f = self.checkpoint_dir+'fm.krm'
        self.model.save(f)

    def load_weights(self):
        f = self.checkpoint_dir + 'fm.krw'
        self.model.load_weights(f)
        self.graph = tf.get_default_graph()

class DeepFMkr(FMkr):
    def __init__(self, args):
        super().__init__(args)

    def single_deep(self):
        x = [0] * len(self.input_type)
        emb3 = [0] * len(x)
        for i in range(len(x)):
            x[i] = Input([1], name='input_%i' % i)
            if self.input_type[i] == self.c_onehot:
                # bsz * 1 * emb_len
                emb3[i] = Embedding(self.input_max[i], self.emb_len, dtype=self.dtype)(x[i])
            elif self.input_type[i] == self.c_float:
                # 这里一个value映射到k个值上, bsz * emb_len
                t3 = Dense(self.emb_len)(x[i])
                # bsz * 1 * emb_len 保持一致
                emb3[i] = Reshape((1, -1))(t3)
        cc = Concatenate(axis=-1)(emb3)
        cc = Flatten()(cc)
        cc = Dense(4, activation='relu')(cc)
        out = Dense(1, activation='sigmoid', name='deep-out')(cc)
        self.model = krm.Model(x, out)

    def build_deepfm(self):
        x = [0]*len(self.input_type)
        emb2 = [0]*len(x)
        emb3 = [0]*len(x)
        for i in range(len(x)):
            x[i] = Input([1], name='input_%i'%i)
            if self.input_type[i] == self.c_onehot:
                # bsz * 1 * 1
                t2 = Embedding(self.input_max[i], 1, dtype=self.dtype)(x[i])
                emb2[i] = Flatten()(t2)
                # bsz * 1 * emb_len
                emb3[i] = Embedding(self.input_max[i], self.emb_len, dtype=self.dtype)(x[i])
            elif self.input_type[i] == self.c_float:
                # bsz * 1
                emb2[i] = Dense(1)(x[i])
                # 这里一个value映射到k个值上, bsz * emb_len
                t3 = Dense(self.emb_len)(x[i])
                # bsz * 1 * emb_len 保持一致
                emb3[i] = Reshape((1,-1))(t3)
        # bsz * 1
        part2 = Add(name='part2')(emb2)
        # bsz * f * emb_len
        cc = Concatenate(axis=1)(emb3)
        sqare = Multiply()([cc,cc])
        # bsz * emb_len
        sumed_sq = MySum(axis=1)(sqare)
        # bsz * emb_len
        sumed = MySum(axis=1)(cc)
        sqare_sm = Multiply()([sumed, sumed])
        part3 = Subtract()([sqare_sm, sumed_sq])
        part3 = Lambda(lambda x: x * 0.5)(part3)
        # bsz * 1
        part3 = MySum(axis=1, name='part3')(part3)
        fm = Add(name='add')([part2, part3])
        # deep part
        deep = Flatten()(cc)
        deep = Dense(4, activation='relu')(deep)
        deep = Dense(1, activation='sigmoid')(deep)
        out = Concatenate()([fm, deep])
        out = Dense(1, activation='sigmoid', name='y-out')(out)
        self.model = krm.Model(x, out)

    def build_model(self):
        x, fm, cc = self.build_fm()
        # deep part
        deep = Flatten()(cc)
        deep = Dense(4, activation='relu')(deep)
        deep = Dense(1, activation='sigmoid')(deep)
        out = Concatenate()([fm, deep])
        out = Dense(1, activation='sigmoid', name='y-out')(out)
        self.model = krm.Model(x, out)

def get_batch(x, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < x.shape[0] else x.shape[0]
    # print(start, end)
    return x[start:end]
    # return x.iloc[start:end, :]

def test():
    # loading base params
    args = FfmArgs()
    # else:
    #     model.restore(sess, args.checkpoint_dir)
    #     for j in range(cnt):
    #         data = get_batch(train_data, args.batch_size, j)
    #         actual_batch_size = len(data)
    #         batch_X = []
    #         for k in range(actual_batch_size):
    #             sample = data.iloc[k, :]
    #             array = transfer_data(sample, fields_dict, all_len)
    #             batch_X.append(array[:-2])
    #         batch_X = np.array(batch_X)
    #         result = model.predict(sess, batch_X)
    #         print(result)
