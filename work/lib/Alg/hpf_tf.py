import numpy as np
import pandas as pd
from scipy.special import digamma, gamma
import logging
import json
import os
# from tqdm import trange
import tensorflow as tf

from .kr_plus import *


class HPFtf(object):
    """
    """

    def __init__(self, u_max, g_max, k = 30, a = 0.3, a_prime = 0.3, b_prime = 1.0, c = 0.3, c_prime = 0.3, d_prime = 1.0, log_f = None):
        self.k = k
        self.a = a
        self.a_prime = a_prime
        self.b_prime = b_prime
        self.c = c
        self.c_prime = c_prime
        self.d_prime = d_prime
        self._log_f = log_f
        self.last_llk = 1e-30
        self.total_llk = 0
        self.total_rmse = 0
        self.total = 0
        self.u_max = u_max
        self.g_max = g_max
        self.u_mapper = IdxMap('data/user_mapper.pkl')
        self.g_mapper = IdxMap('data/item_mapper.pkl')
        self._build_graph()
        self.folder = 'ckpt'
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.6
        gpu_config.gpu_options.allow_growth = True
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        # self.saver = tf.train.Saver([self.Theta, self.Beta])
        try:
            self.load()
        except Exception as e:
            self._log_f(e)

    def _remove_zeros(self, df):
        return df[lambda x: x.ratings > 0]

    def _generate_item_scores(self, df):
        return np.array(df.ratings.tolist(), dtype=np.float32)

    def _initialize_parameters(self):
        n_user = self.u_max
        n_item = self.g_max
        self.input_u = tf.placeholder(tf.int32, [None], name='input_u')
        self.input_g = tf.placeholder(tf.int32, [None], name='input_g')
        self.input_r = tf.placeholder(tf.float32, [None], name='input_r')
        # # n_user * 1
        # ksi = np.random.gamma(self.a_prime, self.b_prime / self.a_prime, size=(n_user, 1))
        # # # nu * k
        # self.Theta = np.random.gamma(self.a, 1 / ksi, size=(n_user, self.k))
        # # # ni * 1
        # eta = np.random.gamma(self.c_prime, self.d_prime / self.c_prime, size=(n_item, 1))
        # # # ni * k
        # self.Beta = np.random.gamma(self.c, 1 / eta, size=(n_item, self.k))
        self.Theta = np.random.gamma(self.a, 1 / self.b_prime, size=(n_user, self.k))
        self.Beta = np.random.gamma(self.c, 1 / self.d_prime, size=(n_item, self.k))
        # 1
        self.k_shp = self.a_prime + self.k * self.a
        # 1
        self.t_shp = self.c_prime + self.k * self.c
        # nu * 1
        self.k_rte = self.a_prime / self.b_prime + self.Theta.sum(axis=1, keepdims=True)
        # ni * 1
        self.t_rte = self.c_prime / self.d_prime + self.Beta.sum(axis=1, keepdims=True)
        # nu * k
        self.Gamma_rte = np.random.gamma(self.a_prime, self.b_prime / self.a_prime, size=(n_user, 1)) + self.Beta.sum(axis=0, keepdims=True)
        # nu * k
        self.Gamma_shp = self.Gamma_rte * self.Theta * np.random.uniform(low=0.85, high=1.15, size=(n_user, self.k))

        self.gamma_shp_tmp_init = np.zeros((n_user, self.k)) + self.a
        # ni * k
        self.Lambda_rte = np.random.gamma(self.c_prime, self.d_prime / self.c_prime, size=(n_item, 1)) + self.Theta.sum(axis=0, keepdims=True)
        # ni * k
        self.Lambda_shp = self.Lambda_rte * self.Beta * np.random.uniform(low=0.85, high=1.15, size=(n_item, self.k))
        self.lambda_shp_tmp_init = np.zeros((n_item, self.k)) + self.c

        # 1
        self.add_k_rte = self.a_prime / self.b_prime
        self.add_t_rte = self.c_prime / self.d_prime
        # 转为tf变量，不然无法assign
        dt = tf.float32
        self.Theta = tf.Variable(self.Theta, dtype=dt)
        self.Beta = tf.Variable(self.Beta, dtype=dt)
        self.k_rte = tf.Variable(self.k_rte, dtype=dt)
        self.t_rte = tf.Variable(self.t_rte, dtype=dt)
        self.Gamma_rte = tf.Variable(self.Gamma_rte, dtype=dt)
        self.Gamma_shp = tf.Variable(self.Gamma_shp, dtype=dt)
        self.Lambda_rte = tf.Variable(self.Lambda_rte, dtype=dt)
        self.Lambda_shp = tf.Variable(self.Lambda_shp, dtype=dt)
        # 用于分批更新psi
        self.gamma_shp_tmp = tf.Variable(self.gamma_shp_tmp_init, dtype=dt)
        self.lambda_shp_tmp = tf.Variable(self.lambda_shp_tmp_init, dtype=dt)

    def _update_phi(self):
        self._log_f("Update phi ...")
        gamma_shp = tf.nn.embedding_lookup(self.Gamma_shp, self.input_u)
        gamma_rte = tf.nn.embedding_lookup(self.Gamma_rte, self.input_u)
        lambda_shp = tf.nn.embedding_lookup(self.Lambda_shp, self.input_g)
        lambda_rte = tf.nn.embedding_lookup(self.Lambda_rte, self.input_g)
        self.phi = tf.exp(tf.digamma(gamma_shp) - tf.log(gamma_rte) + tf.digamma(lambda_shp) - tf.log(lambda_rte))
        sumphi = tf.reduce_sum(self.phi, axis=1, keepdims=True)
        self.phi = self.phi * tf.expand_dims(self.input_r, axis=-1) / sumphi

    # 暂不支持，batch训练，需输入全部数据
    def _build_graph(self):
        self._initialize_parameters()
        self._update_phi()
        self.gamma_shp_tmp2 = tf.scatter_add(self.gamma_shp_tmp, self.input_u, self.phi)
        self.lambda_shp_tmp2 = tf.scatter_add(self.lambda_shp_tmp, self.input_g, self.phi)
        self.batch_val = [self.gamma_shp_tmp2, self.lambda_shp_tmp2]
        self.batch_op = [tf.assign(self.gamma_shp_tmp, self.gamma_shp_tmp2), tf.assign(self.lambda_shp_tmp, self.lambda_shp_tmp2)]
        self.gamma_rte_tmp = tf.divide(tf.ones((self.u_max, self.k)) * self.k_shp, self.k_rte) + tf.reduce_sum(self.Beta, axis=0, keepdims=True)
        self.lambda_rte_tmp = tf.divide(tf.ones((self.g_max, self.k)) * self.t_shp, self.t_rte) + tf.reduce_sum(self.Theta, axis=0, keepdims=True)
        self.theta_tmp = self.gamma_shp_tmp / self.gamma_rte_tmp
        self.beta_tmp = self.lambda_shp_tmp / self.lambda_rte_tmp
        # nu * 1
        self.k_rte_tmp = self.add_k_rte + tf.reduce_sum(self.Theta, axis=1, keepdims=True)
        # ni * 1
        self.t_rte_tmp = self.add_t_rte + tf.reduce_sum(self.Beta, axis=1, keepdims=True)
        self.tmp_val = [self.gamma_shp_tmp, self.gamma_rte_tmp, self.lambda_shp_tmp, self.lambda_rte_tmp, self.theta_tmp, self.beta_tmp, self.k_rte_tmp, self.t_rte_tmp]
        self.assign_op = [tf.assign(self.Gamma_shp, self.gamma_shp_tmp), tf.assign(self.Gamma_rte, self.gamma_rte_tmp), tf.assign(self.Lambda_shp, self.lambda_shp_tmp), tf.assign(self.Lambda_rte, self.lambda_rte_tmp), tf.assign(self.Theta, self.theta_tmp), tf.assign(self.Beta, self.beta_tmp), tf.assign(self.k_rte, self.k_rte_tmp), tf.assign(self.t_rte, self.t_rte_tmp),
            tf.assign(self.gamma_shp_tmp, self.gamma_shp_tmp_init), tf.assign(self.lambda_shp_tmp, self.lambda_shp_tmp_init)]
        self.llk, self.rmse = self._build_llk_rmse()

    def _check_convergence(self, stop_thr, llk):
        return (1 - llk / self.last_llk) <= stop_thr

    # llk 和 rmse 貌似会跳越，所以最短也需要每2个step计算一次
    def _build_llk_rmse(self):
        self._log_f('Calculate log-likelihood and RMSE ...')
        theta = tf.nn.embedding_lookup(self.Theta, self.input_u)
        beta = tf.nn.embedding_lookup(self.Beta, self.input_g)
        y = tf.reduce_sum(theta * beta, axis=1)
        self.y = y
        r = self.input_r
        rmse = tf.sqrt(tf.reduce_mean(tf.square(r - y)))
        user_set, _ = tf.unique(self.input_u)
        uset_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.Theta, user_set), axis=0)
        item_set, _ = tf.unique(self.input_g)
        gset_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.Beta, item_set), axis=0)
        # lgamma=log(gamma())
        llk = tf.reduce_sum(r * tf.log(y) - tf.lgamma(r + 1)) - tf.reduce_sum(uset_emb * gset_emb)
        # llk = llk / n
        return llk, rmse

    def _check_rmse(self, rmse, stop = 1.2):
        if rmse < stop:
            return True
        else:
            return False

    def _train(self, u, g, r, max_iter, check_every = 10, verbose = True, stop_thr = 1e-3, val_df = None):
        last_rmse = 10
        last_llk = -1e20
        # fd = {self.input_u:df.user, self.input_g:df.item, self.input_r:df.ratings}
        fd = {self.input_u: u, self.input_g: g, self.input_r: r}
        for i in range(max_iter):
            self._log_f("Iteration {}".format(i))
            self.sess.run(self.batch_op, feed_dict=fd)
            # self.sess.run(self.tmp_val, feed_dict=fd)
            self.sess.run(self.assign_op, feed_dict=fd)
            if i % check_every == 0:
                llk, rmse = self.sess.run([self.llk, self.rmse], feed_dict=fd)
                if verbose:
                    if val_df is None:
                        self._log_f("Iteration {}, train-log-likelihood : {}, train-RMSE : {}".format(i, llk, rmse))
                    else:
                        val_llk, val_rmse = self.evaluate(val_df)
                        self._log_f("Interation {}, train-log-likelihood : {}, train-RMSE : {}, val-log-likelihood : {}, val-RMSE : {}".format(i, llk, rmse, val_llk, val_rmse))
                # has_converged = self._check_convergence(stop_thr, llk)
                # self.last_llk = llk
                if rmse <= last_rmse:
                    last_rmse = rmse
                else:
                    break
                if llk >= last_llk:
                    last_llk = llk
                else:
                    break
                    # has_converged = self._check_rmse(rmse)
                    # if has_converged:
                    #     break
        self._log_f("Optimization finished.")
        return last_rmse, last_llk

    def train(self, df, max_iter = 100, check_every = 10, verbose = True, stop_thr = 1e-3, val_df = None):
        df.columns = ['user', 'item', 'ratings']
        df = self._remove_zeros(df)
        if val_df is not None:
            val_df.columns = ['user', 'item', 'ratings']
            val_df = self._remove_zeros(val_df)
        u = self.u_mapper.fit_transform(df.user)
        g = self.g_mapper.fit_transform(df.item)
        r = self._generate_item_scores(df)
        start_time = now()
        self._log_f("Starting training ...", std_datetime(start_time))
        rmse, llk = self._train(u, g, r, max_iter, check_every=check_every, verbose=verbose, stop_thr=stop_thr, val_df=val_df)
        elapse = now() - start_time
        self._log_f('train finish ...', elapse)
        self.save(llk)

    def train_all(self, df, max_iter = 100, llk_step = 2):
        """因数据太大无法一次载入，所以需要手动控制训练过程
        循环最后一轮才更新全部变量，前面只更新psi和两个shp
        全部更新过一次变量之后才开始计算llk
        """
        df.columns = ['user', 'item', 'ratings']
        ul = self.u_mapper.fit_transform(df.user)
        gl = self.g_mapper.fit_transform(df.item)
        # rl = self._generate_item_scores(df)
        rl = np.array(df.ratings.tolist(), dtype=np.float32)
        mxln = len(df)
        step = 900 * 10000
        last_llk = -1.1e30
        finish = False
        for i in range(max_iter):
            start = 0
            self._log_f('step: ', i)
            while start < mxln:
                end = start + step
                u, g, r = ul[start:end], gl[start:end], rl[start:end]
                fd = {self.input_u: u, self.input_g: g, self.input_r: r}
                self._log_f(start, end, len(u))
                start = end
                if end < mxln:
                    self.sess.run(self.batch_op, feed_dict=fd)
                else:
                    self.sess.run(self.assign_op, feed_dict=fd)


            if i % llk_step == 0:
                start = 0
                while start < mxln:
                    end = start + step
                    self._log_f('eval', start, end)
                    u, g, r = ul[start:end], gl[start:end], rl[start:end]
                    start = end
                    if end < mxln:
                        llk, rmse = self.evaluate(u,g,r)
                        self._log_f(llk, rmse)
                        continue
                    llk, rmse = self.evaluate(u,g,r, total=True)
                    self._log_f("Iteration {}, train-log-likelihood : {}, train-RMSE : {}".format(i, llk, rmse))
                    if llk > last_llk and llk < 0:
                        last_llk = llk
                        self.save()
                    else:
                        if i > 10:
                            finish = True
            if finish:
                self._log_f('train finished')
                break

    def train_part(self, u, g, r, update_all = False):
        fd = {self.input_u: u, self.input_g: g, self.input_r: r}
        # self.sess.run(self.pre_val, feed_dict=fd)
        self.sess.run(self.batch_op, feed_dict=fd)
        if update_all:
            # self.sess.run(self.tmp_val, feed_dict=fd)
            self.sess.run(self.assign_op, feed_dict=fd)

    def predict(self, user, item):
        u = self.u_mapper.fit_transform(user)
        g = self.g_mapper.fit_transform(item)
        y = self.sess.run(self.y, feed_dict={self.input_u: u, self.input_g: g})
        return y

    def evaluate(self, u, g, r, total = False):
        fd = {self.input_u: u, self.input_g: g, self.input_r: r}
        llk, rmse = self.sess.run([self.llk, self.rmse], feed_dict=fd)
        # llk /= len(u)
        self.total_llk += llk
        self.total_rmse += rmse*len(u)
        self.total += len(u)
        if total and self.total > 0:
            llk = self.total_llk / self.total
            rmse = self.total_rmse / self.total
            self.total_llk = 0
            self.total_rmse = 0
            self.total = 0
        return llk/len(u), rmse

    def topN(self, user, item, n = 10):
        gids = item.copy()
        ul = np.zeros((len(gids)), dtype=np.int32)+user
        y = self.predict(ul, item)
        orders = np.argsort(y)[::-1][:n]
        return gids[orders]

    def save(self, tag = None):
        # tf的保存有大小限制，貌似是2G，所以改为手动保存
        folder = self.folder
        if tag is None:
            tag = 'none'
        str_dump(folder + '/index', tag)
        theta, beta = self.sess.run([self.Theta, self.Beta])
        np.save(folder + '/theta.npy', theta)
        np.save(folder + '/beta.npy', beta)
        # self.saver.save(self.sess, folder+'/{}.ckpt'.format(tag))

    def load(self, tag = None):
        folder = self.folder
        if tag is None:
            tag = str_load(folder + '/index')
        f = folder + '/{}.ckpt'.format(tag)
        theta = np.load(folder + '/theta.npy')
        beta = np.load(folder + '/beta.npy')
        op = [tf.assign(self.Theta, theta), tf.assign(self.Beta, beta)]
        self.sess.run(op)
        # self.saver.restore(sess=self.sess, save_path=f)

    def save4go(self, path):
        if os.path.exists(path):  # 目录存在则删除
            shutil.rmtree(path)
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        # Tag the model, required for Go
        builder.add_meta_graph_and_variables(sess, ["myTag"])
        builder.save()
