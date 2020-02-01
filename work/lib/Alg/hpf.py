import numpy as np
import pandas as pd
from scipy.special import digamma, gamma
import logging
import json
import os
from tqdm import trange


class HPF(object):
    """
    """
    def __init__(self, k=30, a=0.3, a_prime=0.3, b_prime=1.0,
                 c=0.3, c_prime=0.3, d_prime=1.0, log_fp=None):
        self.k = k
        self.a = a
        self.a_prime = a_prime
        self.b_prime = b_prime
        self.c = c
        self.c_prime = c_prime
        self.d_prime = d_prime
        self._log_fp = log_fp
        self.llk = -(10 ** 37)
        self.last_llk = -(10 ** 37)
        self.rmse = 0
        # self._set_logger()

    def _set_logger(self):
        if self._log_fp is None:
            logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                                level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        else:
            logging.basicConfig(filename=self._log_fp, format='%(asctime)s %(levelname)s:%(message)s',
                                level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    def _info(self, msg):
        # logging.info(msg)
        self._log_fp(msg)

    def _remove_zeros(self, df):
        return df[lambda x : x.ratings > 0]

    def _generate_user_mapping(self, df):
        user_set = set(df.user.tolist())
        self._user2ix = {user:ix for ix, user in enumerate(user_set)}
        self._ix2user = {ix:user for user, ix in self._user2ix.items()}

    def _generate_item_mapping(self, df):
        item_set = set(df.item.tolist())
        self._item2ix = {item:ix for ix, item in enumerate(item_set)}
        self._ix2item = {ix:item for item, ix in self._item2ix.items()}

    def _generate_item_scores(self, df):
        self._item_scores = df[['item', 'ratings']].groupby('item')['ratings'].sum().to_dict()
        values = self._item_scores.values()
        max_val = max(values)
        min_val = min(values)
        diff_val = max_val - min_val
        self._item_scores = {item : (score-min_val)/(diff_val) for item, score in self._item_scores.items()}

    def _initialize_parameters(self, df, hpfrec_initialize=False):
        n_user = len(self._user2ix)
        n_item = len(self._item2ix)
        if hpfrec_initialize:
            self.Theta = np.random.gamma(self.a, 1/self.b_prime, size=(n_user, self.k))
            self.Beta = np.random.gamma(self.c, 1/self.d_prime, size=(n_item, self.k))
        else:
            self.ksi = np.random.gamma(self.a_prime, self.b_prime/self.a_prime, size=(n_user, 1))
            self.Theta = np.random.gamma(self.a, 1/self.ksi, size=(n_user, self.k))
            self.eta = np.random.gamma(self.c_prime, self.d_prime/self.c_prime, size=(n_item, 1))
            self.Beta = np.random.gamma(self.c, 1/self.eta, size=(n_item, self.k))
        self.k_shp = self.a_prime + self.k * self.a
        self.t_shp = self.c_prime + self.k * self.c
        self.k_rte = self.a_prime/self.b_prime + self.Theta.sum(axis=1, keepdims=True)
        self.t_rte = self.c_prime/self.d_prime + self.Beta.sum(axis=1, keepdims=True)
        self.Gamma_rte = np.random.gamma(self.a_prime, self.b_prime/self.a_prime, size=(n_user, 1))\
                         + self.Beta.sum(axis=0, keepdims=True)
        self.Gamma_shp = self.Gamma_rte * self.Theta * np.random.uniform(low=0.85, high=1.15, size=(n_user, self.k))
        self.Lambda_rte = np.random.gamma(self.c_prime, self.d_prime/self.c_prime, size=(n_item, 1))\
                         + self.Theta.sum(axis=0, keepdims=True)
        self.Lambda_shp = self.Lambda_rte * self.Beta * np.random.uniform(low=0.85, high=1.15, size=(n_item, self.k))
        self.phi = np.zeros((df.shape[0], self.k))
        self.add_k_rte = self.a_prime / self.b_prime
        self.add_t_rte = self.c_prime / self.d_prime
        
    def _update_phi(self, df):
        self._info("Update phi ...")
        for r in range(df.shape[0]):
            row = df.iloc[r, :]
            uid = row['user']
            gid = row['item']
            rating = row['ratings']
            uix = self._user2ix[uid]
            gix = self._item2ix[gid]
            self.phi[r] = digamma(self.Gamma_shp[uix]) - np.log(self.Gamma_rte[uix]) + \
                          digamma(self.Lambda_shp[gix]) - np.log(self.Lambda_rte[gix])
            self.phi[r] = np.exp(self.phi[r])
            sumphi = np.sum(self.phi[r])
            self.phi[r] = rating * self.phi[r] / sumphi

    def _update_G_L_shp(self, df):
        self._info('Update Gamma and Lambda shape ...')
        for r in range(df.shape[0]):
            row = df.iloc[r, :]
            uid = row['user']
            gid = row['item']
            uix = self._user2ix[uid]
            gix = self._item2ix[gid]
            self.Gamma_shp[uix] += self.phi[r]
            self.Lambda_shp[gix] += self.phi[r]

    def _update_llk_rmse(self, df):
        self.last_llk = self.llk
        self.llk, self.rmse = self._cal_llk_rmse(df)

    def _check_convergence(self, stop_thr):
        return (1 - self.llk / self.last_llk) <= stop_thr

    def _cal_llk_rmse(self, df):
        llk = 0
        rmse = 0
        self._info('Calculate log-likelihood and RMSE ...')
        for r in range(df.shape[0]):
            row = df.iloc[r, :]
            uid = row['user']
            gid = row['item']
            rating = row['ratings']
            uix = self._user2ix[uid]
            gix = self._item2ix[gid]
            yhat = self.Theta[uix].dot(self.Beta[gix].T)
            llk += rating * np.log(yhat) - np.log(gamma(rating + 1))
            rmse += (rating - yhat) ** 2
        user_set = set(df.user.tolist())
        user_factors_sum = np.zeros((self.k, ))
        for user in user_set:
            user_factors_sum += self._get_user_factor(user)
        item_set = set(df.item.tolist())
        item_factors_sum = np.zeros((self.k, ))
        for item in item_set:
            item_factors_sum += self._get_item_factor(item)
        llk -= user_factors_sum.dot(item_factors_sum)
        llk = llk / df.shape[0]
        rmse = np.sqrt(rmse / df.shape[0])
        return llk, rmse

    def _train(self, df, max_iter, check_every=10, verbose=True, stop_thr=1e-3, val_df=None):
        for i in range(max_iter):
            self._info("Iteration {}".format(i))
            self._train_step(df)
            if i % check_every == 0:
                self._update_llk_rmse(df)
                if verbose:
                    if val_df is None:
                        self._info("Iteration {}, train-log-likelihood : {}, train-RMSE : {}".format(i, self.llk, self.rmse))
                    else:
                        val_llk, val_rmse = self.evaluate(val_df)
                        self._info("Interation {}, train-log-likelihood : {}, train-RMSE : {}, \
                                    val-log-likelihood : {}, val-RMSE : {}".format(i, self.llk, self.rmse, val_llk, val_rmse))
                has_converged = self._check_convergence(stop_thr)
                if has_converged:
                    break
        if i % check_every:
            self._update_llk_rmse(df)
        self._info("Optimization finished.")
        self._info("Final train-log-likelihood : {}".format(self.llk))
        self._info("Final train-RMSE : {}".format(self.rmse))
        if val_df is not None:
            val_llk, val_rmse = self.evaluate(val_df)
            self._info("Final val-log-likelihood : {}".format(val_llk))
            self._info("Final val-RMSE : {}".format(val_rmse))

    def _train_step(self, df):
        self._update_phi(df)
        self.Gamma_rte = self.k_shp / self.k_rte + self.Beta.sum(axis=0, keepdims=True)
        self.Gamma_shp[:, :] = self.a
        self.Lambda_shp[:, :] = self.c
        self._update_G_L_shp(df)
        self.Theta = self.Gamma_shp / self.Gamma_rte
        self.Lambda_rte = self.t_shp / self.t_rte + self.Theta.sum(axis=0, keepdims=True)
        self.Beta = self.Lambda_shp / self.Lambda_rte
        self.k_rte = self.add_k_rte + self.Theta.sum(axis=1, keepdims=True)
        self.t_rte = self.add_t_rte + self.Beta.sum(axis=1, keepdims=True)

    def _get_user_factor(self, user):
        uix = self._user2ix[user]
        return self.Theta[uix]

    def _get_item_factor(self, item):
        gix = self._item2ix[item]
        return self.Beta[gix]

    def _save_user_factor(self, folder):
        user_factors = {user:self.Theta[ix].tolist() for user, ix in self._user2ix.items()}
        file_path = os.path.join(folder, 'user_factors.txt')
        with open(file_path, 'w') as outfile:
            json.dump(user_factors, outfile)
        self._info("User factors saved in "+file_path)

    def _save_item_factor(self, folder):
        item_factors = {item:self.Beta[ix].tolist() for item, ix in self._item2ix.items()}
        file_path = os.path.join(folder, 'item_factors.txt')
        with open(file_path, 'w') as outfile:
            json.dump(item_factors, outfile)
        self._info("Item factors saved in "+file_path)

    def _save_item_scores(self, folder):
        file_path = os.path.join(folder, 'item_scores.txt')
        with open(file_path, 'w') as outfile:
            json.dump(self._item_scores, outfile)
        self._info("Item scores saved in "+file_path)

    def _save_to_sql(self):
        pass

    def _save_to_redis(self):
        pass

    def _load_user_factor(self, folder):
        file_path = os.path.join(folder, 'user_factors.txt')
        with open(file_path, 'r') as infile:
            user_mapping = json.load(infile)
        users = [int(u) for u in user_mapping.keys()]
        self._user2ix = {user:ix for ix, user in enumerate(users)}
        self._ix2user = {ix:user for user, ix in self._user2ix.items()}
        self.Theta = np.zeros((len(users), self.k))
        for user, ix in self._user2ix.items():
            self.Theta[ix] = user_mapping[str(user)]
        self._info('Loaded user factors from '+file_path)

    def _load_item_factor(self, folder):
        file_path = os.path.join(folder, 'item_factors.txt')
        with open(file_path, 'r') as infile:
            item_mapping = json.load(infile)
        items = [int(i) for i in item_mapping.keys()]
        self._item2ix = {item : ix for ix, item in enumerate(items)}
        self._ix2item = {ix : item for item, ix in self._item2ix.items()}
        self.Beta = np.zeros((len(items), self.k))
        for item, ix in self._item2ix.items():
            self.Beta[ix] = item_mapping[str(item)]
        self._info('Loaded item factors from '+file_path)

    def _load_item_scores(self, folder):
        file_path = os.path.join(folder, 'item_scores.txt')
        with open(file_path, 'r') as infile:
            self._item_scores = json.load(infile)
        self._info('Loaded item scores from '+file_path)

    def train(self, df, max_iter=100, check_every=10, save_folder=None, verbose=True, stop_thr=1e-3,
              val_df=None, hpfrec_initialize=False):
        df.columns = ['user', 'item', 'ratings']
        df = self._remove_zeros(df)
        if val_df is not None:
            val_df.columns = ['user', 'item', 'ratings']
            val_df = self._remove_zeros(val_df)
        self._generate_user_mapping(df)
        self._generate_item_mapping(df)
        self._generate_item_scores(df)
        self._info("Initializing parameters ...")
        self._initialize_parameters(df, hpfrec_initialize=hpfrec_initialize)
        self._info("Starting training ...")
        self._train(df, max_iter, check_every=check_every, verbose=verbose, stop_thr=stop_thr, val_df=val_df)
        if save_folder is not None:
            self.save(save_folder)

    def predict(self, user, item):
        user_factor = self._get_user_factor(user)
        item_factor = self._get_item_factor(item)
        return np.dot(user_factor, item_factor)

    def evaluate(self, df):
        return self._cal_llk_rmse(df)

    def evaluate_recall_precision(self, df, topn=10, rec_thr=0, exclude_seen=False, show_value=True, verbose=True):
        recall, precision = rec_recall_precision(self, df, topn=topn, threshold=rec_thr, exclude_seen=exclude_seen, verbose=verbose)
        if show_value:
            self._info("Recall : {:.4f}, Precision : {:.4f}".format(recall, precision))
        return recall, precision

    def topN(self, user, n=10, return_scores=False, exclude_seen=False):
        if user not in self._user2ix:
            rec_list = sorted(self._item_scores, key=lambda x:self._item_scores[x])[::-1][:n]
            if not return_scores:
                return rec_list
            else:
                return {rec:self._item_scores[rec] for rec in rec_list}

        user_factor = self._get_user_factor(user)
        scores = user_factor.dot(self.Beta.T)
        orders = np.argsort(scores)[::-1][:n]
        if not return_scores:
            return [self._ix2item[ix] for ix in orders]
        return {self._ix2item[ix] : scores[ix] for ix in orders}

    def save(self, folder):
        self._save_user_factor(folder)
        self._save_item_factor(folder)
        self._save_item_scores(folder)

    def load(self, folder):
        self._load_user_factor(folder)
        self._load_item_factor(folder)
        self._load_item_scores(folder)


def rec_recall_precision(recommender, df, topn = 30, threshold = 0, exclude_seen = True, verbose = False):
    """
        计算推荐模型的召回率(Recall)与准确率(Precision)。

        参数
        -------
        recommender : 推荐模型
        df : pandas.DataFrame, 包含三个字段， user, item, ratings
        topn : int, 推荐物品数量
        threshold : float, 评分阈值
        exclude_seen : bool, 是否排除出现过的物品

        返回
        -------
        recall : float, 召回率 (Recall)
        precision : float, 准确率 (Precision)
    """
    user_items_dict = df[lambda x: x.ratings > threshold].groupby('user').apply(lambda x: set(x.item.tolist())).to_dict()

    hit = 0
    n_recall = 0
    n_precision = 0

    if verbose:
        print("Calculating recall & precision ...")
        user_item_pairs = list(user_items_dict.items())
        n_user = len(user_item_pairs)
        for i in trange(n_user):
            user, items = user_item_pairs[i]
            rec_items = set(recommender.topN(user=user, n=topn, exclude_seen=exclude_seen))
            hit += len(items & rec_items)
            n_recall += len(items)
            n_precision += len(rec_items)
    else:
        for user, items in user_items_dict.items():
            rec_items = set(recommender.topN(user=user, n=topn, exclude_seen=exclude_seen))
            hit += len(items & rec_items)
            n_recall += len(items)
            n_precision += len(rec_items)
    recall = hit / n_recall
    precision = hit / n_precision
    return recall, precision
