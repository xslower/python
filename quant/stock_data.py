# import tushare as ts
# import tensorflow as tf
from sklearn import preprocessing
import csv, pickle
import numpy as np


def divide(a, b):
    if b == 0:
        if a == 0:
            a = 1
        b = a
    return (a - b) / min(a, b)


class Label(object):
    def __init__(self, k_line, d_line):
        self.spliter = [-50, -30, -15, 0, 15, 30, 45, 60, 90]
        self.num_class = len(self.spliter) - 1
        self.k_line = k_line
        self.d_line = d_line
        self.reward_table = np.zeros((len(k_line), 2, 2), dtype=np.float32)
        self.up_table = np.zeros((len(k_line)), dtype=np.float32)
        self.class_table = np.zeros((len(k_line), self.num_class), dtype=np.float32)
        self.decay = 0.9
        self.days = 15

    def _clse_up(self, idx):
        # up = self.k_line[idx][O_CLSE_UP]
        if idx > 0:
            up = divide(self.k_line[idx][O_CLOSE], self.k_line[idx - 1][O_CLOSE])
        else:
            up = 0
        return up

    def reward(self, idx, state, act):
        fee = -0.005
        up = self._clse_up(idx)
        oup = self._clse_up(idx)
        if state == 0:
            if act == 0:
                r = 0
            else:
                r = fee + (up - oup)
        else:
            if act == 0:
                r = fee + oup
            else:
                r = up
        return r * 10

    def _echo_rt(self, table):
        for i in range(len(table)):
            li = table[i]
            print(self.d_line[i], li[0], li[1], np.argmax(li[1]))

    def calc_reward(self):
        for i in range(len(self.k_line) - 1):
            for s in range(2):
                for a in range(2):
                    r = self.reward(i, s, a)
                    self.reward_table[i][s][a] = r
        # self._echo_table(self.reward_table)
        for i in range(len(self.k_line) - 1, 0, -1):
            stock = max(self.reward_table[i][1])
            cash = max(self.reward_table[i][0])
            self.reward_table[i - 1][0][1] += self.decay * stock
            self.reward_table[i - 1][1][1] += self.decay * stock
            self.reward_table[i - 1][0][0] += self.decay * cash
            self.reward_table[i - 1][1][0] += self.decay * cash
        self._echo_rt(self.reward_table)

    def _echo_ut(self, up):
        for i in range(len(up)):
            print(self.d_line[i], up[i])

    def calc_up(self):
        days = self.days
        kl = self.k_line
        for i in range(len(kl) - days):
            # 未来一段时间的
            y = 0 if kl[i][O_CLOSE] < kl[i - 1][O_CLOSE] else 1
            mn, mx = 9999, 0
            mni, mxi = 0, 0
            for j in range(i, i + days):
                val = self.k_line[j][O_CLOSE]
                if val < mn:
                    mn = val
                    mni = j
                if val > mx:
                    mx = val
                    mxi = j
            mul = 100
            up = mn if mni < mxi else mx
            up = divide(up, self.k_line[i][O_CLOSE]) * mul
            self.up_table[i] = y
            # 有衰减的未来一段时间的上涨下跌率
            # decay = 0.98
            # for i in range(1, len(self.k_line)-self.days):
            #     self.up_table[i] = np.sum(self.up_table[i:i+self.days])
            # for i in range(len(self.k_line) - 1, 0, -1):
            #     up = self.up_table[i]
            #     self.up_table[i - 1] += decay * up
            # self._echo_ut(self.up_table)

    # 数值转为概率分布
    def score_to_dis(self, score):
        spliter = self.spliter
        num = len(spliter) - 1
        dis = np.zeros((num,), dtype=np.float32)
        if score <= spliter[0]:
            dis[0] = 0.9
            dis[1] = 0.1
            return dis
        elif score > spliter[-1]:
            dis[-1] = 0.9
            dis[-2] = 0.1
            return dis
        for j in range(num):
            lower, bigger = spliter[j:j + 2]
            if lower < score <= bigger:
                dis[j] = 0.5
                r = (score - lower) / (bigger - lower)
                bonus_rt = round(r * 0.5, 3)
                bonus_lt = 0.5 - bonus_rt
                if j == 0:
                    dis[j] += bonus_lt
                    dis[j + 1] += bonus_rt
                elif j == num - 1:
                    dis[j - 1] += bonus_lt
                    dis[j] += bonus_rt
                else:
                    dis[j - 1] += bonus_lt
                    dis[j + 1] += bonus_rt
                break
        return dis


import random


def shuffle(x, y, src):
    ln = len(x)
    indexes = list(range(ln))
    random.shuffle(indexes)
    tx = [0] * ln
    ty = [0] * ln
    tsrc = [0] * ln
    for i in range(ln):
        tx[i], ty[i], tsrc[i] = x[indexes[i]], y[indexes[i]], src[indexes[i]]
    return tx, ty, tsrc


O_CLOSE = 0
stock_file = 'data/cyb/%s.csv'

# O_CLSE_UP = 1
O_HIGH_UP = 1

# O_OPEN_UP = 1
O_LOW_UP = 2
# O_TURN = 5

O_VOLU = 1


# 从csv文件中读取数据
# 输出格式为：0-date, 1-close, 2-close_up, 3-high_up, 4-low_up, 5-turn_over, 6-volume
def load_file(id, no_stop = False):
    I_DATE, I_OPEN, I_CLOSE, I_HIGH, I_LOW, I_VOL = 0, 1, 2, 3, 4, 5

    def div(a, b):
        return a

    id = stock_id(id)
    csv_file = open(stock_file % id, 'r')
    iter = csv.reader(csv_file)
    date_line = []
    k_line = []
    last_close = 0
    for li in iter:
        if li[I_DATE] == '':
            continue
        for i in range(1, len(li)):
            if li[i] == '':
                li[i] = 0
            else:
                li[i] = float(li[i])
        # 去掉停盘数据
        if no_stop and li[I_VOL] < 1:
            continue
        date_line.append(li[I_DATE])
        new_li = np.zeros([O_VOLU + 1], dtype=np.float32)
        today_close = li[I_CLOSE]
        new_li[O_CLOSE] = div(today_close, last_close)
        # new_li[O_OPEN_UP] = div(li[I_OPEN], last_close)
        # new_li[O_CLSE_UP] = div(today_close, last_close)
        # new_li[O_HIGH_UP] = div(li[I_HIGH], last_close)
        # new_li[O_LOW_UP] = div(li[I_LOW], last_close)
        # new_li[O_HIGH_UP] = li[I_HIGH]
        # 换手和成交量不变
        new_li[O_VOLU] = li[I_VOL]
        last_close = today_close
        k_line.append(new_li)
    # print(trade_data[:10], trade_data[-10:])
    csv_file.close()
    return date_line, np.array(k_line)


def print_table(tb):
    for i in range(len(tb)):
        print(tb[i])


def cac_label(x, dates, spliter = '2013-06-01'):
    obs_len = 450
    idx = 0
    for i in range(len(dates)):
        if dates[i] >= spliter:
            idx = i
            break
    if idx == 0:
        raise Exception('new stock')
    sample = x[:idx]
    if len(sample) < obs_len:
        raise Exception('no enough data')
    else:
        sample = sample[-obs_len:]
    high = max(x[idx:, O_CLOSE])
    y = (high / x[idx][O_CLOSE] - 1)
    return sample, y


def stock_id(id):
    if type(id) is str:
        return id
    id = str(id)
    full_id = '3' + '0' * (5 - len(id)) + id + '.XSHE'
    return full_id

#
def prepare_list(obs_len = 300):
    ss = preprocessing.StandardScaler()
    X = []
    Y = []
    for i in range(1, 740):
        try:
            stock = stock_id(i)
            d_line, k_line = load_file(stock, True)
            x, y = cac_label(k_line, d_line)
            x = ss.fit_transform(x)
            X.append(x)
            Y.append(y)
        except Exception as e:
            continue
    return np.array(X), np.array(Y)


def prepare(stock_id, obs_len = 200, base_id = '000001.XSHG'):
    b_d_line, b_k_line = load_file(base_id)
    d_line, k_line = load_file(stock_id)
    label = Label(k_line, d_line)
    label.calc_up()
    dl = []
    Y = []
    X = []

    ss = preprocessing.StandardScaler()
    for i in range(obs_len, len(b_d_line) - label.days):
        # 判断日期是否相同
        if b_d_line[i] != d_line[i]:
            print(b_d_line[i], d_line[i])
            continue
        y = label.up_table[i]
        # if -1 < y < 1: #涨跌不明显的去掉
        #     continue
        up = divide(k_line[i - 1][O_CLOSE], k_line[i - 2][O_CLOSE]) * 100
        if -4.0 < up < 4.0:
            continue
        vec = np.concatenate((k_line[i - obs_len:i], b_k_line[i - obs_len:i]), axis=1)
        vec = ss.fit_transform(vec)
        X.append(vec)
        Y.append(y)
        dl.append(d_line[i])
    # print(x,y,dl)
    s = Stock(np.array(X), np.array(Y), dl)
    return s
    # print(base[:10], base[-10:])
    # x_norm = ss.fit_transform(b_d_line)
    # pkl_file = 'data/parsed/%s_x.pkl' % stock_id
    # xf = open(pkl_file, 'wb')
    # pickle.dump(x, xf)
    # xf.close()



if __name__ == '__main__':
    pass
# fetch_stock_data('600600')
# fetch_m_data('000001')
