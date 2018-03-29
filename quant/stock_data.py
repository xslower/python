# import tushare as ts
# import tensorflow as tf
from sklearn import preprocessing
import csv, pickle
import numpy as np


def divide(a, b):
    if b == 0:
        b = 1

    return (a - b) / min(a, b)


class Label(object):
    spliter = [-50, -30, -15, 0, 15, 30, 45, 60, 90]
    num_class = len(spliter) - 1

    def __init__(self, k_line, d_line):
        self.k_line = k_line
        self.d_line = d_line
        self.reward_table = np.zeros((len(k_line), 2, 2), dtype=np.float32)
        self.up_table = np.zeros((len(k_line),), dtype=np.float32)
        self.class_table = np.zeros((len(k_line), self.num_class), dtype=np.float32)
        self.decay = 0.9

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
        for i in range(len(self.k_line)):
            up = self._clse_up(i)
            self.up_table[i] = up * 100
        decay = 0.98
        for i in range(len(self.k_line) - 1, 0, -1):
            up = self.up_table[i]
            self.up_table[i - 1] += decay * up

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

    # 给一个概率分布，而不是唯一值
    def calc_class(self):
        self.calc_up()
        for i in range(len(self.up_table)):
            v = self.up_table[i]
            dis = self.score_to_dis(v)
            # print(v, dis)
            self.class_table[i] = dis


class Stock(object):
    def __init__(self, x, y, dates, spliter='2013-01-01'):
        idx = 0
        for i in range(len(dates)):
            if dates[i] >= spliter:
                idx = i
                break
        if idx == 0:
            raise Exception('died stock')
        self.train_x = x[:idx]
        self.train_y = y[:idx]
        self.train_date = dates[:idx]
        self.test_x = x[idx:]
        self.test_y = y[idx:]
        self.test_date = dates[idx:]


O_CLOSE = 0
O_VOLU = 1
stock_file = 'data/stock/%s.csv'


# O_CLSE_UP = 1
O_HIGH_UP = 2
# O_OPEN_UP = 1
# O_LOW_UP = 4
# O_TURN = 5


# 从csv文件中读取数据
# 输出格式为：0-date, 1-close, 2-close_up, 3-high_up, 4-low_up, 5-turn_over, 6-volume
def load_file(id, no_stop = False):
    I_DATE, I_OPEN, I_CLOSE, I_HIGH, I_LOW, I_VOL = 0, 1, 2, 3, 4, 5

    id = stock_id(id)
    csv_file = open(stock_file % id, 'r')
    iter = csv.reader(csv_file)
    date_line = []
    k_line = []
    last_close = 0
    for li in iter:
        if li[I_DATE] == '':
            continue
        # 去掉停盘数据
        for i in range(1, len(li)):
            if li[i] == '':
                li[i] = 0
            else:
                li[i] = float(li[i])
        if no_stop and li[I_VOL] < 1:
            continue
        date_line.append(li[I_DATE])
        new_li = np.zeros([O_VOLU + 1], dtype=np.float32)
        today_close = li[I_CLOSE]
        new_li[O_CLOSE] = today_close
        # new_li[O_OPEN_UP] = divide(li[I_OPEN], last_close) - 1
        # new_li[O_CLSE_UP] = divide(today_close, last_close) - 1
        # new_li[O_HIGH_UP] = divide(li[I_HIGH], last_close) - 1
        # new_li[O_LOW_UP] = divide(li[I_LOW], last_close) - 1
        # new_li[O_HIGH_UP] = li[I_HIGH]
        # 换手和成交量不变
        # new_li[O_TURN] = li[I_TURN]
        new_li[O_VOLU] = li[I_VOL]
        last_close = today_close
        k_line.append(new_li)
    # print(trade_data[:10], trade_data[-10:])
    csv_file.close()
    return date_line, np.array(k_line)


def print_table(tb):
    for i in range(len(tb)):
        print(tb[i])


#
def prepare_single(stock_id, obs_len = 300):
    ss = preprocessing.StandardScaler()
    d_line, k_line = load_file(stock_id, True)
    label = Label(k_line, d_line)
    label.calc_class()
    # print_table(label.class_table)
    samples = []
    # for i in range(len(k_line)-1):
    #     print(k_line[i:i+1])
    #     ss.fit_transform(k_line[i:i+1, 1:])
    # print(type(k_line[0][1]))
    new_dl = []
    y = []
    for i in range(obs_len, len(k_line)):
        # 归一化前面所有的数据
        block = k_line[:i]
        norm = ss.fit_transform(block)
        # 只取后面需要的
        samples.append(norm[-obs_len:])
        new_dl.append(d_line[i])
        y.append(label.up_table[i])
    # print(k_line[:10], samples[0])
    s = Stock(np.array(samples), np.array(y), new_dl)
    # print_table(s.test_x))
    # print_table(s.test_y
    return s


def prepare(stock_id, base_id = '000001.XSHG'):
    base = load_file(base_id)
    stock = load_file(stock_id)

    def merge_row(a, b):
        del (a[0])
        a.extend(b[1:])

    y = []
    j = 0
    for i in range(len(base) - 1):
        if stock[j][1] == stock[j][2] == 0:
            j += 1
            continue
        # 判断日期是否相同
        if base[i][0] == stock[j][0]:
            merge_row(base[i], stock[j])
            # base[i].extend(stock[j][1:])
            # base[i].append(stock[j][0])
            j += 1
        elif base[i][0] < stock[j][0]:
            empty = [0] * (len(stock[j]))
            merge_row(base[i], empty)
            # base[i].extend(empty)
        else:
            raise Exception('base date big than stock date')
    # print(base[:10], base[-10:])
    ss = preprocessing.StandardScaler()
    x_norm = ss.fit_transform(base)
    pkl_file = 'data/parsed/%s_x.pkl' % stock_id
    xf = open(pkl_file, 'wb')
    pickle.dump(x_norm, xf)
    xf.close()


def stock_id(id):
    id = str(id)
    full_id = '0' * (6 - len(id)) + id + '.XSHE'
    return full_id


if __name__ == '__main__':
    d, k, s = prepare_single('000001')
    print(d[:10], k[:10])
# fetch_stock_data('600600')
# fetch_m_data('000001')
