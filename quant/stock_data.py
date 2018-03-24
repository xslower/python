import tushare as ts
# import tensorflow as tf
from sklearn import preprocessing
import csv, pickle
import numpy as np


# tf.nn.l2_normalize()
# tf.layers.batch_normalization()

# fm = ts.get_stock_basics()
# fm.to_csv('data/stock_index.csv')
# ts.get_hist_data('600848', )
# fm = ts.get_index()
# fm.to_csv('data/index.csv')

def fetch_stock_data(id):
    for i in range(1993, 2019):
        # start = '%d-12-31' % i
        end = '%d-01-01' % i
        # id = '000001'
        # df = ts.get_hist_data(id, start=start, pause=1)
        df = ts.get_hist_data(id, end=end, pause=1)
        print(i, df)
        file = 'data/stock/%s.csv' % id
        if i == 1993:
            df.to_csv(file)
        else:
            df.to_csv(file, header=None, mode='a')


def fetch_m_data(id):
    for i in range(1993, 2018):
        start = '%d-12-31' % i
        end = '%d-01-01' % i
        # id = '000001'
        df = ts.get_h_data(id, start=start, end=end, pause=1)
        print(i, df)
        file = 'data/stock/%s.csv' % id
        if i == 1993:
            df.to_csv(file)
        else:
            df.to_csv(file, header=None, mode='a')


O_CLOSE = 0
O_OPEN_UP = 1
O_CLSE_UP = 2
O_HIGH_UP = 3
O_LOW_UP = 4
O_TURN = 5
O_VOLU = 6


# 从csv文件中读取数据
# 输出格式为：0-date, 1-close, 2-close_up, 3-high_up, 4-low_up, 5-turn_over, 6-volume
def load_file(id, no_stop = False):
    I_DATE, I_OPEN, I_CLOSE, I_HIGH, I_LOW, I_TURN, I_VOL = 0, 1, 2, 3, 4, 5, 6

    def divide(a, b):
        if b == 0:
            b = 1
        return a / b

    id = stock_id(id)
    file = 'stock/%s.csv'
    csv_file = open(file % id, 'r')
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
        if no_stop and li[I_TURN] < 1:
            continue
        date_line.append(li[I_DATE])
        new_li = np.zeros([6], dtype=np.float32)
        today_close = li[I_CLOSE]
        new_li[O_CLOSE] = today_close
        new_li[O_OPEN_UP] = divide(li[I_OPEN], last_close) - 1
        new_li[O_CLSE_UP] = divide(today_close, last_close) - 1
        new_li[O_HIGH_UP] = divide(li[I_HIGH], last_close) - 1
        new_li[O_LOW_UP] = divide(li[I_LOW], last_close) - 1
        # 换手和成交量不变
        new_li[O_TURN] = li[I_TURN]
        new_li[O_VOLU] = li[I_VOL]
        last_close = today_close
        k_line.append(new_li)
    # print(trade_data[:10], trade_data[-10:])
    csv_file.close()
    return date_line, np.array(k_line)


obs_len = 300


#
def prepare_single(stock_id):
    ss = preprocessing.StandardScaler()
    d_line, k_line = load_file(stock_id, True)
    samples = []
    # for i in range(len(k_line)-1):
    #     print(k_line[i:i+1])
    #     ss.fit_transform(k_line[i:i+1, 1:])
    # print(type(k_line[0][1]))
    for i in range(obs_len, len(k_line)):
        block = k_line[i - obs_len:i]
        norm = ss.fit_transform(block)
        samples.append(norm)
    # print(k_line[:10], samples[0])
    return d_line[obs_len:], k_line[obs_len:], np.array(samples)


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
    prepare('000001.XSHE')
# fetch_stock_data('600600')
# fetch_m_data('000001')
