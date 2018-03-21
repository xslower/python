import tushare as ts
# import tensorflow as tf
from sklearn import preprocessing
import csv, pickle


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


# 从csv文件中读取数据
# 输出格式为：0-close, 1-close_up, 2-high_up, 3-low_up, 4-turn_over, 5-volume
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
    trade_data = []
    last_close = 0
    for li in iter:
        if li[I_DATE] == '':
            continue
        # 去掉停盘数据
        if no_stop and li[I_TURN] == 0:
            continue
        for i in range(1, len(li)):
            if li[i] == '':
                li[i] = 0
            else:
                li[i] = float(li[i])

        today_close = li[I_CLOSE]
        li[I_OPEN] = today_close
        li[I_CLOSE] = divide(today_close, last_close) - 1
        li[I_HIGH] = divide(li[I_HIGH], last_close) - 1
        li[I_LOW] = divide(li[I_LOW], last_close) - 1

        last_close = today_close
        trade_data.append(li[1:7])
    # print(trade_data[:10], trade_data[-10:])
    csv_file.close()
    return trade_data

obs_len = 300
#
def prepare_single(stock_id):
    ss = preprocessing.StandardScaler()
    line = load_file(stock_id)
    samples = []
    for i in range(obs_len):
        samples.append(None) # 占位，保持与line长度相同

    for i in range(obs_len, len(line)):
        norm = ss.fit_transform(line[i-obs_len:i])
        samples.append(norm)

    return line, samples


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
    full_id = '0' * (6 - len(id)) + id +'.XSHE'
    return full_id


if __name__ == '__main__':
    prepare('000001.XSHE')
# fetch_stock_data('600600')
# fetch_m_data('000001')
