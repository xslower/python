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


def load_file(id, no_stop = False):
    file = 'data/stock/%s.csv'
    csv_file = open(file % id, 'r')
    iter = csv.reader(csv_file)
    trade_data = []
    for li in iter:
        if li[0] == '':
            continue
        # 去掉停盘数据
        if no_stop and li[1] == li[2] == 0:
            continue
        for i in range(1, len(li)):
            if li[i] == '':
                li[i] = 0
            else:
                li[i] = float(li[i])
        trade_data.append(li)
    # print(trade_data[:10], trade_data[-10:])
    csv_file.close()
    return trade_data


def parse_y(yester, today):
    up = (today[2] - today[1]) / today[1] * 100
    y = int(up / 2) + 5
    if y < 0:
        y = 0
    elif y > 9:
        y = 9
    return y


def prepare_single(stock_id):
    xd = load_file(stock_id, True)
    y = [0] * len(xd)
    for i in range(len(xd)):
        y[i] = parse_y(None, xd[i])
        # 去掉日期
        del (xd[i][0])
    ss = preprocessing.StandardScaler()
    # x_norm = xd
    x_norm = ss.fit_transform(xd)
    return x_norm, y


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


def load_train_data(id):
    pkl_file = 'data/parsed/%s_x.pkl' % stock_id
    xf = open(pkl_file, 'rb')
    x = pickle.load(xf)
    return x
    # print(x[:10], x[-10:])


if __name__ == '__main__':
    prepare('000001.XSHE')
# fetch_stock_data('600600')
# fetch_m_data('000001')
