# coding=utf-8

# from sklearn.cross_validation import train_test_split
# import matplotlib.pyplot as plt


import yaml
from header import *
from sklearn.decomposition import PCA
import api_op as api
import utilities as util


def _divide(a, b):
    if b != 0:
        return a / b
    return 0.5


class word2int:
    edu = {'夜大电大函大普通班': 1, '专科(高职)': 2, '专科 (高职)': 2, '专科（高职）': 2, '专科': 3, '专升本': 4, '本科': 5, '硕士研究生': 6, '博士研究生': 7}
    study = {'不详': 0, '业余': 1, '开放教育': 2, '网络教育': 2, '函授': 2, '成人': 3, '自考': 4, '自学考试': 4, '脱产': 5, '普通': 6, '普通全日制': 7,
             '全日制': 7, '研究生': 8}

    @classmethod
    def edu_val(cls, key):
        if key is None or key == '':
            return 0
        if key in cls.edu.keys():
            return cls.edu[key]
        return 1

    @classmethod
    def study_val(cls, key):
        if key is None or key == '':
            return 0
        if key in cls.study.keys():
            return cls.study[key]
        return 1


def objToX(bid):
    row = [bid.amount, bid.owingPrincipal, bid.owingAmount, bid.highestDebt, bid.highestPrincipal, bid.totalPrincipal,
           _divide(bid.amount, bid.owingAmount), _divide(bid.amount, bid.highestDebt),

           bid.successCount, bid.wasteCount, bid.cancelCount, bid.failedCount, bid.normalCount, bid.overdueLessCount, bid.overdueMoreCount,
           # _divide(bid.overdueLessCount, bid.normalCount),

           bid.certificateValidate, bid.nciicIdentityCheck, bid.phoneValidate, bid.creditValidate,

           bid.months, bid.gender, word2int.edu_val(bid.educationDegree), word2int.study_val(bid.studyStyle), bid.age]

    for i in range(len(row)):
        if row[i] is None:
            row[i] = 0
    return row


def dicToX(info):
    row = [info['Amount'], info['OwingPrincipal'], info['OwingAmount'], info['HighestDebt'], info['HighestPrincipal'],
           info['TotalPrincipal'], _divide(info['Amount'], info['OwingAmount']),
           _divide(info['Amount'], info['HighestDebt']),

           info['SuccessCount'], info['WasteCount'], info['CancelCount'], info['FailedCount'], info['NormalCount'],
           info['OverdueLessCount'], info['OverdueMoreCount'],

           info['CertificateValidate'], info['NciicIdentityCheck'], info['PhoneValidate'], info['CreditValidate'],

           info['Months'], info['Gender'], word2int.edu_val(info['EducationDegree']),
           word2int.study_val(info['StudyStyle']), info['Age']]
    # info['CurrentRate']
    for i in range(0, len(row)):
        if row[i] is None:
            row[i] = 0
    return row


num_class = 3
repayOverRate = 98


def class_label(status, overDays):
    y = 0
    od = overDays
    if status < repayOverRate:
        # 产生了坏账式逾期则作为负样本
        if od > 20:
            y = 2
        else:
            # 未还完、也未产生坏账式逾期的标，不能作为样本
            y = -1
    else:  # 基本还完, 但曾经逾期太久的也作为负样本
        if od > 30:
            y = 2
        elif od > 0:
            y = 1
    return y


def init_data(code=None):
    dx = []
    dy = []
    expr = p_bids_real.where()
    if code is not None:
        expr.equal(creditCode=code)
        # else:
        # code = 'all'

    bid_iter = expr.select()
    for bid in bid_iter:
        if bid.overdueDays is None or bid.repayStatus is None or bid.creditCode is None or len(bid.creditCode) > 1:
            continue
        od = bid.overdueDays
        y = class_label(bid.repayStatus, od)
        if y < 0:
            continue
        dy.append(y)
        row = objToX(bid)
        dx.append(row)

    log.info(len(dy))
    return dx, dy


def save_file(dx, dy, pre='all'):
    xf = open('data/%s_x.pkl' % pre, 'wb')
    pickle.dump(dx, xf)
    xf.close()
    yf = open('data/%s_y.pkl' % pre, 'wb')
    pickle.dump(dy, yf)
    yf.close()


def load_file(pre='all'):
    xf = open('data/%s_x.pkl' % pre, 'rb')
    dx = pickle.load(xf)
    yf = open('data/%s_y.pkl' % pre, 'rb')
    dy = pickle.load(yf)
    return dx, dy


def train_test_split(dx, dy, rate=9):
    x = np.array(dx)
    y = np.array(dy)
    pos = len(x) // 10 * rate
    return x[:pos], x[pos:], y[:pos], y[pos:]


class my_svc(object):
    scaler = None
    pca = None
    clf_list = None

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.pca = PCA(n_components='mle', svd_solver='full')
        self.clf_list = []

    def train(self, x, y, weight=None):
        self.scaler.fit(x)
        x_norm = self.scaler.transform(x)
        self.pca.fit(x_norm)
        x_norm = self.pca.transform(x_norm)

        if weight is None:
            weight = {1: 4, 2: 50}
        rbf = svm.SVC(class_weight=weight)
        linear = svm.SVC(kernel='linear', class_weight=weight)
        poly = svm.SVC(kernel='poly', degree=3, class_weight=weight)
        # clf_sigmoid = svm.SVC(kernel='sigmoid', class_weight=weight)
        # self.clf_list.append(rbf)
        self.clf_list.append(linear)
        # self.clf_list.append(poly)


        for clf in self.clf_list:
            clf.fit(x_norm, y)

    # 做预测，多个模型结果合并。任一预测=1则最终=1，即对1的预测进行交集
    def predict(self, x):
        x_norm = self.scaler.transform(x)
        x_norm = self.pca.transform(x_norm)
        y_pred = None
        for clf in self.clf_list:
            y_ = clf.predict(x_norm)
            if y_pred is None:
                y_pred = y_
            else:
                for i in range(len(y_pred)):
                    if y_pred < y_[i]:
                        y_pred[i] = y_[i]
        return y_pred

    def evaluate(self, x_test, y_test):
        x_norm = self.scaler.transform(x_test)
        x_norm = self.pca.transform(x_norm)
        for clf in self.clf_list:
            y_pred = clf.predict(x_norm)
            precise(y_test, y_pred)


def precise(y, y_pred):
    cnt = []
    for i in range(num_class):
        li = []
        for j in range(num_class + 1):
            li.append(0)
        cnt.append(li)
    for i in range(len(y)):
        idx = y[i]
        # print(idx, cnt, num_class)
        cnt[idx][num_class] += 1
        cnt[idx][y_pred[i]] += 1
    log.info(cnt)
    val = cnt[0][0]
    if val == 0:
        val = 1
    for i in range(1, num_class):
        log.info(cnt[i][0] / val * 100)


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    dx, dy = init_data()
    x_train, x_test, y_train, y_test = train_test_split(dx, dy, 9)
    svc = my_svc()
    svc.train(x_train, y_train)
    svc.evaluate(x_test, y_test)
    # start = -800
    # svc.evaluate(dx[start:], dy[start:])


if __name__ == '__main__':
    main()
