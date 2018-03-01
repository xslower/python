# coding=utf-8

# from sklearn.cross_validation import train_test_split
# import matplotlib.pyplot as plt


import yaml
from header import *
import api_op as api
import utilities as util


def _divide(a, b):
    if b != 0:
        return a / b
    return 0.5


class word2int:
    edu = {'夜大电大函大普通班': 1, '专科(高职)': 2, '专科 (高职)': 2, '专科（高职）': 2, '专科': 3, '专升本': 4, '本科': 5, '硕士研究生': 6, '博士研究生': 7}
    study = {'不详': 0, '业余': 1, '开放教育': 2, '网络教育': 2, '函授': 2, '成人': 3, '自考': 4, '自学考试': 4, '脱产': 5, '普通': 6, '普通全日制': 7, '全日制': 7, '研究生': 8}

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
    row = [bid.amount, bid.owingPrincipal, bid.owingAmount, bid.highestDebt, bid.highestPrincipal, bid.totalPrincipal, _divide(bid.amount, bid.owingAmount), _divide(bid.amount, bid.highestDebt),

        bid.successCount, bid.wasteCount, bid.cancelCount, bid.failedCount, bid.normalCount, bid.overdueLessCount, bid.overdueMoreCount,

        bid.certificateValidate, bid.nciicIdentityCheck, bid.phoneValidate, bid.videoValidate, bid.creditValidate, bid.educateValidate,

        bid.months, bid.gender, word2int.edu_val(bid.educationDegree), word2int.study_val(bid.studyStyle), bid.age, bid.currentRate]

    for i in range(0, len(row)):
        if row[i] is None:
            row[i] = 0
    return row


def dicToX(info):
    row = [info['Amount'], info['OwingPrincipal'], info['OwingAmount'], info['HighestDebt'], info['HighestPrincipal'], info['TotalPrincipal'], _divide(info['Amount'], info['OwingAmount']), _divide(info['Amount'], info['HighestDebt']),

        info['SuccessCount'], info['WasteCount'], info['CancelCount'], info['FailedCount'], info['NormalCount'], info['OverdueLessCount'], info['OverdueMoreCount'],

        info['CertificateValidate'], info['NciicIdentityCheck'], info['PhoneValidate'], info['VideoValidate'], info['CreditValidate'], info['EducateValidate'],

        info['Months'], info['Gender'], word2int.edu_val(info['EducationDegree']), word2int.study_val(info['StudyStyle']), info['Age'], info['CurrentRate']]

    for i in range(0, len(row)):
        if row[i] is None:
            row[i] = 0
    return row

def class_label(status, overDays):
    repayOverRate = 98
    y = 0
    od = overDays
    if status < repayOverRate:
        # 产生了坏账式逾期则作为负样本
        if od > 30:
            y = 3
        elif od > 10:
            y = 2
        else:
            y = -1
    else:  # 基本还完, 但曾经逾期太久的也作为负样本
        if od > 30:
            y = 3
        elif od > 10:
            y = 2
        elif od > 0:
            y = 1
    return y

def regress_label(status, overDays):
    repayOverRate = 98
    y = overDays
    if status < repayOverRate:
        if y < 5:
            return -1
        y += 5
    if y > 30:
        y = 30
    return y

num_class = 4
def init_data(code = None):
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
        # 未还完、也未产生坏账式逾期的标，不能作为样本
        y = regress_label(bid.repayStatus, od)
        if y < 0:
            continue
        dy.append(y)
        row = objToX(bid)
        dx.append(row)

    log.info(len(dy))
    return dx, dy


def save_file(dx, dy, pre = 'all'):
    xf = open('data/%s_x.pkl' % pre, 'wb')
    pickle.dump(dx, xf)
    xf.close()
    yf = open('data/%s_y.pkl' % pre, 'wb')
    pickle.dump(dy, yf)
    yf.close()


def load_file(pre = 'all'):
    xf = open('data/%s_x.pkl' % pre, 'rb')
    dx = pickle.load(xf)
    yf = open('data/%s_y.pkl' % pre, 'rb')
    dy = pickle.load(yf)
    return dx, dy


def train_test_split(dx, dy, rate = 9):
    x = np.array(dx)
    y = np.array(dy)
    pos = len(x) // 10 * rate
    return x[:pos], x[pos:], y[:pos], y[pos:]


class my_svc(object):
    scaler = None
    clf_list = None

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.clf_list = []

    def train(self, x, y, weight = None):
        self.scaler.fit(x)
        x_norm = self.scaler.transform(x)

        if weight is None:
            weight = {1: 20, 2: 30, 3: 40}
        rbf = svm.SVC(class_weight=weight)
        # self.clf_list.append(rbf)
        linear = svm.SVC(kernel='linear', class_weight=weight)
        # self.clf_list.append(linear)
        poly = svm.SVC(kernel='poly', degree=3, class_weight=weight)
        # self.clf_list.append(poly)
        # clf_sigmoid = svm.SVC(kernel='sigmoid', class_weight=weight)
        r = svm.SVR()
        self.clf_list.append(r)

        for clf in self.clf_list:
            clf.fit(x_norm, y)

    # 做预测，多个模型结果合并。任一预测=1则最终=1，即对1的预测进行交集
    def predict(self, x):
        x_norm = self.scaler.transform(x)
        y_pred = None
        for clf in self.clf_list:
            y_ = clf.predict(x_norm)
            if y_pred is None:
                y_pred = y_
            else:
                for i in range(len(y_pred)):
                    y_pred[i] |= y_[i]
        return y_pred

    def evaluate(self, x_test, y_test):
        x_test = self.scaler.transform(x_test)
        for clf in self.clf_list:
            y_pred = clf.predict(x_test)
            precise_regress(y_test, y_pred)

def precise_regress(y, y_pred):
    z_max = 0.047
    z_min = 0.045
    cnt = [0,0]
    for i in y:
        if y[i] == 0:
            cnt[1] += 1
            if z_min < y_pred[i] < z_max:
                cnt[0] += 1
            else:
                log.info([y[i], y_pred[i]])
        else:
            # if z_min < y_pred[i] < z_max:
            log.info([y[i], y_pred[i]])
    log.info(cnt)
    # log.info(y)
    # log.info(y_pred)

def precise(y, y_pred, num_class = 2):
    cnt = []
    for i in range(num_class):
        cnt.append([0, 0])
    for i in range(0, len(y_pred)):
        idx = y[i]
        # print(idx, cnt, num_class)
        cnt[idx][1] += 1
        if y_pred[i] == y[i]:
            cnt[idx][0] += 1
    log.info(cnt)
    for i in range(num_class):
        log.info(cnt[i][0] / cnt[i][1] * 100)


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    dx, dy = init_data()
    x_train, x_test, y_train, y_test = train_test_split(dx, dy, 8)
    svc = my_svc()
    svc.train(x_train, y_train)
    svc.evaluate(x_test, y_test)
    # svc.evaluate(dx[-700:], dy[-700:])


if __name__ == '__main__':
    main()
