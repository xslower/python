# 米框策略
import json
import numpy as np
import talib
import math
import pandas
import time
import datetime
from functools import reduce
from sklearn import svm
from sklearn.cluster import KMeans
from collections import Counter


# 回测最高收益1370％
# todo
# 1.研究为什么不买入银之杰
# 2.优化买点卖点，不要刚买入就卖出
# 3.反弹策略
# 4.停盘策略

# class Api():

def stock_info(stock):
    return instruments(stock)


# 全部卖出
def sell_stock(stock):
    order_target_value(stock, 0)


def buy_stock(stock, value):  # 买入股票
    order_value(stock, value)


def logit(msg):
    logger.info(msg)


def history_data(stock, num, fields='close'):
    return history_bars(stock, num, '1d', fields)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息
    # 使用order_shares(id_or_ins, amount)方法进行落单
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


class Type:
    horse = 'horse'
    rebound = 'rebound'
    stopped = 'stopped'
    reopened = 'reopened'


class Field:
    date = 'datetime'
    close = 'close'
    high = 'high'
    low = 'low'
    volume = 'volume'


class Const:
    max_span = 150
    st_sold_fail = 'sold_fail'
    shanghai = "000001.XSHE"


# 选股函数
def chooseStock(need=10):
    max_cap = 300 * 10000 * 10000
    dataframe = get_fundamentals(
        query(
        ).filter(
            fundamentals.eod_derivative_indicator.market_cap_2 < max_cap
        ).order_by(
            fundamentals.eod_derivative_indicator.market_cap_2.asc()
        ).limit(
            need
        )
    )
    stock_list = dataframe.columns.values
    sample_list = []
    for i in range(0, len(stock_list)):
        stock = stock_list[i]
        if is_st_stock(stock):
            continue
        cur = datetime.datetime.now()
        d = (cur - stock_info(stock).de_listed_date).days
        if d > 0:
            continue
        sample_list.append(stock)
    return sample_list


def init(context):
    context.s1 = Const.shanghai
    scheduler.run_daily(dailyCheck, time_rule=market_open(minute=15))
    scheduler.run_weekly(weekCheck, weekday=2)  # 每周周二开盘后5分钟运行
    scheduler.run_monthly(monthCheck, tradingday=1)
    # 最多买入股票数量
    context.stock_num = 5
    context.options = {Type.stopped: Stopped(context.now), Type.reopened: Reopened(),
                       Type.rebound: Rebound(), Type.horse: Horse()}
    context.stock_dict = {}
    context.trained = 0
    context.train_counter = 2


# 每天检查
def dailyCheck(context, data_dict):
    sellCheck(context)
    buyCheck(context)


# 每周做一些辅助处理
def weekCheck(context, data_dict):
    context.options[Type.horse].strengthOrder()

    # context.strength_order = avail_list


# 每月进行一次股票搜索
def monthCheck(context, data_dict):
    if context.trained > 0:
        context.trained -= 1
        return
    else:
        # 重置计数器
        context.trained = context.train_counter

    findPotential(context)


# 计算累积振动幅度
def calcSwing(stock, span):
    high = history_data(stock, span, 'high')
    low = history_data(stock, span, 'low')
    # 震动幅度
    all_up = 0
    # up_num, dw_num = 0, 0
    for i in range(0, len(high)):
        up = calcRate(high[i], low[i])
        # if close[i] > close[i-1]:
        #     up_num+=1
        # else:
        #     dw_num+=1
        # up_list[i] = up
        all_up += abs(up)
    return all_up / span


# 通过三种策略找出未来有可能有行情的股
def findPotential(context):
    horse = Horse()
    rebound = Rebound()
    stopped = Stopped(context.now)
    reopened = Reopened()
    stock_list = chooseStock(300)
    for stock in stock_list:
        date, close = getStandard(stock, 2500)
        # 样本太短不行。老庄股也不要
        if len(close) < 500 or len(close) > 1500:
            continue
        # 当前停盘的股，加入观察列表
        if stopped.check(stock, date):
            continue
        if reopened.check(stock, date, close):
            continue
        if rebound.check(stock, date, close):
            continue
        horse.check(stock, date, close)

    context.options = {Type.stopped: stopped, Type.reopened: reopened,
                       Type.rebound: rebound, Type.horse: horse}


# 获取标准原始数据
def getStandard(stock, num=500):
    fields = [Field.close, Field.date]
    his = history_data(stock, num, fields)
    close_his = his['close']
    # volume_his = his['volume']
    date_his = his['datetime']
    close = []
    # volume = []
    _date = []
    for i in range(0, len(close_his)):
        # 删除价格为0的点
        # 删除停盘日
        # if close_his[i] < 0.1 or volume_his[i] < 1:
        if close_his[i] < 0.1:
            continue
        d = int(date_his[i] // 1000000)
        dt = datetime.date(d // 10000, d // 100 % 100, d % 100)
        _date.append(dt)
        close.append(close_his[i])
        # volume.append(volume_his[i])
    return [_date, close]


def upScore(rate, back_rate):
    doorsill = 1
    if rate < doorsill:
        return 0
    return round((rate - doorsill) ** 2 / back_rate, 3)


def getHighSec(secList, start_date=datetime.date(1, 1, 1)):
    mn, mx = 0, 0
    highSec = {}
    start = 0
    for i in range(start, len(secList)):
        sec = secList[i]
        if sec['start'][0] < start_date:
            continue
        # rate = calcRate(sec['end'][2], sec['start'][2])
        # sec['rate'] = rate
        # 计算平均涨跌幅 大概范围在0.5～5
        rate = sec['rate']
        avg_rate = round(rate / (sec['end'][1] - sec['start'][1]), 3)
        sec['avg_rate'] = avg_rate
        if sec['back_rate'] == 0:
            score = rate
        else:
            score = upScore(rate, sec['back_rate'])
        sec['score'] = score
        if rate > mx:
            mx = rate
            highSec = sec
            # if rate < mn:
            #     mn = rate
            #     highSec[0] = sec
    return highSec


# 计算增长率
def calcRate(mx, mn):
    return round((mx / mn - 1) * 100, 2)


# 相对平滑的分波段
def calcSection(close, date=[], adj=70):
    if len(close) < 1:
        return []
    if len(date) < len(close):
        date += [''] * (len(close) - len(date))
    # 计算波段
    # 预处理，把走势分为向上或向下的波段
    secList = []
    # 划分自然波段
    start, startPr = date[0], close[0]
    i = 1
    while i < len(close):
        end, endPr = date[i], close[i]
        uod = 1
        if startPr > endPr:
            uod = -1
        # sec = [uod, start, startPr, end, endPr]
        sec = {'uod': uod}
        sec['start'] = [start, i - 1, startPr]
        sec['end'] = [end, i, endPr]
        sec['back_rate'] = 0
        if len(secList) == 0:
            secList.append(sec)
        else:
            lastSec = secList[-1]
            if lastSec['uod'] == uod:
                lastSec['end'] = [end, i, endPr]
            else:
                secList.append(sec)
        start = end
        startPr = endPr
        i += 1

        # 把自然波段进行容忍合并
    # 容忍度根据中间波段的长度进行加大
    # adj_sub = k*mid_span + b
    # 100-adj = 5*k+b 
    # 0 = 21*k+b
    k = (adj - 100) / 16
    b = -21 * k
    cnt = 0
    while cnt != len(secList):
        cnt = len(secList)
        i = 1
        while i < len(secList) - 1:
            preSec = secList[i - 1]
            midSec = secList[i]
            nxtSec = secList[i + 1]
            # 若整体向上or向下，则合并波段
            if nxtSec['uod'] == preSec['uod'] and \
                                    nxtSec['uod'] * (nxtSec['end'][2] - preSec['end'][2]) >= 0 and \
                                    nxtSec['uod'] * (nxtSec['start'][2] - preSec['start'][2]) >= 0:
                preSub = abs(preSec['end'][2] - preSec['start'][2])
                midSub = abs(midSec['end'][2] - midSec['start'][2])
                back_rate = midSub / preSub * 100
                # 回调只有4天则直接合并
                # 不然回调幅度要小于要求值才合并
                pre_span = preSec['end'][1] - preSec['start'][1]
                mid_span = midSec['end'][1] - midSec['start'][1]
                cu_adj = 0
                if mid_span < 21:
                    cu_adj = k * mid_span + b
                cu_adj += adj
                merge = False
                if back_rate < cu_adj:
                    merge = True
                if merge:
                    # 回调惩罚: y=kx^2 -> 30=30^2*k -> k=1/30
                    br = back_rate * 2 / 30
                    preSec['back_rate'] += br + nxtSec['back_rate']
                    preSec['back_rate'] = round(preSec['back_rate'], 2)
                    preSec['end'] = nxtSec['end']
                    del (secList[i:i + 2])
                else:
                    i += 1
            else:
                i += 1
    # 最后价格缺失，易导致波段过短
    # 强行合并最后几天过短的波段
    if len(secList) > 3:
        if secList[-1]['end'][1] - secList[-3]['start'][1] < 10:
            secList[-3]['end'] = secList[-1]['end']
            del (secList[-2:])

    # 计算涨跌幅
    for i in range(0, len(secList)):
        sec = secList[i]
        sec['rate'] = calcRate(sec['end'][2], sec['start'][2])
        del (sec['uod'])

    return secList


def sellCheck(context):
    if len(context.portfolio.positions) > 0:
        for stock in context.portfolio.positions.keys():
            sd = context.stock_dict
            if stock in sd.keys():
                sd[stock]['hold'] += 1
            else:
                sd[stock] = {'hold': 1}
            if isCantHold(stock, sd[stock]):
                sell_stock(stock)
                print('卖出：' + stock_info(stock).symbol + ' ' + stock)
                # if cash < context.portfolio.cash: # 卖出成功
                if context.portfolio.positions[stock].quantity < 10:
                    del (context.stock_dict[stock])
                else:
                    sd[stock][Const.st_sold_fail] = sd[stock]['hold']


# 忍受下跌幅度受利润影响
def isCantHold(stock, status):
    hold = status['hold']
    if hold < 2:
        return False
    span = 10
    date, close = getStandard(stock, span + 1)
    # 上涨不卖
    if close[-1] > close[-2]:
        return False
    # 卖出失败继续卖出
    # print(status)
    if Const.st_sold_fail in status.keys():
        fail_days = hold - status[Const.st_sold_fail]
        if hold < 10:  # 失败大于10天则清除状态
            return True
    swing = calcSwing(stock, span)
    door = 8
    if swing > door:  # 平均每天振幅7
        return True

    # 最后判断跌幅
    min_idx, mn, max_idx, mx = getMaxBeforeMin(close[-hold:])
    r = 30.0

    dw = calcRate(mx, close[-1])
    if dw >= r:
        return True
    return False


# 策略买入信号函数
def buyCheck(context):
    stock_list = context.options[Type.horse].strength_order
    left_num = context.stock_num - len(context.portfolio.positions)
    if left_num <= 0:
        return
    stock_cash = context.portfolio.cash / left_num * 1.02  # 每支股票买入的最大仓位
    i = 0
    while i < len(stock_list):
        short_score, score, stock, symbol = stock_list[i]
        if stock in list(context.portfolio.positions.keys()):
            i += 1
            continue
        curr_num = len(context.portfolio.positions)
        left_num = context.stock_num - curr_num
        if 1 == left_num:
            stock_cash = context.portfolio.cash
        if 0 == left_num:
            break
        if Horse.predict(stock) == 0:
            i += 1
            continue
        buy_stock(stock, stock_cash)  # 买入股票
        if curr_num < len(context.portfolio.positions):  # 买入成功
            print('买入：' + str(stock_list[i]))
            del (stock_list[i])
        else:
            i += 1
            # i = 0
            # while i >= 0:
            #     del(stock_list[remove[i]])
            #     i+=1


def buyPoint(context):
    pass


class Horse():
    box = []
    strength_order = []

    def check(self, stock, date, close):
        score = self.longShape(close, date)
        # print(symbol, score, typ)
        if score[0] < 20:
            return False
        pack = score + [stock, stock_info(stock).symbol]
        orderInsert(self.box, pack, 0)
        return True

    @staticmethod
    def predict(stock):
        date, close = getStandard(stock, 500)
        if len(close) < 243:
            return 0
        # 上涨才是信号
        if calcRate(close[-1], close[-2]) < 0:
            return 0
        # 确定均线状态
        # 判断均线方向
        ma = {}
        ma[240] = getMaList(close, 240)
        ma[120] = getMaList(close, 120)
        if ma[120][-1] <= ma[120][-2]:
            return 0
        for i in range(-1, -20, -1):
            if ma[240][i] <= ma[240][i - 1]:
                return 0
        r240 = calcRate(ma[240][-1], ma[240][-10])
        if r240 > 5 or r240 < 0.5:
            return 0
        # 距离240线不能太远
        if calcRate(close[-1], ma[240][-1]) > 50:  # 等待再次下踩再买
            return 0
        # 一定要在挖坑确认时买入
        min_idx, mn, max_idx, mx = getMaxBeforeMin(close[-30:])
        if min_idx - max_idx < 8:  # 挖坑太小
            return 0
        if 30 - min_idx < 3:  # 止跌确认
            return 0
        return 1

    # 用240均线 配合10日振荡幅度 计算强弱
    def shortShape(self, stock):
        span = 260
        date, close = getStandard(stock, span)
        if len(close) < span:
            return -100
        ma = getMaList(close, 240)
        rate = calcRate(ma[-1], ma[-20]) * 100
        sw = calcSwing(stock, 20)
        score = rate / sw
        return [score, rate, sw]

    # 相对大盘的强势排序
    def strengthOrder(self):
        stock_list = self.box
        order = []
        for score, dev, k, stock, symbol in stock_list:
            short_score, rate, sw = self.shortShape(stock)
            if short_score < 0:
                continue
            pack = [short_score, score, stock, symbol]
            orderInsert(order, pack, 0)
        self.strength_order = order[:len(order) // 2]

    # 把1～2年走势形态量化
    # 越平稳振动幅度、总体涨跌越小越好
    @staticmethod
    def longShape(close, date):
        start = 0
        span = 600
        end = len(close)
        if end > span:
            start = end - span
        new_close = close[start:]
        smp_sec = calcSection(new_close, date[start:], 70)
        err_ret = [0, 0]

        # 计算黑马、停盘特征值
        short_sec = calcSection(new_close, date[start:], 30)

        # 画两条上下趋势线
        span = 150
        ln = len(new_close)
        x1, y1, p1, q1 = getMaxMin(new_close[:span])
        x2, y2, p2, q2 = getMaxMin(new_close[ln - span:])
        x2 += ln - span
        p2 += ln - span
        # y1 *= 100
        # y2 *= 100
        # q1 *= 100
        # q2 *= 100
        # 下线 y1 = k1*x1 + b1 -> k = (y1-y2) / (x1-x2)
        k1 = (y1 - y2) / (x1 - x2)
        b1 = y1 - k1 * x1
        # 上线
        k2 = (q1 - q2) / (p1 - p2)
        b2 = q1 - k2 * p1
        cnt = 40
        points = []
        for sec in smp_sec:
            if sec['end'][1] - sec['start'][1] > 10:
                points.append(sec['end'][1])
        if len(points) < cnt:
            cnt -= len(points)
        step = (ln - span * 2) * 2 // cnt
        for i in range(span, ln - span - step, step):
            min_idx, mn, max_idx, mx = getMaxMin(new_close[i:i + step])
            points.append(min_idx + i)
            points.append(max_idx + 1)
        dev = 0
        for i in points:
            dw_y = k1 * i + b1
            up_y = k2 * i + b2
            r = ((new_close[i] - up_y) / up_y) ** 2 + ((new_close[i] - dw_y) / dw_y) ** 2 * 3
            dev += r
        # dw_y = k1 * ln + b1
        # up_y = k2 * ln + b2
        k = abs(k1 + k2) / 2 * 100
        div = dev * (k ** 2 + 1)
        score = round(ln / div, 4)
        return [score, dev, k]


class Stopped():
    box = []
    now = 0

    def __init__(self, now):
        self.now = now

    def check(self, stock, date):
        if self.isCurrentStopped(date, self.now):
            self.box.append(stock)
            return True
        return False

    # 当前是否正停盘
    @staticmethod
    def isCurrentStopped(date, now):
        curr = datetime.date(now.year, now.month, now.day)
        span = 30
        if (curr - date[-1]).days > span:
            # print(curr, date[-1])
            return True
        return False

    # 是否重新开盘了
    @staticmethod
    def checkOpened(date, now):
        curr = datetime.date(now.year, now.month, now.day)
        span = 10  # 在列表中的必然都大于30天，突然有小于10天的，必然开盘了
        if (curr - date[-1]).days < span:
            # print(curr, date[-1])
            return True
        return False


class Reopened():
    box = []

    # 前面一段时间是否停盘过，又开盘了
    # 如果是则打包必要信息到box
    def check(self, stock, date, close):
        span = 60
        for i in range(len(date) - 1, len(date) - 20, -1):
            stop_day, open_day = date[i - 1], date[i]
            # if (date[i] - date[i - 1]).days > span:
            if (open_day - stop_day).days > span:
                # 保存的是停盘日的日期，idx属于偏移量，对于不同的时间是不同的
                score = self.score(close, date, stop_day)
                pack = [score, stop_day, open_day, stock, stock_info(stock).symbol]
                orderInsert(self.box, pack, 0)
                return True
        return False

    @staticmethod
    def predict(stock, open_day):
        date, close = getStandard(stock, 100)
        idx = Reopened.findIdxByDate(date, open_day)
        # 判断开盘后是否下跌，直接上涨的要么涨停，要么可能想出货
        min_i, mn, max_i, mx = getMaxMin(close[idx:])
        if min_i + idx >= -1:  # 最低点是今天，还未止跌
            return False
        # 只要止跌就买
        return True

    # 根据date类型时间，找到在日期列表date中的索引idx
    # idx为负数
    @staticmethod
    def findIdxByDate(date, day, start=-50, end=-1):
        idx = 0
        for i in range(start, end):
            if (date[i] - day).days >= 0 and \
                            (date[i - 1] - day).days < 0:
                idx = i
                break
        return idx

    # 以停盘期间，相对大盘的涨跌幅度计算分值
    @staticmethod
    def score(close, date, stop_day):
        sh_date, sh_close = getStandard(Const.shanghai)
        # 根据停盘日，找到对应的idx，以便找到
        idx = [0, 0]
        idx[0] = Reopened.findIdxByDate(date, stop_day)
        if idx[0] == 0:  # 没找到，出错了
            return 0
        idx[1] = idx[0] + 1
        up = calcRate(close[idx[1]], close[idx[0]])
        # 找到停盘时大盘的位置和价格
        begin_end = [stop_day, date[idx[1]]]
        sh_idx = [0, 0]
        for s in range(0, 2):
            dev = (begin_end[s] - sh_date[-1]).days
            sh_idx[s] = Reopened.findIdxByDate(sh_date, begin_end[s], dev)
        sh_up = calcRate(sh_close[sh_idx[1]], sh_close[sh_idx[0]])
        return sh_up - up


class Rebound():
    box = []

    def check(self, stock, date, close):
        if self.isRebound(close, date):
            self.box.append(stock)
            return True
        return False

    @staticmethod
    def isRebound(close, date):
        # 判断是否反弹
        start = 0
        span = 600
        end = len(close)
        if end > span:
            start = end - span
        new_close = close[start:]
        smp_sec = calcSection(new_close, date[start:], 70)
        dw_sec = {}
        dw_idx = -1
        for i in range(len(smp_sec) - 1, 0, -1):
            sec = smp_sec[i]
            if sec['rate'] < -50 and (end - sec['end'][1]) < 20:
                dw_sec = sec
                dw_idx = i
        if dw_idx > 0:
            return True
        return False

    @staticmethod
    def predict(stock):
        pass

    @staticmethod
    def score(close, date):
        # 计算反弹特征值
        # if dw_idx == 0:
        #     return []
        # elif dw_idx > 0:
        #     score = (dw_sec['rate'] ** 2 + smp_sec[dw_idx - 1]['rate']) / dw_sec['back_rate']
        #     return [[score, dw_sec]]
        pass


def orderInsert(order, pack, score_idx):
    i = 0
    while i < len(order):
        if pack[score_idx] > order[i][score_idx]:
            order.insert(i, pack)
            break
        i += 1
    if i == len(order):
        order.append(pack)


# 获取一串值的最大最小值
def getMaxMin(val_list):
    mx, mn = -1, 9999
    max_idx, min_idx = 0, 0
    for i in range(0, len(val_list)):
        if val_list[i] > mx:
            mx = val_list[i]
            max_idx = i
        if val_list[i] < mn:
            mn = val_list[i]
            min_idx = i
    return [min_idx, mn, max_idx, mx]


# 先找最大值，再从后面找最小值
def getMaxBeforeMin(val_list):
    mx = 0
    max_idx = 0
    for i in range(0, len(val_list)):
        if val_list[i] > mx:
            mx = val_list[i]
            max_idx = i
    mn = mx
    min_idx = max_idx
    for i in range(max_idx, len(val_list)):
        if val_list[i] < mn:
            mn = val_list[i]
            min_idx = i
    return [min_idx, mn, max_idx, mx]


def getMinBeforeMax(val_list):
    mn = 99999
    min_idx = 0
    for i in range(0, len(val_list)):
        if val_list[i] < mn:
            mn = val_list[i]
            min_idx = i
    mx = mn
    max_idx = min_idx
    for i in range(min_idx, len(val_list)):
        if val_list[i] > mx:
            mx = val_list[i]
            max_idx = i
    return [min_idx, mn, max_idx, mx]


def getMaList(close, day):
    ma = [0] * len(close)
    for i in range(day - 1, len(close)):
        s = sum(close[i - day + 1:i + 1])
        m = s / day
        ma[i] = m
    return ma


def print(d1, d2=None, d3=None, d4=None, d5=None):
    logit(serialize(d1))
    data = [d2, d3, d4, d5]
    for d in data:
        if d != None:
            logit(serialize(d))


def serialize(d):
    if isinstance(d, type(np.array([]))):
        return str(list(d))
    elif isinstance(d, type(datetime.date(1, 1, 1))):
        return str(d)
    elif isinstance(d, type('')):
        return d
    else:
        return str(d)
