import talib
import numpy as np
import math
import pandas
import time 
import datetime 
from functools import reduce 
from sklearn import svm 


span = 80
# 计算分类属性
# 价格偏离、振动力度、每日振幅和、量比
# y 需要 原长度
def calcAttr(sample, y):
    if len(sample) != len(y):
        print('len sample is not = len y')
        return []
    attrs = []
    i = len(sample)
    while i > span:
        i-=1
        avg_pri = 0
        avg_vol = 0
        for j in range(i-span, i):
            avg_pri += sample[j][2]
            avg_vol += sample[j][3]
        avg_vol /= span
        if sample[i][3] < avg_vol: # 如果当日成交量低于平均量，则去除
            del(y[i])
            continue
        avg_pri /= span
        vol = 0
        for j in range(i-5, i):
            vol += sample[j][3]
        vol /= 5
        # 价格偏离
        dev_pri = (sample[i][2] - avg_pri) / avg_pri
        dev_vol = (vol - avg_vol) / avg_vol
        # 振幅
        sw_all = 0
        sw_day = 0
        for j in range(i-span, i):
            sw_all += abs(sample[j][2] - avg_pri) / avg_pri
            sw_day += (sample[j][5] - sample[j][4]) / sample[j][4]
        attrs.append([avg_pri, avg_vol, sw_all])
    return attrs


def calcClass(secList, length):
    y = [0] * length
    doorsill = 80
    for sec in secList:
        start = sec[1]
        end = sec[3]
        # 标示普通结点
        for j in range(start, end):
            y[j] = sec[0]
        # 标示大涨跌幅结点
        if sec[4] > sec[2]:
            rate = (sec[4] - sec[2]) / sec[2] * 100
        else:
            rate = (sec[2] - sec[4]) / sec[4] * 100
        if rate > doorsill:
            cnt = 5
            for j in range(start, start+cnt):
                y[j] = sec[0]*2
    return y

def signal(sample, y, close, volume):
    i = len(volume)
    
    while i > 0:
        i-=1
        if volume[i] / volume[i-1] < 1.5:
            del(sample[i], y[i], volume[i])

def calcSection(close):
    # 计算波段
    # 预处理，把走势分为向上或向下的波段
    secList = []
    # 划分自然波段
    start, startPr = 0, close[0]
    i = 1
    while i < len(close):
        end, endPr = i, close[i]
        direct = 1
        if startPr > endPr:
            direct = -1
        sec = [direct, start, startPr, end, endPr]
        if len(secList) == 0:
            secList.append(sec)
        else:
            lastSec = secList[-1]
            if lastSec[0] == sec[0]:
                lastSec[3] = end
                lastSec[4] = endPr
            else:
                secList.append(sec)
        start = end
        startPr = endPr
        i+=1
    # 把自然波段进行容忍合并
    upAdj = 0.3 # 上涨回调幅度严格
    dwAdj = 0.9 # 下跌反弹幅度宽松
    cnt = 0
    while cnt != len(secList):
        cnt = len(secList)
        i = 1
        while i < len(secList)-1:
            preSec = secList[i-1]
            midSec = secList[i]
            nxtSec = secList[i+1]
            # 若整体向上or向下，则合并波段
            if nxtSec[0] == preSec[0] and \
            nxtSec[0] * (nxtSec[4] - preSec[4]) > 0 and \
            nxtSec[0] * (nxtSec[2] - preSec[2]) > 0:
                preRate = abs(preSec[4] - preSec[2]) / preSec[2] * 100
                midRate = abs(midSec[4] - midSec[2]) / midSec[2] * 100
                adj = dwAdj
                if nxtSec[0] == 1:
                    adj = upAdj
                if midSec[3] - midSec[1] < 2 or\
                preRate < 8 or midRate / preRate < adj: 
                # 回调只有一天则直接合并
                # 前波段涨跌幅小于8％则直接合并
                # 不然回调幅度要小于要求值才合并
                    preSec[3] = nxtSec[3]
                    preSec[4] = nxtSec[4]
                    del(secList[i:i+2])
                else:
                    i+=1
            else:
                i+=1
    doorsill = 80
    highSec = []
    for sec in secList:
        if sec[4] > sec[2]:
            rate = (sec[4] - sec[2]) / sec[2] * 100
        else:
            rate = (sec[2] - sec[4]) / sec[4] * 100
        if rate > doorsill:
            highSec.append(sec)

    return [secList, highSec]

def getSample(stock, num):
    his = get_history(num, '1d', 'close')[stock]
    date_his = his.keys()
    close_his = his.values
    volume_his = get_history(num, '1d', 'volume')[stock].values
    low_his = get_history(num, '1d', 'low')[stock].values
    high_his = get_history(num, '1d', 'high')[stock].values
    close = []
    volume = []
    low = []
    high = []
    date = []
    for i in range(0, len(date_his)):
        # 删除价格为0的点
        # 删除停盘日
        if close_his[i] < 0.1 or volume_his[i] < 1:
            continue
        close.append(close_his[i])
        volume.append(volume_his[i])
        low.append(low_his[i])
        high.append(high_his[i])
        date.append(date_his[i])
    error = [[], [], []]
    if len(date) < 500:
        return error
    macd =  IMacd(np.array(close), 12, 26, 9, 0, 3)
    kdj = IKdj(np.array(high), np.array(low), np.array(close), 9, 3, 3, 0, 3)
    pma = IMa(np.array(close), 10, 0, 3)
    vma = IMa(np.array(volume), 30, 0, 1)
    start = 31
    # 打包数据格式
    sample = []
    for i in range(start, len(date)):
        sub = close[i] - pma.avgs[i]
        if sub > 0:
            sub /= pma.avgs[i]
        else:
            sub /= close[i]
        vol = volume[i] - vma.avgs[i]
        if vol > 0:
            vol /= vma.avgs[i]
        else:
            vol /= volume[i]
        pack = [sub, vol, kdj.ks[i], kdj.js[i], macd.diffs[i], macd.bars[i]]
        sample.append(pack)
    return [sample, close[start:], volume[start:]]

def train(context, stockList):
    all_att = []
    all_y = []
    for stock in stockList:
        sample, close, vol = getSample(stock, 1000)
        print(sample)
        if len(sample) < 500: # 次新股不做为样本
            continue
        secList, highSec = calcSection(close)
        if len(highSec) < 1:
            continue
        # print(stock, secList)
        y = calcClass(secList, len(sample))
        if len(sample) != len(y):
            print('len attrs', len(attrs), 'len y', len(y))
            continue
        if len(y) != len(vol):
            print('vol != y')
            continue
        signal(sample, y, close, vol)
        
        print(len(y), y)
        all_att += sample
        all_y += y
    context.clf = 0
    if len(all_att) < 1:
        print('empty sample')
        return
    clf = svm.SVC()
    clf.fit(all_att, all_y)
    context.clf = clf

def predict(context, stock):
    if context.clf == 0:
        return 0
    sample, close, vol = getSample(stock, span+1)
    if len(sample) < 1: # 属性计算条件不足忽略
        return 0
    y = context.clf.predict(attrs)
    print(stock, y)
    return y[0]

#选股函数
def chooseStock(need = 10):

    dataframe = get_fundamentals(
        query(
         # fundamentals.ma.twine
        ).order_by(
            fundamentals.equity_valuation_indicator.market_cap_2.asc()
        ).limit(
            need
        )
    )
    stockList = dataframe.columns.values
            
    return stockList

#init方法是您的初始化逻辑，context对象可以在任何函数之间传递
def init(context): 
    #滑点默认值为2‰
    context.set_slippage(0.002)
    #交易费默认值为0.25‰
    context.set_commission(0.00025)
    #基准默认为沪深300
    context.set_benchmark("000300.SH")
    #最多买入股票数量
    context.stock_num = 1
    #调仓周期
    #下面为几个定时执行的策略代码，可放开注释替换上面的执行时间
    task.daily(daily_check, time_rule=market_close(minute=5))  #每天收盘前5分钟运行
    task.weekly(option_stock, weekday=2, time_rule=market_open(minute=5))  #每周周二开盘后5分钟运行
    # task.monthly(option_stock, tradingday=1 ,time_rule=market_open(minute=5))  #每月第1个交易日开盘后5分运行

    stockList = chooseStock(50)
    train(context, stockList)

#每天检查
def daily_check(context, data_dict):
    # print(context.portfolio.positions)
    if len(context.portfolio.positions) > 0:
        for stock in context.portfolio.positions.keys():
            his=get_history(5, '1d', 'close')[stock].values
            # 先找最近最高值
            max = 0
            idx = 0
            for i in range(0, len(his)-1):
                if his[i] > max:
                    idx = i
                    max = his[i]
            # 再从最高点往后找最小值
            min = max
            for i in range(i, len(his)-1):
                if his[i] < min:
                    min = his[i]
            if (max - min) / max > 0.1: #回调超过10％则卖出
                order_target_value(stock,0)
                print('卖出', stock)

#操作股票
def option_stock(context,data_dict):
    # 有空位时才选股
    if len(context.portfolio.positions) >= context.stock_num:
        return
    stockPack = chooseStock(50)
    stockList = []
    for stock in stockPack:
        y = predict(context, stock)
        if y > 1:
            stockList.append(stock)
            
    buy_stock(context,stockList)  #买入股票
    

#策略买入信号函数
def buy_stock(context, stockList):
    stock_buy_num = context.stock_num
    left_num = stock_buy_num - len(context.portfolio.positions)
    if left_num <= 0:
        return
    stock_value = context.portfolio.cash/left_num  #每支股票买入的最大仓位
    i = 0
    j = 3
    while i < left_num and j < len(stockList):
        stock = stockList[j]
        if stock in list(context.portfolio.positions.keys()):
            continue
        print('买入', stock)
        order_value(stock, stock_value)     #买入股票
        i+=1
        j+=1
   
    
#每天开盘前进行选股    
def before_trade(context):
    context.stockList = chooseStock()
    
#日或分钟或实时数据更新，将会调用这个函数
def handle_data(context,data_dict):
    pass
