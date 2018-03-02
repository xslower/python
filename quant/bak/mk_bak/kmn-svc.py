import talib
import numpy as np
import math
import pandas
import time 
import datetime 
from datetime import *
from functools import reduce 
from sklearn import svm 
from sklearn.cluster import KMeans
from collections import Counter  
# todo
# 方案二
# 通过成交量超过60日均线，找波段
# 方案三
# 通过斜率找波段

def idfClass(secList, length):
    y = [0] * length
    for sec in secList:
        start = sec['start'][1]
        end = sec['end'][1]
        # 标示普通结点
        for j in range(start, end):
            y[j] = sec['uod']
    return y

def calcSection(close, date = []):
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
        sec = {'uod': uod, 'start': [start, i-1, startPr], 'end': [end, i, endPr]}
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
        i+=1
    # 把自然波段进行容忍合并
    upAdj = 0.5 # 上涨回调幅度严格
    dwAdj = 0.8 # 下跌反弹幅度宽松
    cnt = 0
    while cnt != len(secList):
        cnt = len(secList)
        i = 1
        while i < len(secList)-1:
            preSec = secList[i-1]
            midSec = secList[i]
            nxtSec = secList[i+1]
            # 若整体向上or向下，则合并波段
            if nxtSec['uod'] == preSec['uod'] and \
            nxtSec['uod'] * (nxtSec['end'][2] - preSec['end'][2]) >= 0 and \
            nxtSec['uod'] * (nxtSec['start'][2] - preSec['start'][2]) >= 0:
                preSub = abs(preSec['end'][2] - preSec['start'][2])
                midSub = abs(midSec['end'][2] - midSec['start'][2])
                adj = dwAdj
                if nxtSec['uod'] == 1:
                    adj = upAdj
                # 回调只有一天则直接合并
                # 前波段涨跌幅小于8％则直接合并
                # 不然回调幅度要小于要求值才合并
                if midSec['end'][1] - midSec['start'][1] < 10 or\
                midSub / preSub < adj: 
                    preSec['end'] = nxtSec['end']
                    del(secList[i:i+2])
                else:
                    i+=1
            else:
                i+=1
    # 最后价格缺失，易导致波段过短
    # 最后三个波段如果总天数少于5天，则强行合并
    if len(secList) > 3:
        if secList[-1]['end'][1] - secList[-3]['start'][1] < 9:
            secList[-3]['end'] = secList[-1]['end']
            del(secList[-2:])

    return secList

# 获取超过某些条件的波段
def getHighSec(secList):
    doorsill = 160
    highSec = []
    for sec in secList:
        if sec['end'][2] > sec['start'][2]:
            rate = sec['end'][2] / sec['start'][2] * 100
        else:
            rate = sec['start'][2] / sec['end'][2] * 100
        if rate > doorsill:
            # 计算平均涨跌幅
            sec['rate'] = rate / (sec['end'][1] - sec['start'][1])
            highSec.append(sec)
    return highSec

def getSample(stock, num, now):
    his = get_history(num, '1d', 'close')[stock]
    date_his = his.keys()
    close_his = his.values
    volume_his = get_history(num, '1d', 'volume')[stock].values
    close = []
    volume = []
    date = []
    for i in range(0, len(close_his)):
        # 删除价格为0的点
        # 删除停盘日
        if close_his[i] < 0.1 or volume_his[i] < 1:
            continue
        if i+1 < len(date_his):
            t = date_his[i+1]
            t = str(t.date())
        else:
            t = now - timedelta(days=1)
            t = str(t.date())
        date.append(t)
        close.append(close_his[i])
        volume.append(volume_his[i])
    return [date, close, volume]

max_span = 30
# 计算k-means聚类样本
def getKmsX(close, volume):
    pma = IMa(np.array(close), 10, 0, 3).avgs
    # vma = IMa(np.array(volume), max_span, 0, 1).avgs
    vma = MA(volume, max_span)
    kmsX = []
    for i in range(max_span, len(close)):
        up = close[i]/close[i-1] - 1
        yup = close[i-1]/close[i-2] -1
        pr = close[i] / pma[i]
        vol = volume[i] / vma[i]
        yvol = volume[i-1]/vma[i-1]
        kmsX.append([up, yup, pr, vol, yvol])
    return kmsX

# 计算信号样本, 并生成svm的原始标签
def getSignalY(highSec, kmsX):
    sigX = []
    svmY = [0] * len(kmsX)
    for sec in highSec:
        # 暂时只使用买点信号
        if sec['uod'] < 0:
            continue
        # kmsX比close短max_span
        start = sec['start'][1] - max_span
        if start < 0:
            continue
        for i in range(start-1, start+4):
            svmY[i] = 1
            sigX.append(kmsX[i])
    return [svmY, sigX]

# 计算svm的样本
def getSvmX(close, volume, kmsY, target = 0):
    startIdx = len(close) - len(kmsY)
    svmX = []
    for cleIdx in range(startIdx, len(close)):
        kmsIdx = cleIdx - startIdx
        if kmsY[kmsIdx] != target:
            continue
        secList = calcSection(close[:cleIdx])
        if len(secList) < 3: #前面数据量过小，少于3个波段
            svmX.append([]) #加个占位符，方便后面过滤Y
            continue
        vma = MA(volume[:cleIdx], cleIdx)[-1]
        part = []
        for k in range(-3,0):
            sec = secList[-3]
            # x轴偏移，方便计算
            dis = sec['start'][1]
            y1 = sec['start'][2]
            x1 = 0
            y2 = sec['end'][2]
            x2 = sec['end'][1] - dis
            b = y1
            k = (y2 - b) / x2
            x3 = (x2 // 2)
            y3 = k * x3 + b
            mid = close[x3 + dis]
            # 中间点的偏差
            dev = mid / y3 - 1
            # 波段涨跌幅
            up = sec['end'][2] / sec['start'][2] - 1
            # 时间跨度
            scope = x2
            secVma = MA(volume[dis:dis+scope], scope)[-1]
            rate = secVma / vma
            part += [up, scope, rate, dev]
        # del(part[-1])
        svmX.append(part)
    return svmX

def getSvmY(orgSvmY, kmsY, svmX, target = 0):
    svmY = []
    for i in range(0, len(kmsY)):
        if kmsY[i] == target:
            svmY.append(orgSvmY[i])
    if len(svmX) != len(svmY):
        return svmY
    i = len(svmX)
    while i > 0:
        i -= 1
        if len(svmX[i]) == 0:
            del(svmX[i], svmY[i])
    return svmY

def train(context, stockList):
    allKmsX = []
    allSigX = []
    cache = {}
    validLIst = []
    for stock in stockList:
        date, close, vol = getSample(stock, 1000, context.now)
        if len(close) < 500: # 次新股不做为样本
            continue
        secList = calcSection(close, date)
        highSec = getHighSec(secList)
        if len(highSec) < 1:
            continue
        # print(stock, highSec)
        kmsX = getKmsX(close, vol)
        svmY, sigX = getSignalY(highSec, kmsX)
        if len(sigX) < 1:
            continue
        allKmsX += kmsX
        allSigX += sigX
        validLIst.append(stock)
        cache[stock] = [close, vol, svmY, kmsX]
    kms = KMeans(n_clusters=6, random_state=0).fit(allKmsX)
    sigY = kms.predict(allSigX)
    context.kms = kms
    target = getTarget(sigY)
    allSvmX = []
    allSvmY = []
    for stock in validLIst:
        close, vol, svmY, kmsX = cache[stock]
        kmsY = kms.predict(kmsX)
        svmX = getSvmX(close, vol, kmsY, target)
        leftY = getSvmY(svmY, kmsY, svmX, target)
        allSvmX += svmX
        allSvmY += leftY
    context.svc = 0
    if len(allSvmX) < 1:
        print('empty sample')
        return
    
    print(Counter(allSvmY))
    svc = svm.SVC(class_weight={1:80})
    svc.fit(allSvmX, allSvmY)
    context.svc = svc
    context.kms_target = target

def getTarget(sigY):
    stat = Counter(sigY)
    print(stat)
    target = 0
    mx = 0
    for t, n in stat.items():
        if n > mx:
            mx = n
            target = t
    print(target)
    return target
        
def predict(context, stock):
    if context.svc == 0:
        return 0
    date, close, vol = getSample(stock, 500, context.now)
    if len(date) < 200: # 属性计算条件不足忽略
        return 0
    dis = -max_span-1
    kmsX = getKmsX(close[-31:], vol[-31:])
    kmsY = context.kms.predict(kmsX)
    svmX = getSvmX(close, vol, kmsY, context.kms_target)
    if len(svmX) < 1 or len(svmX[0]) < 1:
        return 0
    y = context.svc.predict(svmX)[0]
    if y > 0:
        print(stock, y)
    return y

#init方法是您的初始化逻辑，context对象可以在任何函数之间传递
def init(context): 
    #滑点默认值为2‰
    context.set_slippage(0.002)
    #交易费默认值为0.25‰
    context.set_commission(0.00025)
    #基准默认为沪深300
    context.set_benchmark("000300.SH")
    #最多买入股票数量
    context.stock_num = 10
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
            hold = context.portfolio.positions[stock].hold_days
            if hold > 10:
                hold = 10
            if hold == 1:
                return
            his = get_history(hold, '1d', 'close')[stock].values
            # 先找最近最高值
            mx = 0
            idx = 0
            for i in range(0, len(his)-1):
                if his[i] > mx:
                    idx = i
                    mx = his[i]
            # 再从最高点往后找最小值
            mn = mx
            for i in range(i, len(his)-1):
                if his[i] < mn:
                    mn = his[i]
            if (mx - mn) / mx > 0.1: #回调超过10％则卖出
                print('卖出', stock)
                order_target_value(stock,0)

#操作股票
def option_stock(context,data_dict):
    # 有空位时才选股
    if len(context.portfolio.positions) >= context.stock_num:
        return
    stockPack = chooseStock(30)
    stockList = []
    for stock in stockPack:
        y = predict(context, stock)
        if y > 0:
            stockList.append(stock)
            
    buy_stock(context,stockList)  #买入股票
    

#策略买入信号函数
def buy_stock(context, stockList):
    stock_buy_num = context.stock_num
    left_num = stock_buy_num - len(context.portfolio.positions)
    if left_num <= 0:
        return
    stock_value = context.portfolio.cash/left_num  #每支股票买入的最大仓位
    print(stock_value)
    i, j = 0, 0
    while i < left_num and j < len(stockList):
        stock = stockList[j]
        if stock not in list(context.portfolio.positions.keys()):
            print('买入', stock)
            order_value(stock, stock_value)     #买入股票
        i+=1
        j+=1

#选股函数
def chooseStock(need = 10):
    dataframe = get_fundamentals(
        query(
        ).filter(
            fundamentals.ma.twine
        # ).order_by(
        #     fundamentals.equity_valuation_indicator.market_cap_2.asc()
        ).limit(
            need
        )
    )
    stockList = dataframe.columns.values
            
    return stockList
 
# def plot(context, X, y):
#     import matplotlib.pyplot as plt
#     wclf = context.svc
#     xx = np.linspace(-5, 5)
#     ww = wclf.coef_[0]
#     wa = -ww[0] / ww[1]
#     wyy = wa * xx - wclf.intercept_[0] / ww[1]
    
#     # plot separating hyperplanes and samples
#     h1 = plt.plot(xx, wyy, 'k--', label='with weights')
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#     plt.legend()
    
#     plt.axis('tight')
#     plt.show()
    
#每天开盘前进行选股    
def before_trade(context):
    pass
    
#日或分钟或实时数据更新，将会调用这个函数
def handle_data(context,data_dict):
    pass
