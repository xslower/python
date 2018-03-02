import talib
import numpy as np
import math
import pandas
import time 
import datetime 
from datetime import *
from functools import reduce 
from sklearn import svm 
from sklearn import preprocessing
from sklearn.cluster import KMeans
from collections import Counter  
# 短线策略

# 相对平滑的分波段
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

    return secList

# 获取超过某些条件的波段
# 去掉下降波段
def getHighSec(secList):
    doorsill = 4
    highSec = []
    for sec in secList:
        if sec['end'][2] < sec['start'][2]:
            continue
        rate = (sec['end'][2] / sec['start'][2] - 1) * 100
        if rate < doorsill:
            continue
        highSec.append(sec)
        # print(sec['rate'])
    return highSec

# 获取标准原始数据
def getSample(stock, num, now):
    his = get_history(num, '1d', 'close')[stock]
    date_his = his.keys()
    close_his = his.values
    volume_his = get_history(num, '1d', 'volume')[stock].values
    high_his = get_history(num, '1d', 'high')[stock].values
    low_his = get_history(num, '1d', 'low')[stock].values
    date = []
    close = []
    volume = []
    high = []
    low = []
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
        high.append(high_his[i])
        low.append(low_his[i])
    return [date, close, volume, high, low]
    
max_span = 120

def getSpanVal(close, volume):
    span = 2
    span_cle = []
    span_vol = []
    for i in range(len(close), span-1, -span):
        span_cle.append(sum(close[i-span+1:i+1]))
        span_vol.append(sum(volume[i-span+1:i+1]))
    return [span_cle, span_vol]
    
# 计算k-means聚类样本
def getSvmXY(close, volume, high, low):
    pma = IMa(np.array(close), 120, 0, 3).avgs
    # vma = IMa(np.array(volume), max_span, 0, 1).avgs
    vma = MA(volume, max_span)
    svmX = []
    svmY = []
    for i in range(max_span, len(close)-1):
        cy = getShortY(close[i-1:i+1])
        if cy == 0:
            continue
        up = close[i] / close[i-1]    #75,50
        yup = close[i-1] / close[i-2]
        h_up = high[i] / close[i-1] #48,66
        l_up = low[i] / close[i-1]  #81,35
        pr = close[i] / pma[i] #120=87,21
        vol = volume[i] / vma[i]  #64,40
        yvol = volume[i-1] / vma[i-1]   #62,46
        svmX.append([up, yup, h_up, l_up, pr, vol, yvol])
        # span_cle, span_vol = getSpanVal(close[:i+1], volume[:i+1])
        # up = span_cle[-1] / span_cle[-2]
        # vol = span_vol[-1] / span_vol[-2]
        # svmX.append([up])
        svmY.append(getShortY(close[i:i+2]))
    return [svmX, svmY]

def getShortY(close):
    if close[-2] == 0:
        return 9
    rate = round((close[-1] / close[-2] - 1) * 100)
    cls = 0
    # if rate < -6:
    #     cls = -2
    # elif rate < -2:
    #     cls = -1
    # elif rate <= 2:
    #     cls = 0
    # elif rate <= 6:
    #     cls = 1
    # else:
    #     cls = 2
    if rate < -3:
        cls = -1
    elif rate > 3:
        cls = 1
    return cls
    

# 计算信号样本, 并生成svm的原始标签
def getOrgSvmY(highSec, svmX):
    upX = []
    svmY = [0] * len(svmX)
    for sec in highSec:
        # kmsX比close短max_span
        start = sec['start'][1] - max_span
        if start < 0:
            continue
        for i in range(start-1, start+1):
                svmY[i] = 1
    return [svmY, upX]

def train(context, stockList):
    allSvmX = []
    allSvmY = []
    cache = {}
    validLIst = []
    for stock in stockList:
        date, close, vol, high, low = getSample(stock, 1000, context.now)
        if len(close) < 600: # 次新股不做为样本
            continue
        # secList = calcSection(close, date)
        # highSec = getHighSec(secList)
        # if len(highSec) < 1:
        #     continue
        # print(stock, highSec)
        svmX, svmY = getSvmXY(close, vol, high, low)
        
        # testSvm(close, svmY)
        # svmX = preprocessing.scale(svmX)
        allSvmX += svmX[:-1] # 最后一个无标签所以无效
        allSvmY += svmY[:-1]

    context.svc = 0
    if len(allSvmX) < 1:
        print('empty sample')
        return   
    end = len(allSvmX) // 3
    # print(allSvmX[:10])
    # allSvmX = preprocessing.scale(allSvmX)
    # print(allSvmX[:10])
    trainX = allSvmX[:end*2]
    testX = allSvmX[end*2:]
    trainY = allSvmY[:end*2]
    testY = allSvmY[end*2:]
    vs = Counter(allSvmY)
    print(vs)
    # rate = vs[0] // vs[1]
    svc = svm.SVC(class_weight={-1:8,1:8})
    svc.fit(trainX, trainY)
    preY = svc.predict(testX)
    context.svc = svc
    verify(preY, testY)
    
def verify(preY, testY):
    result = {}
    s, e = -1, 1
    for i in range(s, e+1):
        result[i] = {'all':0, 'rate':0}
        for j in range(s, e+1):
            result[i][j] = 0
        
    for i in range(0, len(testY)):
        y = testY[i]
        result[y][preY[i]] += 1
        result[y]['all'] += 1
    for i in range(s, e+1):
        if result[i]['all'] > 0:
            result[i]['rate'] = result[i][i] / result[i]['all'] *100

    print(result)
    
def predict(context, data_dict, stock):
    if context.svc == 0:
        return 0
    date, close, vol, high, low = getSample(stock, 600, context.now)
    if len(date) < 300: # 属性计算条件不足忽略
        return 0
    dis = -2 - max_span
    close.append(data_dict[stock].last)
    close.append(0)
    vol.append(data_dict[stock].volume*1.06)
    vol.append(0)
    high.append(data_dict[stock].high)
    high.append(0)
    low.append(data_dict[stock].low)
    low.append(0)
    svmX, errY = getSvmXY(close[dis:], vol[dis:], high[dis:], low[dis:])
    if len(svmX) < 1:
        return 0
    y = context.svc.predict(svmX)[0]
    # if y != 0:
    return y


def getMaxMin(val_list):
    mx, mn = 0, 9999
    max_idx, min_idx = 0, 0
    for i in range(0, len(val_list)):
        if val_list[i] > mx:
            mx = val_list[i]
            max_idx = i
        if val_list[i] < mn:
            mn = val_list[i]
            min_idx = i
    return [min_idx, mn, max_idx, mx]


#init方法是您的初始化逻辑，context对象可以在任何函数之间传递
def init(context): 
    context.set_slippage(0.002)
    context.set_commission(0.00025)
    context.set_benchmark("399102.SZ")
    #最多买入股票数量
    context.stock_num = 5
    context.availList = []
    context.stockDict = {}
    task.daily(daily_check, time_rule=market_close(minute=10))  #每天收盘前5分钟运行
    # task.weekly(option_stock, weekday=2, time_rule=market_open(minute=15))  #每周周二开盘后5分钟运行
    # task.monthly(option_stock, tradingday=1 ,time_rule=market_open(minute=5))  #每月第1个交易日开盘后5分运行

    stockList = chooseStock(80)
    train(context, stockList)

def sell_stock(context):
    if len(context.portfolio.positions) > 0:
        for stock in context.portfolio.positions.keys():
            # context.stockDict[stock]['hold'] += 1
            # print('hold', context.stockDict[stock], context.portfolio.positions[stock].hold_days)
            
            
            hold = context.portfolio.positions[stock].hold_days
            if isCantHold(stock, hold):
                print('卖出', stock)
                order_target_value(stock,0)
                # del(context.stockDict[stock])

#每天检查
def daily_check(context, data_dict):
    option_stock(context,data_dict)
    sell_stock(context)
    buy_stock(context)  #买入股票

#操作股票
def option_stock(context, data_dict):
    stockPack = chooseStock(80)
    stockList = []
    for stock in stockPack:
        if stock in list(context.portfolio.positions.keys()):
            continue
        if is_st(stock):
            continue
        y = predict(context, data_dict, stock)
        if y > 0:
            stockList.append(stock)
    context.availList = stockList

#策略买入信号函数
def buy_stock(context):
    stockList = context.availList
    stock_buy_num = context.stock_num
    left_num = stock_buy_num - len(context.portfolio.positions)
    if left_num <= 0:
        return
    stock_cash = context.portfolio.cash/left_num*1.02  #每支股票买入的最大仓位
    i = 0
    while i < left_num and i < len(stockList):
        stock = stockList[i]
        succ = len(context.portfolio.positions)
        if i == left_num-1:
            stock_cash = context.portfolio.cash
        print('买入', stock)
        order_value(stock, stock_cash)     #买入股票
        i+=1

#选股函数
def chooseStock(need = 10):
    dataframe = get_fundamentals(
        query(
        # ).filter(
        #     fundamentals.ma.twine
        ).order_by(
            fundamentals.equity_valuation_indicator.market_cap_2.asc()
        ).limit(need)
    )
    stockList = dataframe.columns.values
    return stockList[20:]

# 忍受下跌幅度受利润影响
def isCantHold(stock, hold):
    if hold < 1:
        return False
    his = get_history(hold+1, '1d', 'close')[stock].values
    # 先找最近最高值
    mx = 0
    idx = 0
    for i in range(0, len(his)):
        if his[i] > mx:
            idx = i
            mx = his[i]

    up = (mx / his[0] - 1) * 100
    if up > 8:
        return True
    if hold > 10:
        return True
    r = 3.0
    # if up <= 2.0:
    #     pass
    # elif up <= 8.0:
    #     r = (up + r) / 2
    # else:
    #     r = 5.0

    dw = (mx / his[-1] - 1) * 100
    if dw >= r:
        return True
    return False
   
    
#每天开盘前进行选股    
def before_trade(context):
    pass
    
#日或分钟或实时数据更新，将会调用这个函数
def handle_data(context,data_dict):
    pass
