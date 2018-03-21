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
# 1.通过分合并大超大波段来区分趋势，上涨过程买点宽松，反弹买点严格
# 方案二
# 通过成交量超过60日均线，找波段
# 方案三
# 通过斜率找波段

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
    # 把自然波段进行容忍合并
    upAdj = 0.3 # 上涨回调幅度严格
    dwAdj = 0.4 # 下跌反弹幅度宽松
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
    # 最后三个波段如果总天数少于20天，则强行合并
    if len(secList) > 3:
        if secList[-1]['end'][1] - secList[-3]['start'][1] < 20:
            secList[-3]['end'] = secList[-1]['end']
            del(secList[-2:])

    return secList

# 获取超过某些条件的波段
# 去掉下降波段
def getHighSec(secList):
    doorsill = 30
    highSec = []
    for sec in secList:
        if sec['end'][2] < sec['start'][2]:
            continue
        rate = (sec['end'][2] / sec['start'][2] - 1) * 100
        if rate < doorsill:
            continue
        sec['rate'] = rate
        # 计算平均涨跌幅 大概范围在0.5～5
        avg_rate = rate / (sec['end'][1] - sec['start'][1])
        if avg_rate < 0.6: # 去掉涨幅太慢的
            continue
        sec['avg_rate'] = avg_rate
        highSec.append(sec)
        # print(sec['rate'])
    return highSec

# 强制把过短的波段合并
def forceMerge(secList):
    i = 1
    while i < len(secList)-1:
        if secList[i+1]['end'][1] - secList[i-1]['start'][1] < 30:
            secList[i-1]['end'] = secList[i+1]['end']
            del(secList[i:i+2])
        else:
            i+=1

# 获取标准原始数据
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

max_span = 150

# 生成svm的原始标签
def getOrgSvmY(highSec, close):
    svmY = [0] * len(close)
    for sec in highSec:
        start = sec['start'][1]
        end = sec['end'][1]
        rate = sec['rate']
        cls = 1
        pre_up = 0
        thres = rate * 0.3
        svmY[start+1] = cls
        for i in range(start+1, start+15):
            up = (close[i] / close[i-1] - 1) * 100
            if up < 0:
                continue
            pre_up += up
            if pre_up < thres:
                svmY[i] = cls
            else:
                break
    return svmY

# 计算svm的样本
def getSvmXY(close, volume, orgSvmY = [0]):
    dis_len = len(close) - len(orgSvmY)
    svmX = []
    svmY = []
    for i in range(dis_len, len(close)):
        #当日涨跌幅小于2的，不作为信号点
        up = close[i] / close[i-1] - 1
        if abs(up) < 0.04:
            continue
        secList = calcSection(close[:i+1])
        if len(secList) < 4: #前面数据量过小，少于3个波段
            # svmX.append([]) #加个占位符，方便后面过滤Y
            continue
        part = []
        itrSt, itrEd= -4, -1
        # 只有三个波段,或最后波段够长则取后三段
        # sec = secList[-1]
        # if len(secList) == 3 or sec['end'][1] - sec['start'][1] > 5:
        #     itrSt, itrEd = -3, 0
        # 量化三个波段的特征信息
        vma = MA(volume[:i+1], max_span)[-1]
        for k in range(itrSt, itrEd):
            sec = secList[k]
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
        svmX.append(part)
        svmY.append(orgSvmY[i-dis_len])
    return [svmX, svmY]
 
def testSvm(close, svmY):
    ln = len(close) - len(svmY)
    for i in range(0, len(svmY)):
        if svmY[i] > 0:
            secList = calcSection(close[:i+ln+1])
            del(secList[:-4])
            print('sec', i, secList)

def verify(preY, testY):
    result = {}
    s, e = 0, 1
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
    
def train(context, stock_list):
    allSvmX = []
    allSvmY = []
    for stock in stock_list:
        date, close, vol = getSample(stock, 1000, context.now)
        if len(close) < 600: # 次新股不做为样本
            continue
        secList = calcSection(close, date)
        highSec = getHighSec(secList)
        if len(highSec) < 1:
            continue
        # print(stock, highSec)
        orgSvmY = getOrgSvmY(highSec, close)
        svmX, svmY = getSvmXY(close, vol, orgSvmY[max_span:])
        allSvmX += svmX
        allSvmY += svmY

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
    svc = svm.SVC(class_weight={1:50})
    svc.fit(trainX, trainY)
    preY = svc.predict(testX)
    context.svc = svc
    verify(preY, testY)
       
def predict(context, stock):
    if context.svc == 0:
        return 0
    date, close, vol = getSample(stock, 600, context.now)
    if len(close) < 300: # 属性计算条件不足忽略
        return 0
    svmX, svmY = getSvmXY(close, vol)
    if len(svmX) < 1 or len(svmX[0]) < 1:
        return 0
    y = context.svc.predict(svmX)[0]
    # if y > 0:
    #     print(svmX)
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
    context.stock_list = []
    context.avail_list = []
    context.stockDict = {}
    task.daily(dailyCheck, time_rule=market_open(minute=5))  #每天开盘后5分钟运行
    task.weekly(optionStock, weekday=2, time_rule=market_open(minute=15))  #每周周二开盘后5分钟运行
    # task.monthly(optionStock, tradingday=1 ,time_rule=market_open(minute=5))  #每月第1个交易日开盘后5分运行

    chooseStock(context, 200)
    train(context, context.stock_list)

def sellStock(context):
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
def dailyCheck(context, data_dict):
    sellStock(context)
    buyStock(context)  #买入股票

#操作股票
def optionStock(context,data_dict):
    stock_list = context.stock_list
    avail_list = []
    for stock in stock_list:
        if stock in list(context.portfolio.positions.keys()):
            continue
        y = predict(context, stock)
        if y > 0:
            avail_list.append(stock)
    context.avail_list = avail_list

#策略买入信号函数
def buyStock(context):
    stock_list = context.avail_list
    stock_buy_num = context.stock_num
    left_num = stock_buy_num - len(context.portfolio.positions)
    if left_num <= 0:
        return
    stock_cash = context.portfolio.cash/left_num*1.02  #每支股票买入的最大仓位
    i = 0
    while i < left_num and i < len(stock_list):
        stock = stock_list[i]
        succ = len(context.portfolio.positions)
        if i == left_num-1:
            stock_cash = context.portfolio.cash
        print('买入', stock)
        order_value(stock, stock_cash)     #买入股票
        i+=1

#选股函数
def chooseStock(context, need = 10):
    dataframe = get_fundamentals(
        query(
        # ).filter(
        #     fundamentals.ma.twine
        ).order_by(
            fundamentals.equity_valuation_indicator.market_cap_2.asc()
        ).limit(
            need
        )
    )
    stock_list = []
    for i in range(0, len(dataframe.columns.values)):
        stock = dataframe.columns.values[i]
        if is_st(stock):
            continue
        date, close, vol = getSample(stock, 300, context.now)
        if len(close) < 300:
            continue
        stock_list.append(stock)
    
    context.stock_list = stock_list
    print('pool', len(stock_list), stock_list)
    return stock_list



# 忍受下跌幅度受利润影响
def isCantHold(stock, hold):
    if hold <= 1:
        return False
    his = get_history(hold, '1d', 'close')[stock].values
    # 先找最近最高值
    mx = 0
    idx = 0
    for i in range(0, len(his)):
        if his[i] > mx:
            idx = i
            mx = his[i]

    up = (mx / his[0] - 1) * 100
    r = 3.0
    if up <= 3.0:
        r = 3
    elif up <= 19.0:
        r = (up + 3) / 2
    else:
        r = 11.0

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
