import talib
import numpy as np
import math
import pandas
import time 
import datetime 
from functools import reduce 
from sklearn import svm 


# 计算偏离度
def calcSwing(priceList):
    ret = [0,0]
    if len(priceList) < 6:
        return ret
    x1 = 0
    y1 = sum(priceList[0:3]) / 3
    x2 = len(priceList)-1
    y2 = sum(priceList[x2-2:x2+1]) / 3
    b = y1 
    k = (y2 - b) / x2
    sw = 0
    for x in range(1,x2-1):
        p = priceList[x]
        y = k * x + b
        s = abs(p-y) / y
        sw += s
    return [sw, k]

# 价格走势划分波段函数
def calcSection(date, close):
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
    doorsill = 170
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

    return [secList, highSec]
