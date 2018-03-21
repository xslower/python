import talib
import numpy as np
import math
import pandas
import time 
import datetime 
from datetime import *
from functools import reduce 

#init方法是您的初始化逻辑，context对象可以在任何函数之间传递
def init(context): 
    context.set_slippage(0.002)
    context.set_commission(0.00025)
    context.set_benchmark("000300.SH")
    #最多买入股票数量
    context.stock_num = 1
    context.observer = {}
    # task.weekly(option_stock, weekday=2, time_rule=market_open(minute=5))
    #下面为几个定时执行的策略代码，可放开注释替换上面的执行时间
    task.daily(daily_check, time_rule=market_close(minute=15))  #每天收盘前5分钟运行
    task.weekly(option_stock, weekday=2, time_rule=market_open(minute=5))  #每周周二开盘后5分钟运行
    # task.monthly(option_stock, tradingday=1 ,time_rule=market_open(minute=5))  #每月第1个交易日开盘后5分运行

#每天检查
def daily_check(context, data_dict):
    # print(context.portfolio.positions)
    if len(context.portfolio.positions) > 0:
        for stock in context.portfolio.positions.keys():
            hold = context.portfolio.positions[stock].hold_days
            sell = False
            if hold > 10:
                if isSellPoint(stock, context) > 0:
                    sell = True
            else:
                his = get_history(hold, '1d', 'close')[stock].values
                # 先找最近最高值
                mx = 0
                idx = 0
                for i in range(0, len(his)-1):
                    if his[i] > mx:
                        idx = i
                        mx = his[i]
                if mx / his[-1] - 1 > 0.03:
                    sell = True
            if sell:
                order_target_value(stock,0)
                print('卖出', stock)
    buy_stock(context)
                
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

def calcRate(mx, mn):
    return (mx/mn - 1) * 100

# 缠绕则观察
def observer(stock, context):
    date, close, volume = getSample(stock, 1000, context.now)
    if len(close) < 500: # 次新股短时间不会有行情
        return 0
    # 判断均线缠绕
    ma = {}
    ma[5] = MA(close, 5)
    ma[10] = MA(close, 10)
    ma[20] = MA(close, 20)
    ma[30] = MA(close, 30)
    ma[60] = MA(close, 60)
    curr_ma = [0]*5
    yest_ma = [0]*5
    befr_ma = [0]*5
    i = 0
    for k in ma:
        curr_ma[i] = ma[k][-1]
        yest_ma[i] = ma[k][-2]
        befr_ma[i] = ma[k][-3]
        i+=1
    scma = sorted(curr_ma)
    syma = sorted(yest_ma)
    sbma = sorted(befr_ma)
    for i in range(1, len(scma)):
        if (scma[i] / scma[i-1] - 1) * 100 > 1.5:
            return 0
        if (syma[i] / syma[i-1] - 1) * 100 > 1.5:
            return 0
        if (sbma[i] / sbma[i-1] - 1) * 100 > 1.5:
            return 0
    return 1

# 缠绕买点
def isBuyPoint(stock, context):
    date, close, volume = getSample(stock, 1000, context.now)
    if len(close) < 63:
        return False
    # nidx, mn, xidx, mx = getMaxMin(close)
    # if calcRate(close[-1], mn) > 50: #前期不能有太高的涨幅
    #     return 0
    # if calcRate(mx, close[-1]) > 40 and len(close) - xidx < 100: #最近也不能有大跌
    #     return 0
    ma = {}
    ma[5] = MA(close, 5)
    ma[10] = MA(close, 10)
    ma[20] = MA(close, 20)
    ma[30] = MA(close, 30)
    ma[60] = MA(close, 60)
    # 判断均线方向
    if ma[20][-1] < ma[20][-2] or ma[10][-1] < ma[10][-2]:
    # if ma[10][-1] >= ma[10][-2] and ma[20][-1] >= ma[20][-2]:
        return False
    rate = calcRate(close[-1], close[-2])
    if rate < 2 or rate > 7:
        return False
    return True

def isSellPoint(stock, context):
    date, close, volume = getSample(stock, 300, context.now)
    if len(close) < 61:
        return 0
    ama = MA(close, 20)
    if close[-1] < ama[-1] and close[-2] > ama[-2]:
        return 1
    return 0

def option_stock(context,data_dict):
    # 有空位时才选股
    stock_list = chooseStock(80)
    for stock in stock_list:
        if is_st(stock):
            continue
        if observer(stock, context) > 0:
            context.observer[stock] = 2
    print(context.observer)
    # print(len(avail_list))
    
def buy_stock(context):
    stock_list = []
    del_list = []
    for stock in context.observer:
        if isBuyPoint(stock, context):
            context.observer[stock] += 1
            if context.observer[stock] > 2:
                stock_list.append(stock)
        else:
            context.observer[stock] -= 1
            if context.observer[stock] == 0:
                del_list.append(stock)
    for stock in del_list:
        del(context.observer[stock])
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
        if succ != len(context.portfolio.positions): #购买成功
            del(context.observer[stock])
        i+=1

#策略卖出信号函数
def sell_stock(context,stock_list,data_dict):
    for stock in list(context.portfolio.positions.keys()):
        if not (stock in stock_list):
           order_target_value(stock,0)  #如果不在股票列表中则全部卖出
           

#选股函数
def chooseStock(need = 10):
    max_cap = 300 * 10000 * 10000
    dataframe = get_fundamentals(
        query(
        ).filter(
        #     fundamentals.ma.twine
        # ).filter(
            fundamentals.equity_valuation_indicator.market_cap_2 < max_cap
        ).order_by(
            fundamentals.equity_valuation_indicator.market_cap_2.asc()
        ).limit(
            need
        )
    )
    stock_list = dataframe.columns.values

    return stock_list


#每天开盘前，进行监查是否需要卖出  
def before_trade(context):
    pass
    
#日或分钟或实时数据更新，将会调用这个函数
def handle_data(context,data_dict):
    pass
