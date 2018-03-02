import talib
import numpy as np
import math
import pandas
import time 
import datetime 
from functools import reduce 

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
    # task.weekly(option_stock, weekday=2, time_rule=market_open(minute=5))
    #下面为几个定时执行的策略代码，可放开注释替换上面的执行时间
    task.daily(daily_check, time_rule=market_close(minute=5))  #每天收盘前5分钟运行
    task.weekly(option_stock, weekday=2, time_rule=market_open(minute=5))  #每周周二开盘后5分钟运行
    # task.monthly(option_stock, tradingday=1 ,time_rule=market_open(minute=5))  #每月第1个交易日开盘后5分运行

#每天检查
def daily_check(context, data_dict):
    # print(context.portfolio.positions)
    if len(context.portfolio.positions) > 0:
        for stock in context.portfolio.positions.keys():
            his=get_history(20, '1d', 'close')[stock].values
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
                print('卖出', stock)
                order_target_value(stock,0)

#操作股票
def option_stock(context,data_dict):
    # 有空位时才选股
    if len(context.portfolio.positions) >= context.stock_num:
        return
    context.stock_list = choose_stock_finance()
    stock_list = context.stock_list

    buy_stock(context,stock_list)  #买入股票
    

#策略买入信号函数
def buy_stock(context, stock_list):
    stock_buy_num = context.stock_num
    left_num = stock_buy_num - len(context.portfolio.positions)
    if left_num <= 0:
        return
    stock_value = context.portfolio.cash/left_num  #每支股票买入的最大仓位
    i = 1
    for stock in stock_list:
        if stock in list(context.portfolio.positions.keys()):
            continue
        print('买入', stock)
        order_value(stock, stock_value)     #买入股票
        i+=1
        if i >= left_num:
            break
    
#策略卖出信号函数
def sell_stock(context,stock_list,data_dict):
    for stock in list(context.portfolio.positions.keys()):
        if not (stock in stock_list):
           order_target_value(stock,0)  #如果不在股票列表中则全部卖出
           

#选股函数
def choose_stock_finance(need = 10):
    max = 500000000
    dataframe = get_fundamentals(
        query(
         
        ).filter(
            fundamentals.ma.twine
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
