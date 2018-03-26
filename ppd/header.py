# coding=utf-8
import sys, os

sys.path.append('../lib')

from flask import Flask, request, redirect, url_for
import yaml
import pickle
import json
import urllib.parse as urlparse
import time
import time_plus
import logging as log
import numpy as np
from sklearn import svm, preprocessing
from pp_client import pp_client as pcli
from rsa_client import rsa_client as rsa
from model import *

APPID = "ae8d97c987ec487ea3638e57fabd5d4a"


class url:
    # HOST = 'http://106.15.198.173:90/'
    HOST = 'http://songwf.top/'
    AUTH = "https://ac.ppdai.com/oauth2/login?"
    # 投标接口
    buy_bid = 'https://openapi.ppdai.com/invest/BidService/Bidding'
    # 可投列表
    loan_list = 'https://openapi.ppdai.com/invest/LLoanInfoService/LoanList'
    # 借款信息
    loan_info = 'https://openapi.ppdai.com/invest/LLoanInfoService/BatchListingInfos'
    # 貌似是某些标的投资人列表
    lender_list = 'https://openapi.ppdai.com/invest/LLoanInfoService/BatchListingBidInfos'
    # 标是否已满
    bid_status = 'https://openapi.ppdai.com/invest/LLoanInfoService/BatchListingStatusInfos'

    # 购买债权
    buy_debt = 'https://openapi.ppdai.com/invest/BidService/BuyDebt'
    # 债权列表
    debt_list = 'https://openapi.ppdai.com/invest/LLoanInfoService/DebtListNew'
    # 债权转让信息
    debt_info = 'https://openapi.ppdai.com/invest/LLoanInfoService/BatchDebtInfos'
    # 用户投标记录
    u_bid_list = 'https://openapi.ppdai.com/invest/BidService/BidList'
    # 批量获取标的详细信息
    bid_info = 'https://openapi.ppdai.com/invest/LLoanInfoService/BatchListingInfos'
    # 还款情况
    repay = 'https://openapi.ppdai.com/invest/RepaymentService/FetchLenderRepayment'
    # 账户余额
    left = 'https://openapi.ppdai.com/balance/balanceService/QueryBalance'

    @staticmethod
    def auth():
        par = {'AppID': APPID, 'ReturnUrl': url.HOST + 'token'}
        return url.AUTH + urlparse.urlencode(par)


pcli.APP_ID = APPID
xslower_id = 'b92691f48df9496d973113d2ae89d5a1'  # xslower
niude_id = '3695a8a3008342e18752b0e3da66bc99'  # niude


class config(object):
    bid_amount = 54
    open_ids = [xslower_id]
    users = []
    stand_by_users = []
    tokens = []
    bid_count = 0
    bid_count_limit = 10000
    wait_time = 1800
    # scaler = preprocessing.StandardScaler()
    svc = None

    @classmethod
    def init(cls):
        for uid in cls.open_ids:
            u = p_user.read(open_id=uid)
            if u is None:
                raise Exception('open_id failed: ' + uid)
            u.tk = u.access_token
            cls.users.append(u)
            # if cls.scaler is None:
            #     cls.scaler = preprocessing.StandardScaler()

    @classmethod
    def refresh_token(cls):
        if len(cls.users) == 0:
            cls.init()
        for u in cls.users:
            ret = pcli.refresh_token(u.open_id, u.refresh_token)
            u.access_token = ret['AccessToken']
            u.refresh_token = ret['RefreshToken']
            u.expires_in = ret['ExpiresIn']
            u.tk = u.access_token
            u.save()

    @classmethod
    def reload_token(cls):
        if len(cls.users) == 0:
            cls.init()
        for u in cls.users:
            nu = p_user.read(open_id=u.open_id)
            if nu.access_token != u.access_token:
                u.tk = u.access_token = nu.access_token

    @classmethod
    def get_token(cls, open_id):
        if len(cls.users) == 0:
            cls.init()
        for u in cls.users:
            if u.open_id == open_id:
                return u.tk
        raise Exception('open_id not found')

    @classmethod
    def get_all_tokens(cls):
        if len(cls.users) == 0:
            cls.init()
        tokens = []
        for u in cls.users:
            tokens.append(u.tk)
        return tokens

    @classmethod
    def try_refresh_token(cls):
        # token相关错误则刷新则先尝试重新读db，以防两个线程互相刷新
        need_refresh = False
        for i in range(len(cls.users)):
            old_u = cls.users[i]
            new_u = p_user.read(open_id=old_u.open_id)
            if old_u.access_token == new_u.access_token:  # 没刷新过
                need_refresh = True
            else:  # 如果已经刷新过，则使用新数据
                cls.users[i] = new_u

        if need_refresh:
            cls.refresh_token()

    # @classmethod
    # def add_bid_count(cls):
    #     cls.bid_count += 1
    #
    # @classmethod
    # def limit_bid(cls):
    #     if cls.bid_count > cls.bid_count_limit:
    #         log.info('sleeping~')
    #         time.sleep(cls.wait_time)
    #         cls.bid_count = 0


def save_pid(pre):
    pid = os.getpid()
    f = open('log/pid-' + pre, 'w')
    f.write(str(pid))
    f.close()


def get_dict_vals(dic, key):
    if dic is None:
        log.info('dic is none')
        return []
    elif key not in dic.keys():
        log.info('dic: [%s], key: [%s]', dic, key)
        return []
    return dic[key]


def get_list_vals(rows, key):
    vals = []
    for row in rows:
        vals.append(row[key])
    return vals


def get_not_aa_vals(rows, key):
    vals = []
    for row in rows:
        if len(row['CreditCode']) > 1:  # AA or AAA
            continue
        vals.append(row[key])
    return vals


class decay:
    _def = 1800
    _start = 0

    @classmethod
    def span(cls):
        if cls._start == 0:
            cls._start = int(time.time())
            return cls._def
        now = int(time.time())
        span = now - cls._start
        cls._start = now
        return span + 60


# 用以临时记录处理过的id，避免重复处理
class cache:
    last_ids = []
    curr_ids = []
    counter = 0

    @classmethod
    def reset(cls):
        cls.counter += 1
        if cls.counter % 2 == 0:
            cls.last_ids = cls.curr_ids
        else:
            cls.last_ids.extend(cls.curr_ids)
        cls.curr_ids = []
