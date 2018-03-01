# coding=utf-8

import datetime

import requests
import json
from rsa_client import rsa_client as rsa


# Openapi提交请求封装
class pp_client:
    # oauth2授权地址
    AUTHORIZE_URL = "https://ac.ppdai.com/oauth2/authorize"
    # 刷新Token地址
    REFRESHTOKEN_URL = "https://ac.ppdai.com/oauth2/refreshtoken"

    HEADER = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Accept-Language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4',
        'Content-Type': 'application/json;charset=utf-8'
    }

    APP_ID = ''

    # 获取授权
    # @appid 应用ID
    # @code 授权码
    @classmethod
    def authorize(cls, code):
        data = {"AppID": cls.APP_ID, "Code": code}
        r = requests.post(pp_client.AUTHORIZE_URL, data=data)
        result = r.text
        print("authorize_data:%s" % (result))
        return result

    # 刷新AccessToken
    # AppId 应用ID
    # OpenId 用户唯一标识
    # RefreshToken 刷新令牌Token
    @classmethod
    def refresh_token(cls, openid, refreshtoken):
        data = {"AppID": cls.APP_ID, "OpenId": openid, "RefreshToken": refreshtoken}
        r = requests.post(pp_client.REFRESHTOKEN_URL, data=data)
        result = r.json()
        print("refresh_token_data:%s" % (result))
        return result

    # 向拍拍贷网关发送请求
    # Url 请求地址
    # Data 请求报文
    # AppId 应用编号
    # Sign 签名信息
    # AccessToken 访问令牌
    @classmethod
    def send(cls, url, data, accesstoken=''):
        utctime = datetime.datetime.utcnow()
        sort_data = rsa.sort(data)
        sign = rsa.sign(sort_data)
        timestamp = utctime.strftime('%Y-%m-%d %H:%M:%S')
        headers = {"X-PPD-APPID": cls.APP_ID,
                   "X-PPD-SIGN": sign,
                   "X-PPD-TIMESTAMP": timestamp,
                   "X-PPD-TIMESTAMP-SIGN": rsa.sign("%s%s" % (cls.APP_ID, timestamp)),
                   "Accept": "application/json;charset=UTF-8"}
        headers = dict(headers, **pp_client.HEADER)
        if accesstoken.strip():
            headers["X-PPD-ACCESSTOKEN"] = accesstoken
        i = 0
        while True:
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                if r is not None:
                    result = r.json()
                    return result
            except Exception as e:
                i += 1
                if i > 3:
                    raise e
        # print("receive_data:\n%s" % (result))
