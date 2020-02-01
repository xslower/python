# coding: utf-8
# time module plus

import time
import datetime


minute = 60
hour = 3600
day = 3600 * 24

def somedayBefore(days):
    today=datetime.date.today()
    someday=datetime.timedelta(days=days)
    before=today-someday
    return before

# 当前时间的时间戳
def now():
    return int(time.time())

def timestamp(str):
    format = '%Y-%m-%d'
    if str.find(' ') > 0:
        format += ' %X'
    tm = time.strptime(str, format)
    return int(time.mktime(tm))

def std_date(stamp=None):
    tm = time.localtime()
    if stamp is not None:
        tm = time.localtime(stamp)
    return time.strftime('%Y-%m-%d', tm)


def std_datetime(stamp=None):
    tm = time.localtime()
    if stamp is not None:
        tm = time.localtime(stamp)
    return time.strftime('%Y-%m-%d %X', tm)
