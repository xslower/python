import logging as log

import tensorflow as tf
import numpy as np
import stock_data

SELL = 0
BUY = 1

log.basicConfig(filename='log/learn.log', filemode="w", level=log.INFO, format='%(message)s')


def dt():
    return tf.float32


def divied(a, b):
    if b == 0:
        b = 1
    return a / b


def precise(y, y_pred, num_class):
    cnt = []
    for i in range(num_class):
        li = []
        for j in range(num_class + 1):
            li.append(0)
        cnt.append(li)
    for i in range(len(y)):
        idx = y[i]
        # print(idx, cnt, num_class)
        cnt[idx][num_class] += 1
        cnt[idx][y_pred[i]] += 1
    log.info(cnt)
    val = cnt[0][0]
    if val == 0:
        val = 1
    for i in range(num_class):
        log.info(divied(cnt[i][i], cnt[i][-1]) * 100)
