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


# 数值转为概率分布
# todo 测试
def scoreToDis(score, spliter):
    dis = np.zeros((len(spliter) - 1,), dtype=np.float32)
    if score <= spliter[0]:
        dis[0] = 0.9
        dis[1] = 0.1
        return dis
    elif score > spliter[-1]:
        dis[-1] = 0.9
        dis[-2] = 0.1
        return dis

    for j in range(len(spliter) - 1):
        lower, bigger = spliter[j:j + 2]
        if lower < score <= bigger:
            dis[j] = 0.5
            r = (score - lower) / (bigger - lower)
            bonus_l = r * 0.5
            bonus_r = (1 - r) * 0.5
            if j == 0:
                dis[j] += bonus_l
                dis[j + 1] += bonus_r
            elif j == len(spliter) - 1:
                dis[j] += bonus_r
                dis[j - 1] += bonus_l
            else:
                dis[j + 1] += bonus_r
                dis[j - 1] += bonus_l
            break
