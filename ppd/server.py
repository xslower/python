# coding=utf8

from header import *
# from svm import *
# from api_op import *
import svc
import api_op as api

# from header import config

_key = 'Result'
_msg = 'ResultMessage'
_code = 'Code'

def try_buy_bid(list_id, amount):
    succ = False
    for i in range(len(config.users)):
        u = config.users[i]
        ret = api.buy_bid(u.tk, list_id, amount)
        keys = ret.keys()
        log.info('%s', ret)
        if ret is None or _key not in keys:
            if _code in keys and ret[_code] == 'GTW-BRQ-INVALIDTOKEN':
                config.refresh_token()
            continue
        code = ret[_key]
        if code == 0:
            succ = True
            break
        elif code == 4001:  # 用户余额不足
            continue
        else:
            break
    return succ


def buy_bids(good_list, amount):
    succ = []
    fail = []
    for i in range(len(good_list)):
        lid = good_list[i]
        if try_buy_bid(lid, amount):
            succ.append(lid)
        else:
            fail.append(lid)
    # 伪随机效果，以免重复投标重复失败
    # config.users[0], config.users[1] = config.users[1], config.users[0]
    # config.users.reverse()
    return succ, fail


def update_bid_db(binfo_list, succ_list, fail_list, bad_list, field, mod):
    # 更新db信息
    if len(binfo_list) > 0:
        p_bids_avail.multi_insert(*binfo_list)
        # field = 'debtId'
        # mod = p_debt_list
        if len(bad_list) > 0:
            mod.where().in_(field, *bad_list).update(status=-2)
        if len(fail_list) > 0:
            mod.where().in_(field, *fail_list).update(status=-1)
        if len(succ_list) > 0:
            mod.where().in_(field, *succ_list).update(status=2)
        log.info('succ %s fail %s bad %s ', succ_list, fail_list, bad_list)


def _merge_bid(binfo, bids):
    for i in range(len(binfo)):
        # ids.append(binfo[i][k_list_id])
        for j in range(len(bids)):
            if binfo[i][k_list_id] == bids[j][k_list_id]:
                binfo[i].update(bids[j])
                del (bids[j])
                break
    return binfo


def bids_filter(bids):
    ids = []
    for dic in bids:
        if len(dic['CreditCode']) > 1:  # AA or AAA
            continue
        # if dic['NormalCount'] < 10: # 直接过滤掉还款次数少的人
        #     continue
        if dic['Months'] > 9:
            continue
        ids.append(dic[k_list_id])
    return ids


# 暂时只读第一页，
def fetch_loan_list(page = 1):
    log.info('fetch loan list page %d', page)
    bid_info_list = []
    bids = api.get_loan_list(page)
    if bids is None or len(bids) == 0:
        log.info('empty list')
        return
    # p_loan_list.multi_insert(*bids)
    # ids = get_not_aa_vals(bids, k_list_id)
    ids = bids_filter(bids)
    log.info('fetched ids %s', len(ids))
    ids_ln = len(ids)
    if ids_ln == 0:
        log.info('all is AA bid. return')
        return
    if ids_ln <= 10:
        binfo = api.get_bid_info(ids)
        predict_bid(binfo)
        log.info('%s', 'predict done')
        # binfo = _merge_bid(binfo, bids)
        # bid_info_list.extend(binfo)
        # log.info('%s', 'record done.')
        # break
    else:
        i = 0
        while True:
            start = i * api.page_limit
            end = min(start + api.page_limit, ids_ln)
            if end <= start:
                break
            i += 1
            log.info('%d  %d to %d', i, start, end)
            binfo = api.get_bid_info(ids[start:end])
            if binfo is None or len(binfo) == 0:
                break
            predict_bid(binfo)
    # 递归读取后面的页面
    if len(bids) > 35 and page < 10:
        fetch_loan_list(page + 1)

def predict_bid(binfo):
    id_key = k_list_id
    data_x = []
    ids = []
    for dic in binfo:
        _id = dic[id_key]
        # if dic['CreditCode'] == 'AA':
        #     # AA不投
        #     continue
        row = svc.dicToX(dic)
        data_x.append(row)
        ids.append(_id)
    if len(data_x) == 0:
        return
    # log.info('before prediect. ids: %s', ids)
    # 合并多种预测的结果，任何一个预测危险都不投
    y_pred = config.svc.predict(data_x)
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            succ = try_buy_bid(ids[i], config.bid_amount)
            if succ:
                log.info('buyed id: %s', ids[i])
                #     config.add_bid_count()
    log.info('predict: %s', y_pred)


def run():
    code = [None]
    for c in code:
        dx, dy = svc.init_data(c)
        config.svc = svc.my_svc()
        config.svc.train(dx, dy)
        config.svc.evaluate(dx[-1200:], dy[-1200:])
    # run
    wait_sec = 1

    i = 0
    while True:
        try:
            fetch_loan_list()
            # i += 1
            # if i % 500 == 0:
            #     i = 0
            #     amount_control()
            #     config.reload_token()
            # config.limit_bid()
            log.info(time_plus.std_datetime())
            time.sleep(wait_sec)
            log.info('wake up')
        except Exception as e:
            log.info('%s', e)


def main():
    config.init()
    mode = 'norm'
    log.basicConfig(filename='log/%s.log' % mode, filemode="w", level=log.INFO, format='%(message)s')
    save_pid(mode)
    run()


if __name__ == '__main__':
    main()
