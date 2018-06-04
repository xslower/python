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


def try_buy_debt(debt_id):
    succ = False
    for u in config.users:
        ret = api.buy_debt(u.tk, debt_id)
        log.info('buy_debt %d ret %s ', debt_id, ret)
        keys = ret.keys()
        if ret is None or _key not in keys:
            if _code in keys and ret[_code] == 'GTW-BRQ-INVALIDTOKEN':
                config.reload_token()
            continue
        code = ret[_key]
        if code == 0:
            succ = True
            break
    return succ


def buy_debts(debt_ids):
    succ = []
    fail = []
    for i in range(len(debt_ids)):
        did = debt_ids[i]
        if try_buy_debt(did):
            succ.append(did)
        else:
            fail.append(did)
    return succ, fail


# 直接购买，用以收集数据
def pre_buy(debt):
    ret = try_buy_debt(debt[k_debt_id])
    if ret:
        debt['status'] = 3
    else:
        debt['status'] = -3
    p_debt_list.insert_or_update(**debt)
    log.info(debt)


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


class credit_code:
    AAA, AA, A, B, C, D, E, F, G, H, Non = 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    map = {'H': H, 'G': G, 'F': F, 'E': E, 'D': D, 'C': C, 'B': B, 'A': A, 'AA': AA, 'AAA': AAA, None: Non}
    rate = {AAA: 8, AA: 11.5, A: 16, B: 22, C: 28, D: 38, E: 58, F: 78, G: 98, Non: 150}
    limit = {AAA: 500, AA: 2000, A: 100, B: 90, C: 80, D: 70, E: 40, F: 30, G: 20, Non: 0}
    want_rate = {'pre_buy': 3.0, 'up2': 2.0, 'up1': 1.4, 'eq': 0.8, 'dw': 1.6}

    @classmethod
    def set_want_rate(cls, pre_buy, up2, up1, eq, dw):
        cls.want_rate = {'pre_buy': pre_buy, 'up2': up2, 'up1': up1, 'eq': eq, 'dw': dw}
        log.info(cls.want_rate)


# def deal_debt(binfo, succ_list, fail_list, bad_list):
#     id_key = k_debt_id
#     data_x = []
#     ids = []
#     credit_aa = []
#     for dic in binfo:
#         _id = dic[id_key]
#         if dic['CreditCode'] == 'AA':
#             credit_aa.append(_id)
#             continue
#         row = dicToX(dic)
#         data_x.append(row)
#         ids.append(_id)
#     if len(credit_aa) > 0:
#         succ, fail = buy_debts(credit_aa)
#         succ_list.extend(succ)
#         fail_list.extend(fail)
#     if len(data_x) == 0:
#         return
#     log.info('before prediect. ids: %s', ids)
#     norm_x = config.scaler.transform(data_x)
#     pd_y = []
#     for clf in config.clfs:
#         y = clf.predict(norm_x)
#         pd_y.append(y)
#     good_list = []
#     # 合并多种预测的结果，任何一个预测危险都不投
#     for i in range(len(norm_x)):
#         is_bad = False
#         for j in range(len(config.clfs)):
#             if pd_y[j][i] == 1:
#                 is_bad = True
#                 break
#         if is_bad:
#             bad_list.append(ids[i])
#         else:
#             good_list.append(ids[i])
#     succ, fail = buy_debts(good_list)
#     succ_list.extend(succ)
#     fail_list.extend(fail)
#     return



# 债权标过滤
def filter_debt(debt):
    cc = credit_code
    cur_code = cc.map[debt['CurrentCreditCode']]
    sale_rate = debt['PriceforSaleRate']
    ought_rate = cc.rate[cur_code]
    price = debt['PriceforSale']

    # 逾期利率惩罚
    over_due = debt['PastDueNumber']
    if over_due > 0:  # 第一次加6个点，之后每多一次多3个点
        ought_rate += 6 + (over_due - 1) * 3
    # 剩余的期数太少，导致真实利率下降。
    left_num = debt['OwingNumber']
    if left_num == 1:
        ought_rate += 1.0
    elif left_num == 2:
        ought_rate += 0.5
    elif left_num >= 18:
        ought_rate += 0.5
    elif left_num >= 24:
        ought_rate += 1.0
    elif left_num >= 30:
        ought_rate += 1.5
    # 根据最近还款日短近，增加利率要求。
    days = debt['Days']
    if days <= 15:  # 还款距离越短，利率要求越高
        ought_rate += (15 - days) ** 2 * 0.007

    # 利率让利超过一定限额，且金额较小，则直接购买不走模型。目的就是买数据
    str_code = debt['CreditCode']
    if str_code == 'AA' and sale_rate > ought_rate:
        pre_buy(debt)
        return False

    # 限额根据利率优惠的幅度上下调整
    limit = cc.limit[cur_code]  # * (1 + (sale_rate - ought_rate) / 10)

    if price > limit:  # 债权份额大于对应等级的限额
        return False
    code = cc.map[str_code]
    if cur_code > code + 1:  # 升2-3级
        ought_rate += cc.want_rate['up2']
    elif cur_code > code:  # 升1级
        ought_rate += cc.want_rate['up1']
    elif cur_code == code:  # 没有升降
        ought_rate += cc.want_rate['eq']
    else:  # 降级
        ought_rate += cc.want_rate['dw']
    if sale_rate < ought_rate:  # 转让的利率 小于 要求的利率。
        return False
    return True


def fetch_debt_list():
    bid_info_list = []
    page = 1
    stamp = time.time()
    start_time = time_plus.std_datetime(stamp - decay.span())
    log.info(start_time)
    while True:
        debt_info_list = []
        debt_base = api.get_debt_list(page, start_time)
        if debt_base is None or len(debt_base) == 0:
            return
        debt_ids = get_list_vals(debt_base, 'DebtdealId')
        i = 0
        while True:
            start = i * api.page_limit
            end = min(start + api.page_limit, len(debt_ids))
            if start >= end:
                break
            i += 1
            debt_info = api.get_debt_info(debt_ids[start:end])
            if debt_info is None or len(debt_info) == 0:
                continue
            lid2did = {}
            for debt in debt_info:
                debt_id = debt[k_debt_id]
                if filter_debt(debt):
                    lid = debt[k_list_id]
                    lid2did[lid] = debt_id
                    debt['status'] = 1
                    debt_info_list.append(debt)

            if len(lid2did) == 0:
                time.sleep(0.2)
                continue
            log.info('list_ids: %s', lid2did)
            # if end < len(debt_ids):  # 没到最后一页
            #     if len(list_ids) < 10:
            #         continue
            #     else:  # 超过10条，则请求前10条
            #         ids = list_ids[:10]
            #         list_ids = list_ids[10:]
            # else:  # 最后一页无论多少都请求
            #     ids = list_ids
            bid_info = api.get_bid_info(list(lid2did.keys()))
            if bid_info is None or len(bid_info) == 0:
                continue
            for bid in bid_info:
                bid[k_debt_id] = lid2did[bid[k_list_id]]
                # bid[k_rate] = bid[k_current_rate]
                # bid[k_current_rate] =
            predict_bid(bid_info)
            bid_info_list.extend(bid_info)

        log.info('page:%d debt len %d', page, len(debt_info_list))
        if len(debt_info_list) > 0:
            # log.info(debt_info_list)
            p_debt_list.multi_insert(*debt_info_list)
        # 当前页超过40条记录，则尝试请求第二页，但30页之后一般为垃圾可以忽略
        if len(debt_base) < 40 or page > 20:
            break
        page += 1


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
            succ = try_buy_debt(ids[i])
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
        config.svc.evaluate(dx[-800:], dy[-800:])
    # run
    wait_sec = 1

    i = 0
    while True:
        try:
            fetch_debt_list()
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
    log.basicConfig(filename='log/debt.log', filemode="w", level=log.INFO, format='%(message)s')
    save_pid('debt')
    run()


if __name__ == '__main__':
    main()
