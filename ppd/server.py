# coding=utf8

from header import *
# from svm import *
# from api_op import *
import svc
import api_op as api

_key = 'Result'
_msg = 'ResultMessage'

def try_buy_bid(list_id, amount):
    succ = False
    for i in range(len(config.users)):
        u = config.users[i]
        ret = api.buy_bid(u.tk, list_id, amount)
        if ret is None or _key not in ret.keys():
            continue
        code = ret[_key]
        if code == 0:
            succ = True
            break
        elif code == 4001:  # 用户余额不足，换人
            continue
        else:
            log.info('%s', ret)
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


def try_buy_debt(debt_id):
    succ = False
    for u in config.users:
        ret = api.buy_debt(u.tk, debt_id)
        log.info('buy_debt %d ret %s ', debt_id, ret)
        if ret is None or _key not in ret.keys():
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


def buy_any(ids, amount=config.bid_amount, debt=False):
    if debt:
        return buy_debts(ids)
    else:
        return buy_bids(ids, amount)


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


# def deal_loan(binfo, succ_list, fail_list, bad_list):
#     data_x = []
#     ids = []
#     for dic in binfo:
#         _id = dic[api.k_list_id]
#         if dic['CreditCode'] == 'AA':
#             if dic['CurrentRate'] >= 10:
#                 succ = try_buy_bid(_id, 200)
#                 if succ:
#                     succ_list.append(_id)
#                 else:
#                     fail_list.append(_id)
#             # 利率低于10的AA不投
#             continue
#         row = dicToX(dic)
#         data_x.append(row)
#         ids.append(_id)
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
#     succ, fail = buy_bids(good_list)
#     succ_list.extend(succ)
#     fail_list.extend(fail)


def fetch_loan_list():
    bid_info_list = []
    succ_list = []
    fail_list = []
    bad_list = []
    page = 1
    while True:
        bids = api.get_loan_list(page)
        if bids is None or len(bids) == 0:
            return
        # p_loan_list.multi_insert(*bids)
        ids = get_not_aa_vals(bids, k_list_id)
        log.info('ids %s', ids)
        if len(ids) == 0:
            return
        if len(ids) <= 10:
            binfo = api.get_bid_info(ids)
            predict_bid(binfo, succ_list, fail_list, bad_list)
            binfo = _merge_bid(binfo, bids)
            bid_info_list.extend(binfo)
            break
        else:
            i = 0
            while True:
                start = i * api.page_limit
                end = min(start + api.page_limit, len(ids))
                if end <= start:
                    break
                i += 1
                # log.info('%d %d %d', i, start, end)
                binfo = api.get_bid_info(ids[start:end])
                if binfo is None or len(binfo) == 0:
                    continue
                predict_bid(binfo, succ_list, fail_list, bad_list)
                binfo = _merge_bid(binfo, bids)
                bid_info_list.extend(binfo)
        if len(bids) < 20:
            break
        page += 1
    update_bid_db(bid_info_list, succ_list, fail_list, bad_list, field='listingId', mod=p_bids_avail)


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
    cc = api.credit_code
    cur_code = cc.map[debt['CurrentCreditCode']]
    sale_rate = debt['PriceforSaleRate']
    ought_rate = cc.rate[cur_code]
    price = debt['PriceforSale']

    # 逾期利率惩罚
    over_due = debt['PastDueNumber']
    if over_due > 0:  # 第一次加5个点，之后每多一次多2个点
        ought_rate += 5 + (over_due - 1) * 2
    # 剩余的期数太少，导致真实利率下降。
    left_num = debt['OwingNumber']
    if left_num == 1:
        ought_rate += 1.2
    elif left_num == 2:
        ought_rate += 0.6
    # 根据最近还款日短近，增加利率要求。
    days = debt['Days']
    if days <= 15:  # 还款距离越短，利率要求越高
        ought_rate += (15 - days) ** 2 * 0.011

    # 限额根据利率优惠的幅度上下调整
    limit = cc.limit[cur_code] #* (1 + (sale_rate - ought_rate) / 10)

    # 利率让利超过一定限额，且金额较小，则直接购买不走模型。目的就是买数据
    # if sale_rate > ought_rate + cc.want_rate['pre_buy'] and \
    #                 price < limit / 5:
    #     pre_buy(debt)
    #     return False

    if price > limit:  # 债权份额大于对应等级的限额
        return False
    code = cc.map[debt['CreditCode']]
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
    succ_list = []
    fail_list = []
    bad_list = []
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
            predict_bid(bid_info, succ_list, fail_list, bad_list, True)
            bid_info_list.extend(bid_info)

        log.info('page:%d debt len %d', page, len(debt_info_list))
        if len(debt_info_list) > 0:
            # log.info(debt_info_list)
            p_debt_list.multi_insert(*debt_info_list)
        # 当前页超过40条记录，则尝试请求第二页，但30页之后一般为垃圾可以忽略
        if len(debt_base) < 40 or page > 30:
            break
        page += 1
    update_bid_db(bid_info_list, succ_list, fail_list, bad_list, field='debtId', mod=p_debt_list)


def predict_bid(binfo, succ_list, fail_list, bad_list, debt=False):
    if debt:
        id_key = api.k_debt_id
    else:
        id_key = api.k_list_id
    data_x = []
    ids = []
    for dic in binfo:
        _id = dic[id_key]
        if dic['CreditCode'] == 'AA':
            # AA不投
            continue
        row = svc.dicToX(dic)
        data_x.append(row)
        ids.append(_id)
    if len(data_x) == 0:
        return
    log.info('before prediect. ids: %s', ids)
    # 合并多种预测的结果，任何一个预测危险都不投
    y_pred = config.svc.predict(data_x)
    log.info('predict: %s', y_pred)
    for i in range(len(y_pred)):
        is_bad = False
        if y_pred[i] == 1:
            is_bad = True
        if is_bad:
            bad_list.append(ids[i])
        else:
            if debt:
                succ = try_buy_debt(ids[i])
            else:
                succ = try_buy_bid(ids[i], config.bid_amount)
            if succ:
                succ_list.append(ids[i])
            else:
                fail_list.append(ids[i])


# 根据余额调节各标的要求
def amount_control():
    total = 0
    rev = False
    for i in range(len(config.users)):
        u = config.users[i]
        left = api.left_amount(u.tk)
        if left < config.bid_amount and i < 1:
            rev = True
        total += left
    if rev:  # 有余额少的就扔到后面去
        config.users.reverse()
    # for i in range(len(config.stand_by_users) - 1, -1, -1):
    #     u = config.stand_by_users[i]
    #     left = api.left_amount(u.tk)
    #     if left > 3 * config.bid_amount:
    #         config.users.append(u)
    #         del (config.stand_by_users[i])
    #         # delete.append(i)
    #         total += left
    # log.info('%s ' * len(config.users), *config.users)
    f = open('config.yml', 'r')
    conf = yaml.load(f)
    for key in conf:
        cf = conf[key]
        if cf['ge'] <= total < cf['lt']:
            api.credit_code.set_want_rate(**cf['rate'])


def run(debt=False):
    code = [None]
    for c in code:
        dx, dy = svc.init_data(c)
        config.svc = svc.my_svc()
        config.svc.train(dx, dy)
    # run
    wait_sec = 0.5
    if debt:
        # wait_sec = 2
        fetch_list = fetch_debt_list
    else:
        fetch_list = fetch_loan_list

    amount_control()
    i = 0
    while True:
        fetch_list()
        i += 1
        if i % 200 == 0:
            i = 0
            amount_control()
            config.reload_token()
        time.sleep(wait_sec)


def main():
    config.init()
    mode = 'norm'
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    log.basicConfig(filename='log/%s.log' % mode, filemode="w", level=log.INFO, format='%(message)s')
    save_pid(mode)
    if mode == 'norm':
        run()
    elif mode == 'debt':
        run(True)
    else:
        log.info('not support arg')


if __name__ == '__main__':
    main()
