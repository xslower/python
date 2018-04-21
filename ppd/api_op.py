# coding=utf-8

from header import *

page_limit = 10


# 用户已投标情况
def get_bid_list(token, start, end, idx):
    data = {'StartTime': time_plus.std_date(start), 'EndTime': time_plus.std_date(end), 'PageIndex': idx, 'PageSize': page_limit}
    ret = pcli.send(url.u_bid_list, data, token)
    # print(ret)
    bl = []
    max_page = 1
    k_bl = 'BidList'
    k_tp = 'TotalPages'
    keys = ret.keys()
    if k_bl in keys and k_tp in keys:
        bl = ret[k_bl]
        max_page = ret[k_tp]
    else:
        log.info(ret)
    return bl, max_page


# 标的信息
def get_bid_info(ids):
    data = {'ListingIds': ids}
    ret = pcli.send(url.bid_info, data)
    key = 'LoanInfos'
    return get_dict_vals(ret, key)


# 标的还款情况
def get_repay_info(token, lid):
    data = {'ListingId': lid}
    ret = pcli.send(url.repay, data, token)
    key = 'ListingRepayment'
    return get_dict_vals(ret, key)


# 可投标信息
def get_loan_list(page = 1):
    data = {'PageIndex': page}
    ret = pcli.send(url.loan_list, data)
    key = 'LoanInfos'
    return get_dict_vals(ret, key)


# 可买债权信息
def get_debt_list(page = 1, start = None):
    data = {'PageIndex': page, 'Levels': 'AA,A,B,C,D,E'}
    if start is not None:
        data['StartDateTime'] = start
    ret = pcli.send(url.debt_list, data)
    key = 'DebtInfos'
    return get_dict_vals(ret, key)


# 债权的信息
def get_debt_info(ids):
    data = {'DebtIds': ids}
    ret = pcli.send(url.debt_info, data)
    key = 'DebtInfos'
    return get_dict_vals(ret, key)


def buy_bid(tk, lid, amount):
    data = {'ListingId': lid, 'Amount': amount}
    data['UseCoupon'] = 'true'
    ret = pcli.send(url.buy_bid, data, tk)
    return ret


def buy_debt(tk, debt_id):
    data = {'debtDealId': debt_id}
    ret = pcli.send(url.buy_debt, data, tk)
    return ret


def left_amount(tk):
    ret = pcli.send(url.left, {}, tk)
    key = 'Balance'
    if ret is not None and key in ret.keys():
        return ret[key][0][key]
    log.warn(ret)
    return -1


def get_ids(bids):
    ids = []
    for bid in bids:
        ids.append(bid.listingId)
    return ids


# 请求用户出借记录
def bid_list(open_id, start_after = 0, span = 20):
    log.info('bid_list:')
    token = config.get_token(open_id)
    now = time_plus.now()
    # now = time_plus.timestamp('2017-08-12 00:00:00')
    end = now
    start = now - time_plus.day * span
    while True:
        log.info('%s %s', time_plus.std_date(start), time_plus.std_date(end))
        i = 1
        max_page = 100
        has_record = False
        while i <= max_page:
            bl, max_page = get_bid_list(token, start, end, i)
            log.info('%d %d', i, max_page)
            if len(bl) < 1:
                break
            p_bids_real.multi_insert(*bl)
            ids = get_list_vals(bl, k_list_id)
            i += 1
            if len(ids) == 0:
                continue
            log.info(ids)
            p_bids_real.where().in_('listingId', *ids).update(openId=open_id)
            binfo = get_bid_info(ids)
            if binfo is None or len(binfo) == 0:
                continue
            has_record = True
            p_bids_real.multi_insert(*binfo)
            if i % 2 == 0:
                time.sleep(1)
        if not has_record:
            break
        end = start
        start -= time_plus.day * span
        if start < start_after:
            break


# 更新还款信息。这个要定期更新,一周更新一次。12-8
def update_repay(open_id):
    tk = config.get_token(open_id)
    bid_iter = p_bids_real.where(openId=open_id).ge('listingId', 0).lt('repayStatus', 98).select()
    for bid in bid_iter:
        lid = bid.listingId
        if lid < 10:
            continue
        log.info(lid)
        # 确认是哪种标
        if bid.bidType == 1:
            p_repay = p_repay_norm
        elif bid.bidType == 2:
            p_repay = p_repay_debt
        else:
            log.info('bid type', bid.bidType)
            p_repay = p_repay_norm

        # 获取还款信息
        pay_info = get_repay_info(tk, lid)
        if len(pay_info) < 1:
            continue
        days = 0
        pay_month = 0
        months = 0
        # 删除原有数据，防止提前还款导致无效数据存在
        p_repay.where(listingId=lid).delete()
        # 计算还款状态
        for rep in pay_info:
            # p_repay.insert_or_update(**rep)
            if rep[k_rp_overdue] > days:
                days = rep[k_rp_overdue]
            months += 1
            if 1 <= rep[k_rp_status] <= 3:
                pay_month += 1
        bid.overdueDays = days
        p_repay.multi_insert(*pay_info)
        # 提前还款会删除后面还款期，所以pay_month与months依然应该相等
        bid.repayStatus = pay_month / months * 100
        if bid.bidType == 1 and months < bid.months and bid.repayStatus < 99:
            log.info(bid)

        bid.save()
        time.sleep(0.5)
    pass


# 用来计算还款状态和逾期天数, 没啥用了
def fix_data():
    bid_iter = p_bids_real.where().select()
    for bid in bid_iter:
        repay_iter = p_repay_norm.where(listingId=bid.listingId).select()
        days = 0
        pay_month = 0
        months = 0
        for rep in repay_iter:
            months += 1
            if rep.overdueDays > days:
                days = rep.overdueDays
            if 1 <= rep.repayStatus <= 3:
                pay_month += 1
                # month = rep.orderId
        bid.overdueDays = days
        log.info('%d %d %d', pay_month, months, bid.months)
        if months < bid.months:
            bid.repayStatus = 100
        else:
            bid.repayStatus = pay_month / months * 100
        bid.save()


def test():
    tk = config.get_token(xslower_id)
    info = left_amount(tk)


# 债权还款信息合并到update_repay里了
def update_debt():
    with open('data/debt_list.json', 'r') as f:
        debt_list = json.load(f)
    ln = len(debt_list) // 10
    for i in range(0, ln, 1):
        start = i * 10
        end = min(start + 10, len(debt_list))
        ids = debt_list[start:end]
        print(ids)
        bids = get_bid_info(ids)
        if len(bids) > 0:
            p_bids_real.multi_insert(*bids)
    p_bids_real.where().in_(k_list_id, *debt_list).update(bidType=2, openId=xslower_id)



def update_all():
    pcli.timeout = 30
    log.basicConfig(filename='log/update.log', filemode="w", level=log.INFO, format='%(message)s')
    span = 3600 * 24 * 6
    harf_day = 3600 * 12
    last = int(time.time()) - 3600 * 24 * 30
    while True:
        now = int(time.time())
        if now - last < span:
            time.sleep(harf_day)
            continue
        # 只更新最近20天的投标信息
        start_after = int(time.time()) - 30 * 24 * 3600
        bid_list(xslower_id, start_after)
        # bid_list(niude_id, start_after)
        update_repay(xslower_id)
        # update_repay(niude_id)
        config.refresh_token()
        os.popen('sh restart_norm.sh', mode='w')
        # os.popen('restart_debt.sh')
        last = now
        log.info(time_plus.std_datetime(now))


if __name__ == '__main__':
    # update_debt()
    update_all()
    # config.reload_token()
"""
todo 
1.仔细研究下数据归一化，非常重要
3.用来尝试给神经网络加速
4.一二个月之后统计下ABCD的真实收益情况，如果相差较大则分开建模。
"""
