# coding=utf-8
import sys

sys.path.append("../lib")

import json
import yaml
from mini_orm import *

db_def = {
    "host": "rm-uf6tz1g9l077nkd0d.mysql.rds.aliyuncs.com",
    "port": 3306,
    "user": "rdsroot",
    "password": "yinnuo123!@#",
    "database": "lab"
}
try:
    f = open('data/db.yml', 'r')
    db_config = yaml.load(f)
except:
    db_config = db_def
Conn.connect(**db_config)

k_list_id = 'ListingId'
k_debt_id = 'DebtId'
k_rate = 'Rate'
k_current_rate = 'CurrentRate'

k_rp_overdue = 'OverdueDays'
k_rp_status = 'RepayStatus'


class p_user(Model):
    open_id = Field()
    access_token = Field()
    refresh_token = Field()
    expires_in = Field()
    debt = Field()


# 记录实际投标信息
class p_bids_real(Model):
    listingId = Field()
    openId = Field()
    title = Field()
    # rate = Field()
    currentRate = Field()
    amount = Field()
    bidAmount = Field()
    months = Field()
    creditCode = Field()
    borrowName = Field()
    gender = Field()
    registerTime = Field()
    educationDegree = Field()
    graduateSchool = Field()
    studyStyle = Field()
    age = Field()
    successCount = Field()
    wasteCount = Field()
    cancelCount = Field()
    failedCount = Field()
    normalCount = Field()
    overdueLessCount = Field()
    overdueMoreCount = Field()
    owingPrincipal = Field()
    owingAmount = Field()
    certificateValidate = Field()
    nciicIdentityCheck = Field()
    phoneValidate = Field()
    videoValidate = Field()
    creditValidate = Field()
    educateValidate = Field()
    highestPrincipal = Field()
    highestDebt = Field()
    totalPrincipal = Field()
    auditingTime = Field()
    overdueDays = Field()
    repayStatus = Field()
    bidType = Field()

# 保持正常标的还款信息
class p_repay_norm(Model):
    listingId = Field()
    orderId = Field()
    dueDate = Field()
    repayDate = Field()
    repayPrincipal = Field()
    repayInterest = Field()
    owingPrincipal = Field()
    owingInterest = Field()
    owingOverdue = Field()
    overdueDays = Field()
    repayStatus = Field()

# 保存债权还款信息
class p_repay_debt(Model):
    listingId = Field()
    orderId = Field()
    dueDate = Field()
    repayDate = Field()
    repayPrincipal = Field()
    repayInterest = Field()
    owingPrincipal = Field()
    owingInterest = Field()
    owingOverdue = Field()
    overdueDays = Field()
    repayStatus = Field()

#这个是记录自动购买标的情况的
class p_bids_avail(Model):
    listingId = Field()
    title = Field()
    rate = Field()
    currentRate = Field()
    amount = Field()
    months = Field()
    creditCode = Field()
    borrowName = Field()
    gender = Field()
    registerTime = Field()
    educationDegree = Field()
    graduateSchool = Field()
    studyStyle = Field()
    age = Field()
    successCount = Field()
    wasteCount = Field()
    cancelCount = Field()
    failedCount = Field()
    normalCount = Field()
    overdueLessCount = Field()
    overdueMoreCount = Field()
    owingPrincipal = Field()
    owingAmount = Field()
    certificateValidate = Field()
    nciicIdentityCheck = Field()
    phoneValidate = Field()
    videoValidate = Field()
    creditValidate = Field()
    educateValidate = Field()
    highestPrincipal = Field()
    highestDebt = Field()
    totalPrincipal = Field()
    auditingTime = Field()
    status = Field()

# 记录债权自动购买情况的
class p_debt_list(Model):
    debtId = Field()
    listingId = Field()
    seller = Field()
    bidDateTime = Field()
    owingNumber = Field()
    owingPrincipal = Field()
    owingInterest = Field()
    days = Field()
    priceforSaleRate = Field()
    priceforSale = Field()
    preferenceDegree = Field()
    creditCode = Field()
    currentCreditCode = Field()
    listingAmount = Field()
    listingMonths = Field()
    listingTime = Field()
    listingRate = Field()
    pastDueNumber = Field()
    status = Field()


def test():
    debt = {'DebtId': 43947627, 'Seller': 'pdu2552164374', 'StatusId': 1, 'Lender': 'pdu2552164374',
            'BidDateTime': '2017-07-23T10:43:01.58', 'OwingNumber': 2, 'OwingPrincipal': 34.56, 'OwingInterest': 0.95,
            'Days': 8, 'PriceforSaleRate': 22.0, 'PriceforSale': 35.02, 'PreferenceDegree': -6.52,
            'ListingId': 60429176, 'CreditCode': 'E', 'CurrentCreditCode': 'A', 'ListingAmount': 5000.0,
            'ListingTime': '2017-07-23T10:42:37.007', 'ListingMonths': 6, 'ListingRate': 22.0, 'PastDueNumber': 0,
            'AllowanceRadio': 0.0, 'PastDueDay': 0, 'status': 3}
    p_debt_list.insert_or_update(**debt)


def t_read():
    js = '{"OpenID":"b92691f48df9496d973113d2ae89d5a1","AccessToken":"475f0f2b-2539-4030-bc98-87ede82a1aa3","RefreshToken":"27819357-dc7a-4112-8ec1-3f81ea3284d8","ExpiresIn":604800}'
    r = json.loads(js)
    tk = p_user.read(open_id=r["OpenID"])
    print(tk.__dict__)


if __name__ == '__main__':
    test()
