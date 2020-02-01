import pandas as pd
import numpy as np
import os

data_name = "tb_cate_modified.csv"
path_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/" + data_name)

data = pd.read_csv(path_data, encoding="utf8")
# 把 nan 数据替换为 None
data = data.where(data.notnull(), None)

filter_name = "filter.xlsx"
path_filter = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/" + filter_name)
filter_ = pd.read_excel(path_filter, encoding="utf8")

r_first_cid_list = []
for i in range(len(filter_)):
    if filter_["filter"][i] == 0:
        r_first_cid_list.append(filter_["first_cid"][i])
print(r_first_cid_list)

remove1 = [" 动漫", " 公益", " 其他", " 书籍", " 游戏", " 教育", " 旅游", " 手艺人", " 设计师"]
remove2 = [" CPU", " 电源", " 电脑周边 ", " 电脑视听配件 ", " 固态硬盘", " 机箱", " 内存", " 散热器/风扇",
           " 声卡", " 显卡", " 显示器&支架 ", " 硬件套装", " 主板", " 智能电脑硬件"]

def get_second_cid_list(r_second_name_list, first_cid):
    data_ = data.loc[data["first_cid"] == first_cid]
    res = []
    for i in r_second_name_list:
        a = data_.loc[data_["second_name"] == i, "second_cid"]
        if len(a) == 1:
            second_cid = int(a)
        else:
            second_cid = list(a)[0]
        res.append(second_cid)
    return res

r_second_cid_list = get_second_cid_list(remove1, 121266001) + get_second_cid_list(remove2, 11)
print(r_second_cid_list)


np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "id_list/r_first_cid_list.npy"), r_first_cid_list)
np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "id_list/r_second_cid_list.npy"), r_second_cid_list)

