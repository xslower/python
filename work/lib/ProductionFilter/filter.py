import pandas as pd
import numpy as np
import os

data_name = "tb_cate_modified.csv"
path_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/" + data_name)
data = pd.read_csv(path_data, encoding="utf8")

# 读取已持久化的剔除商品的一级类目 id 和二级类目 id 列表
r_first_cid_list = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "id_list/r_first_cid_list.npy"))
r_second_cid_list = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "id_list/r_second_cid_list.npy"))


class Judgement():
    def __init__(self, source_cid_list, data=data):
        # 如果只输入一个 source_cid
        if type(source_cid_list) is int:
            source_cid_list = [source_cid_list]
        self.source_cid_list = source_cid_list
        self.data = data.where(data.notnull(), None)

    def get_first_cid(self, source_cid):
        first_cid = data.loc[data["source_cid"] == source_cid, "first_cid"]
        return int(first_cid)

    def get_second_cid(self, source_cid):
        second_cid = data.loc[data["source_cid"] == source_cid, "second_cid"]
        return int(second_cid)

    def main(self, r_first_cid_list=r_first_cid_list, r_second_cid_list=r_second_cid_list):
        def judge(source_cid):
            first_cid = self.get_first_cid(source_cid)
            second_cid = self.get_second_cid(source_cid)
            if first_cid in r_first_cid_list:
                return False
            if second_cid in r_second_cid_list:
                return False
            return True

        judge_res = []
        for source_cid in self.source_cid_list:
            judge_res.append(judge(source_cid))
        return judge_res

if __name__ == "__main__":

    # source_cid = [50050521, 50025007, 50015372, 50023105]
    source_cid = 50050521
    a = Judgement(source_cid)

    res = a.main()
    print(res)