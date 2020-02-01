import pandas as pd
import os

print("Running from", os.path.dirname(os.path.realpath(__file__)))

data = pd.read_csv("data/tb_cate_modified.csv", encoding="utf8")
# 把 nan 数据替换为 None
data = data.where(data.notnull(), None)


first_name = []
for i in data["first_name"]:
    if i:
        if i in first_name:
            continue
        else:
            first_name.append(i)

a = data.loc[data["first_name"] == "众筹 ", "second_name"]


remove1 = [" 动漫", " 公益", " 其他", " 书籍", " 游戏", " 教育", " 旅游", " 手艺人", " 设计师"]
remove2 = [" CPU", " 电源", " 电脑周边 ", " 电脑视听配件 ", " 固态硬盘", " 机箱", " 内存", " 散热器/风扇",
           " 声卡", " 显卡"," 显示器&支架 "," 硬件套装"," 主板"," 智能电脑硬件"]


b = data.loc[data["second_name"] == " 智能电脑硬件", "source_cid"]
print(b)