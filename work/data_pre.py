# coding=utf-8
import sys

sys.path.append("../lib")

import json
import yaml
from mini_orm import *
import numpy as np
import pickle

# db_def = {
#     "host": "rm-uf6tz1g9l077nkd0do.mysql.rds.aliyuncs.com",
#     "port": 3306,
#     "user": "rdsroot",
#     "password": "yinnuo123!@#",
#     "database": "lab"
# }
# try:
#     f = open('data/db.yml', 'r')
#     db_config = yaml.load(f)
# except:
#     db_config = db_def
db_config = {"host": "192.168.68.222", "port": 3306, "user": "root", "password": "", "database": "title_filter"}
Conn.connect(**db_config)


class sentence_vector(Model):
    sentence = Field()
    vector_array = Field()
    segmentation = Field()
    hit_target = Field()

def prepare_data():
    labeled_len = 12336
    sens = sentence_vector.where().lt('id', labeled_len).select()
    X = []
    Y = []
    max_len = 22
    zero = [0] * 300
    for s in sens:
        arr = s.vector_array.replace("'", '')
        s.vector_array = arr
        # s.update()
        try:
            vec = json.loads(arr)
        except:
            continue
        if len(vec) < max_len:
            vec = [zero] * (max_len - len(vec)) + vec
        print(s.sentence, len(vec))
        y = s.hit_target
        X.append(vec)
        Y.append(y)
    fpx = open('data/train_x', mode='wb')
    pickle.dump(X, fpx)
    fpy = open('data/train_y', mode='wb')
    pickle.dump(Y, fpy)
    # print('max len:', max_len)

if __name__ == '__main__':
    prepare_data()