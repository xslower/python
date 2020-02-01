# coding=utf-8
import yaml
import numpy as np
import pickle
import jieba
from jieba import posseg as pseg
import random
import json
from mini_orm import *
import os,sys

dir, _ = os.path.split(os.path.abspath(sys.argv[0]))
path = dir+'/db_conf.yml'

db_conf_local = {"host": "192.168.67.223", "port": 3306, "user": "root", "password": "123456", "database": "qtt", "charset": "utf8"}
db_conf_rds = {"host": "rm-uf6vo95pi21xv41u8oo.mysql.rds.aliyuncs.com", "port": 3306, "user": "root", "password": "RY0uUYQUeOLaF8qn", "database": "biaoqing_deal", "charset": "utf8"}

try:
    with open(path, mode='r', encoding='utf-8') as f:
        db_config = yaml.load(f, Loader=yaml.loader.BaseLoader)
except Exception as e:
    print(e)
    db_config = db_conf_rds

Conn.set_config(db_config)
