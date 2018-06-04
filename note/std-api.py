'''基本语法'''
iter([]) # 把一个可迭代的类型转为迭代器 ps:有个毛用，直接遍历不就ok了


from collections import defaultdict
dd = defaultdict(list) # 带默认值的map，参数是一个类型. key取不到时返回类型的初始化值

from itertools import zip_longest
zip_longest([], fillvalue=None) # 把一堆sequence并排放一起，遍历到最长的那个结束为止，其它的用fillvalue填充

import re # 正则表达式

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
' a b '.strip().split() # strip=trim() 去除前后空格; split()默认以空格分割
_WORD_SPLIT.split('a;b,c') # 正则分割
{'a':1}.get('b', 2) # 第二个值是default value

def upload(arg):
    import requests
    url = 'http'
    fn = 'iam-a-file'
    r = requests.post(url, data=None, files={'upload_file':open(fn, 'rb')}, timeout=30) # 上传文件
    return r.json()

# 多线程
from multiprocessing import Pool
pool = Pool(2)
pool.map(upload,[1,2])
pool.close()
pool.join()





