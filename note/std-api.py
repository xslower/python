'''基本语法'''
## set=去重的list tuple=不能修改的list
a = 1
z = str(a).zfill(5) # z='00001'
a = [1,2,3]
a.remove(2) # 去掉值为2的元素
a.pop(1) # 去掉位置=1的元素
del(a[1]) # 与上面等价

iter([]) # 把一个可迭代的类型转为迭代器 ps:有个毛用，直接遍历不就ok了
zip([1,2,3], [4,5,6]) # 同时迭代两个迭代器，输出[1,4],[2,5],[3,6]

# 排序
d = {'a':3,'b':2,'c':1}
l = list(d.items())
l.sort(key=lambda x:x[1]) # [('c', 1), ('b', 2), ('a', 3)]

from collections import defaultdict,Counter,deque, defaultdict, OrderedDict
dd = defaultdict(list) # 带默认值的map，参数是一个类型. key取不到时返回类型的初始化值
#(见：https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431953239820157155d21c494e5786fce303f3018c86000)
c = Counter([1,1,2]) #计数器类，可以统计string/list内各个元素的频率
d = deque() #一个双向链表，头尾插入删除比list性能高，支持appendleft/popleft
dd = defaultdict() #key不存在时可以返回默认值
odo = OrderedDict() #遍历时按key的顺序访问

import datetime
a = datetime.datetime.strptime('2018年11月30日','%Y年%m月%d日')

from itertools import zip_longest
zip_longest([], fillvalue=None) # 把一堆sequence并排放一起，遍历到最长的那个结束为止，其它的用fillvalue填充
from functools import reduce
reduce(lambda x, y: x+y, [1,2,3]) # =6, 依后面系列的顺序调用前面的函数。

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

import pickle
with open('a', 'wr') as f:
    d = pickle.load(f)
with open('a', 'w') as f:
    pickle.dump(d, f)


# 关于值复制
import copy
a = []
b = a #这种情况是将a和b放在同一个引用上了，不算是copy
b = a.copy() #这个方法和下面的2个方法，虽然id(b) != id(a)，但是里面的对象id是一样的
b = a[:]
b = list(a)
b = copy.deepcopy(a) #只有这种方法copy了列表里面的每一个对象


import difflib
scr = difflib.SequenceMatcher(a=a, b=b).quick_ratio() # 计算两个字符串的相似度

from itertools import chain
L=[ [1,2,3],[4,5,6],[7,8,9]]
lst = list(chain(*L)) # 类似numpy的concatenate, 合并list