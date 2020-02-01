#
import re, random, os
import math
import pickle, json, yaml
import numpy as np
from time_plus import *
import cacheout

'''存储相关的方法'''


def shuffle2(x, y):
    ln = len(x)
    indexes = list(range(ln))
    random.shuffle(indexes)
    tx = [0] * ln
    ty = [0] * ln
    for i in range(ln):
        tx[i], ty[i] = x[indexes[i]], y[indexes[i]]
    return tx, ty


def shuffle3(a, b, c):
    ln = len(a)
    indexes = list(range(ln))
    random.shuffle(indexes)
    ta = [0] * ln
    tb = [0] * ln
    tc = [0] * ln
    for i in range(ln):
        ta[i], tb[i], tc[i] = a[indexes[i]], b[indexes[i]], c[indexes[i]]
    return ta, tb, tc


def shuffle4(a, b, c, d):
    ln = len(a)
    rand_idx = list(range(ln))
    random.shuffle(rand_idx)
    ta = [None] * ln
    tb = [None] * ln
    tc = [None] * ln
    td = [None] * ln
    for i in range(ln):
        ta[i], tb[i] = a[rand_idx[i]], b[rand_idx[i]]
        tc[i] = c[rand_idx[i]]
        td[i] = d[rand_idx[i]]
    return ta, tb, tc, td


dump_path = '../data/%s_%s.npy'


def dump2(pre, a, b, shuffle = True):
    if shuffle:
        a, b = shuffle2(a, b)
    np.save(dump_path % (pre, 'a'), np.array(a))
    np.save(dump_path % (pre, 'b'), np.array(b))


def dump3(pre, a, b, c, shuffle = True):
    if shuffle:
        a, b, c = shuffle3(a, b, c)
    np.save(dump_path % (pre, 'a'), np.array(a))
    np.save(dump_path % (pre, 'b'), np.array(b))
    np.save(dump_path % (pre, 'c'), np.array(c))


def dump4(pre, a, b, c, d, shuffle = True):
    if shuffle:
        a, b, c, d = shuffle4(a, b, c, d)
    np.save(dump_path % (pre, 'a'), np.array(a))
    np.save(dump_path % (pre, 'b'), np.array(b))
    np.save(dump_path % (pre, 'c'), np.array(c))
    np.save(dump_path % (pre, 'd'), np.array(d))


def pickle_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None


def pickle_dump(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def json_load(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def json_dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def yaml_dump(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f)


def yaml_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, )


def str_dump(path, obj):
    with open(path, 'w') as f:
        f.write(str(obj))


def str_load(path):
    try:
        with open(path, 'r') as f:
            ret = f.read(-1)
        return ret
    except:
        return ''


def int_dump(path, val):
    with open(path, 'w') as f:
        f.write(str(val))


def int_load(path):
    try:
        with open(path, 'r') as f:
            ret = int(f.read(-1))
        return ret
    except:
        return 0


'''字符串处理相关方法'''

_shr = re.compile(r'<[^>]+>', re.S)
_dig = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def strip_html(content):
    # 全部替换为空格，不然有些不加标点的就连一起了
    return _shr.sub(' ', content)


# 判断字符串是否包含数字
def has_number(in_str):
    for k in range(len(in_str)):
        if in_str[k] in _dig:
            return True
    return False


def str_full2half(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def str_half2full(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring


_rsr = re.compile('\s{2,}')


# 把全部超过2块的空格转为1块，因为没意义
def reduce_space(content):
    content = content.strip()
    while content.find('  ') >= 0:
        content = _rsr.sub(' ', content)
    return content


'''数学计算相关的方法'''


def divide(a, b):
    if b == 0:
        return 100
    return round(a / b * 100, 2)


def find_max_n(scores, N):
    ret_idx = np.zeros([N], dtype=np.int32)
    ret_scr = np.zeros([N], dtype=np.float32)
    for i in range(len(scores)):
        for j in range(N):
            if scores[i] > ret_scr[j]:
                for t in range(N - 1, j, -1):
                    ret_idx[t] = ret_idx[t - 1]
                    ret_scr[t] = ret_scr[t - 1]
                ret_idx[j] = i
                ret_scr[j] = scores[i]
                break
    return ret_idx


def find_min_n(scores, N, inf = 1 << 31):
    ret_idx = np.zeros([N], dtype=np.int32)
    ret_scr = np.zeros([N], dtype=np.float32) + inf
    for i in range(len(scores)):
        for j in range(N):
            if scores[i] < ret_scr[j]:
                for t in range(N - 1, j, -1):
                    ret_idx[t] = ret_idx[t - 1]
                    ret_scr[t] = ret_scr[t - 1]
                ret_idx[j] = i
                ret_scr[j] = scores[i]
                break
    return ret_idx

def softmax(inList):
    n = len(inList)
    outList = np.zeros((n))
    soft_sum = 0
    for idx in range(0,n):
        outList[idx] = math.exp(inList[idx])
        soft_sum += outList[idx]
    for idx in range(0,n):
        outList[idx] = outList[idx] / soft_sum
    return outList

def calc_percent(in_list):
    s = sum(in_list)
    in_list = np.array(in_list, dtype=np.float32)
    in_list /= s
    # for i in range(len(in_list)):
    #     in_list[i] /= s
    return in_list

def prob_rand_idx(rate):
    """随机变量的概率函数"""
    #
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    idx = 0
    randnum = random.randint(1, sum(rate))

    for idx in range(len(rate)):
        start += rate[idx]
        if randnum <= start:
            break
    return idx

# 不严格二分查找，返回后还需要判断一次
# arr必须为从小到大排序
def bi_search(arr, kw):
    left = 0
    right = len(arr)-1
    pos = 0
    while left <= right:
        pos = (right+left)//2
        if arr[pos] == kw:
            break
        elif kw < arr[pos]:
            right = pos-1
        else:
            left = pos+1
    return pos

# 基于指定概率分布生成的随机idx
class prob_rand():
    def __init__(self, rate, vals = None):
        scope = 0
        # prob = np.array(rate, dtype=np.float32)
        prob = calc_percent(rate)
        # 这个是概率积分，是递增的，可以用二分查找
        for i in range(len(prob)):
            scope += prob[i]
            prob[i] = scope
        self.prob = prob
        self.vals = vals

    def rand_idx(self):
        prob = self.prob
        r = random.random()
        pos = bi_search(prob, r)
        if r > prob[pos]:
            pos += 1
        return pos

    def rand_val(self):
        pos = self.rand_idx()
        return self.vals[pos]

# 基于三角形分布(等差数列)的随机idx
class tri_rand():
    # 要求rate是排序的
    def __init__(self, rate):
        a = rate[0]
        b = rate[-1]
        ln = len(rate)
        d = (b - a) / (ln - 1)
        self.ln = ln
        self.a = a
        self.dv = 0.5 - a / d
        self.sdv = np.square(self.dv)
        self.d = d
        # 不一定为整数
        self.total = ln * (a + b) / 2
        self.neg = 1
        if d < 0:
            self.neg = -1

    # 等差数列公式 s=n(a1+an)/2 -> s = a*n+n(n-1)d/2 -> (n-(1/2-a/d))^2=s*2/d+(1/2-a/d)^2 -> n= sqrt(s*2/d +(1/2-a/d)^2) +(1/2-a/d)
    # 等比数列公式 s=a1*(1-q)^n/(1-q) -> n = log_(1-q)^(s*(1-q)/a1)
    # 在个数较多时，使用公式计算结果比遍历应该要快些
    def rand_idx(self):
        rnum = random.randint(1, int(self.total))
        # self.rec.append(rnum)
        n = self.neg * np.sqrt(rnum * 2 / self.d + self.sdv) + self.dv
        if n is np.nan:  # 不知道为啥n会=nan
            return 0
        n = math.ceil(n) - 1
        if n >= self.ln:
            return self.ln - 1
        if n < 0:
            return 0
        return n


# 把一个计数dict转为两个排序后的列表, 正序
def get_sorted_list(dic, desc = False):
    sorted = list(dic.items())
    sorted.sort(key=lambda x: x[1], reverse=desc)
    key_list = np.zeros([len(sorted)], dtype=np.int32)
    rate_list = np.zeros([len(sorted)], dtype=np.int32)
    for i in range(len(sorted)):
        k, cnt = sorted[i]
        key_list[i] = k
        rate_list[i] = cnt
    return key_list, rate_list


def postpone_pos(ndarr, a, cnt = 100):
    b = a + cnt
    if b >= len(ndarr):
        b = len(ndarr) - 1
    val = ndarr[a]
    for i in range(a + 1, b + 1):
        ndarr[i - 1] = ndarr[i]
    ndarr[b] = val

# numpy 转one-hot
def convert_to_one_hot(x, total=None):
    if type(x) is not np.ndarray:
        x = np.array(x)
    if total is None:
        total = max(x)
    return np.eye(total)[x.reshape(-1)]

def split(d, rate = 0.7):
    pos = int(len(d) * rate)
    return d[:pos], d[pos:]

# 如果是-1,1分类，则mid=0
def bin_precise(logits, labels, mid = 0.5):
    total = len(logits)
    right = 0
    for i in range(total):
        val = 0 if logits[i] < mid else 1
        if val == labels[i]:
            right += 1
    return right / total * 100


def level_precise(logits, labels):
    lv = {9: [0, 0], 8: [0, 0], 7: [0, 0], 6: [0, 0]}
    for i in range(len(labels)):
        v = logits[i]
        for k in lv:
            a = k / 10
            b = a + 0.1
            if a <= v < b:
                lv[k][1] += 1
                if labels[i] == 1:
                    lv[k][0] += 1
                break
    acc = {}
    for k in lv:
        div = lv[k][1]
        if div == 0:
            div = 1
        acc[k] = lv[k][0] / div * 100
    return lv, acc


def cat_precise(logits, labels):
    total = len(logits)
    right = 0
    for i in range(total):
        if logits[i] == labels[i]:
            right += 1
    return right / total * 100


class Map(object):
    # 不能使用ttl，性能相差100倍
    def __init__(self, maxlen = None, lru = False):
        if lru:
            self._map = cacheout.LRUCache(maxsize=maxlen)
        else:
            self._map = cacheout.Cache(maxsize=maxlen)
        self._record = []

    # 这里默认value是个list，不然无法append
    def append(self, key, v):
        li = self._map.get(key)
        if li is None:
            li = [v]
        else:
            li.append(v)
        self._map.set(key, li)
    # def pop
    # 这里默认value是个int
    def incr(self, key, val = 1):
        v = self._map.get(key)
        if v is None:
            v = val
        else:
            v += val
        self._map.set(key, v)

    def items(self):
        return self._map.items()

    def keys(self):
        return self._map.keys()

    def set(self, key, val):
        self._map.set(key, val)

    def set_keep_max(self, key, val):
        org = self._map.get(key)
        if org is not None:
            val = max(org, val)
        self._map.set(key, val)

    def get(self, key, default = None):
        return self._map.get(key, default)

    def delete(self, key):
        self._map.delete(key)

    def record_del(self, k1):
        self._record.append(k1)

    def do_del(self):
        for k1 in self._record:
            self._map.delete(k1)
        self._record = []

    def len(self):
        return len(self._map)

    def __len__(self):
        return len(self._map)

    def clear(self):
        self._map.clear()


class TwoLvMap(Map):
    def __init__(self, max_len = None, lru = False, sub_max = None):
        super(TwoLvMap, self).__init__(max_len, lru)
        # self._map = cacheout.Cache(maxsize=max_len, ttl=ttl)
        # self._record = []
        self._sub_max_len = sub_max

    def get2(self, k1, k2):
        sub_map = self._map.get(k1)
        if sub_map is None:
            return None
        val = sub_map.get(k2)
        return val

    def set2(self, k1, k2, v):
        sub_map = self._map.get(k1)
        if sub_map is None:
            sub_map = Map(maxlen=self._sub_max_len)
        sub_map.set(k2, v)
        self._map.set(k1, sub_map)

    def set2_keep_max(self, k1, k2, v):
        sub_map = self._map.get(k1)
        if sub_map is None:
            sub_map = Map(maxlen=10000)
            sub_map.set(k2, v)
        else:
            sub_map.set_keep_max(k2, v)
        self._map.set(k1, sub_map)

    # 这里默认最后一级是个list，不然无法append
    def append2(self, k1, k2, v):
        sub_map = self._map.get(k1)
        if sub_map is None:
            sub_map = {}
        li = sub_map.get(k2)
        if li is None:
            li = [v]
        else:
            li.append(v)
        sub_map[k2] = li
        self._map.set(k1, sub_map)

    def incr2(self, k1, k2, val):
        v = self.get2(k1, k2)
        if v is None:
            v = val
        else:
            v += val
        self.set2(k1, k2, v)

import sched


class SchServ():
    def __init__(self, func, span = 1800, log_func = print):
        self._sch = sched.scheduler(time.time)
        # 执行间隔，单位s
        self._span = span
        _pid = os.getpid()
        str_dump('pid', _pid)
        self._i = 0
        self._f = func
        self.log = log_func

    def wrapper(self):
        self._i += 1
        self._f()
        self._sch.enter(self._span, 0, self.wrapper)
        self.log('wait for next wakeup', self._i, std_datetime())

    def run(self):
        self.wrapper()
        self._sch.run()

    def start(self):
        while True:
            self._i += 1
            self._f()
            self.log('wait for next wakeup', self._i, std_datetime())
            time.sleep(self._span)
