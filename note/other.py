# encoding = utf8

"""hpfrec"""
import pandas as pd
import hpfrec
udf = pd.DataFrame()
uid = 1001
# 手动reindex是ok的
users = pd.Index(udf.user.unique())
items = pd.Index(udf.item.unique())
hpfdf = pd.DataFrame({'UserId': users.get_indexer(udf.user), 'ItemId': items.get_indexer(udf.item), 'Count': udf.ratings.values.copy()})
hm = hpfrec.HPF(k=30, maxiter=100, stop_crit='train-llk', save_folder='ckpt', reindex=False, stop_thr=1e-3)
hm.fit(hpfdf)
top = hm.topN(uid, 500)
top = items[top]

# 自动reindex，结果不对，不知道为什么
udf.columns = ['UserId', 'ItemId', 'Count']
hm = hpfrec.HPF(k=30, maxiter=100, stop_crit='train-llk', stop_thr=1e-3, save_folder='hpf', reindex=True)

"""faiss"""
import faiss
import numpy as np
dir = '50pct2/'
theta = pd.read_csv(dir+'Theta', header=None)
beta = pd.read_csv(dir+'Beta', header=None)

d = 30
fidx = faiss.IndexFlatIP(d)
beta = beta.values.astype('float32')
# 不然会报错
beta = np.ascontiguousarray(beta, 'float32')
# beta[:, 0] += np.arange(len(beta)) / 1000.
print(beta)
fidx.add(beta)
theta = theta.values
theta = np.ascontiguousarray(theta, 'float32')
_, ret = fidx.search(theta[:2], 500)
print(ret)

