import numpy as np

#
a = np.array((1, 2, 3)) # 转为np.array类型，依然是sequence
b = np.array((2, 3, 4), dtype=np.int32) # 转为np的矩阵或张量，所以子元素的长度要求相同
c=a@b #内积，=keras的Dot()
c=a+b # n*1 + 1*k 时= a(复制k列)+b(复制n行)=n*k
c=a*b # 就是按位相乘

# 貌似是拼装数组的
np.hstack((a, b))  # array([1, 2, 3, 2, 3, 4])
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.hstack((a, b))  # array([[1, 2],[2, 3],[3, 4]])
np.append(a, [4], axis=None)  # 前后两个参数的shape必须相同； axis=None会把前后两个参数flatten掉，然后连在一起；axis=i会把第i维连接，类似tf.concat
np.concatenate([a, b], axis=0) # 与上面相同

np.insert(a, 1, [5])  # 在位置1之前插入5
np.delete(a, [1,2]) # 删除指定位置的元素
# 生成一个指定范围的数组
np.arange(3, 7)  # array([3, 4, 5, 6])
np.empty([2, 2], dtype=np.int32)  # 生成一个指定形状和类型的张量，不进行初始化，所以值是不定的
np.zeros([3, 3], dtype=np.int8)  # 生成一个指定形状和类型的张量，全=0
np.ones([2, 2, 2], dtype=np.float16)  # 生成指定形状和类型的张量，全=1
np.ndarray([2,3], dtype=np.float32) # 生成一个随机数组
np.amax(a, axis=1)  # =np.max() 求数组中最大的值，不带axis则flatten后求，带上axis则求某一维的最大值，例如np.max(x3d, axis=0) 则x3d[max][i][j] =x2d
np.argmax(a, axis=1)  # 上面是求最大值，这个是取最大值所在的index。axis是计算指定维内的最大值的idx，如果不指定axis，则会把a flatten之后取idx.
np.clip(a, a_min=1, a_max=10) # 把a中所有元素限制在min/max之间
mean = np.mean(a, axis=0) # 均值，不指定axis计算全部值的均值，指定后计算延着axis维的值的均值
std = np.std(a, axis=0)  # 标准差 (a-meam)/std # 标准化

new_a = a[np.newaxis, :]  # 在最前面增加一个纬度。new_a=[[1,2,3]]
a = np.reshape(a, [2,3])
new_b = a[:, np.newaxis]  # 最后面增加一维，=[[1],[2],[3]]
b = np.squeeze(a, axis=None) # 把len=1的维度消去，[[0],[1]]=[0,1], axis指定纬度后只消指定维,指定维度的len必须=1
c = np.tile(a, (3,)) # 在指维度上复制，在2维复制=(3,1)，3维=(5,1,1)

'''概率'''
np.random.randint(low=0, high=10, size=(10,8)) # 生成一个指定shape的平均分布随机整数矩阵, [low, high]
np.random.random_integers(low=0, high=10, size=(8,8)) # 功能同上，[low, high)
np.random.normal(loc=0.0, scale=1.0, size=(5,5)) # 生成一个指定shape的正态分布随机数矩阵。loc=偏移，scale=标准差

np.linalg.norm(a, ord=None) # ord=None时求平方和的开方，ord!=None时就是ord次方再开ord方

np.save('aa', np.array(b, dtype=np.int32)) # 这里如果np.array()不指定dtype，则b的shape任意，但指定了则只能是元素短长的矩阵或张量
b.tolist() # 转list


# 关于值复制
# 在np矩阵中，要想交换0,1的位置，t=w[0],w[0]=w[1],w[1]=t 经实验是不行的，估计是引用复制导致的
a = b.copy() # 与py本变量一样
# 靠谱的方法
w = np.array([[1,2],[2,3]])
t = w[1].tolist()
w[1] = w[0].tolist()
w[0] = t

# 转one-hot,keras有utils.to_categorical
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

# 排序
list1 = [[4,3,2],[2,1,4]]
array=np.array(list1)
#array([[4, 3, 2],
#      [2, 1, 4]])
array.sort(axis=1)
#array([[2, 3, 4],
#      [1, 2, 4]])
# axis=1,说明是按照列进行排序，也就是说，每一行上的元素沿着列的方向实现了递增，
# 如[4, 3, 2]变为了[2, 3, 4]，[2, 1, 4]变为了[1, 2, 4]
array.sort(axis=0)
#array([[1, 2, 4],
#      [2, 3, 4]])
# axis=0,说明是按照行进行排序，也就是说，每一列上的元素沿着行的方向实现了递增，
# 如[2, 1]变为了[1, 2]，[3, 2]变为了[2, 3]
np.sort(array, axis=None) #array([1, 2, 2, 3, 4, 4])
# 当axis=None，将所有元素统一排序
li = np.array([1,2,3,4,5,6,7,8])
scores = np.array([8,7,6,5,4,3,2,1])
idxs = np.argsort(scores) # 获取排序后的索引, 不会改变原数组的顺序
a = li[idxs] #根据scores把li排序
# 想保持li排序选择score最大的一些
pos = np.array([False]*len(li), dtype=np.bool)
pos[idxs[-5:]] = True
a = li[pos] #
'''sklearn'''

from sklearn.feature_extraction.text import TfidfVectorizer

a = ['我 很 牛逼','我 超级 无敌 牛逼', '我 太 帅 了']
tfidf = TfidfVectorizer(analyzer=lambda x:x.split(' '))
m = tfidf.fit_transform(a)
for v in m:
    print(v.__dict__) # v是个对象，v.data是此句所有词的信息量，v.indices是每个词的idx，对应于下面get_feature_name给出的
print(tfidf.get_feature_names()) # 返回的是语料库所有词的list，顺序跟词的信息量无关，例如['我','是','帅哥']
print(tfidf.vocabulary_) # 与上面对应的dict，例{'我':0,'是':1,'帅哥':2}




