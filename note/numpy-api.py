import numpy as np

# 貌似是拼装数组的
a = np.array((1, 2, 3))
b = np.array((2, 3, 4))
np.hstack((a, b))  # array([1, 2, 3, 2, 3, 4])
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.hstack((a, b))  # array([[1, 2],[2, 3],[3, 4]])
np.append(a, [4], axis=None)  # 前后两个参数的shape必须相同； axis=None会把前后两个参数flatten掉，然后连在一起；axis=i会把第i维连接，类似tf.concat
np.concatenate([a, b], axis=0) # 与上面相同

np.insert(a, 1, [5])  # 在位置1之前插入5

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
new_b = a[:, np.newaxis]  # 最后面增加一维，=[[1],[2],[3]]
b = np.squeeze(a, axis=None) # 把len=1的维度消去，[[0],[1]]=[0,1], axis指定纬度后只消指定维,指定维度的len必须=1

'''概率'''
np.random.randint(low=0, high=10, size=(10,8)) # 生成一个指定shape的平均分布随机整数矩阵, [low, high]
np.random.random_integers(low=0, high=10, size=(8,8)) # 功能同上，[low, high)
np.random.normal(loc=0.0, scale=1.0, size=(5,5)) # 生成一个指定shape的正态分布随机数矩阵。loc=偏移，scale=标准差

np.linalg.norm(a, ord=None) # ord=None时求平方和的开方，ord!=None时就是ord次方再开ord方
