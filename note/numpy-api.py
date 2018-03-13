import numpy as np

# 貌似是批装数组的
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b)) # array([1, 2, 3, 2, 3, 4])
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.hstack((a,b)) # array([[1, 2],[2, 3],[3, 4]])

# 生成一个指定范围的数组
np.arange(3,7) # array([3, 4, 5, 6])
np.empty([2, 2], dtype=np.int32) # 生成一个指定形状和类型的张量，并以随机值填充
np.amax(a, axis=1) #=np.max() 求数组中最大的值，不带axis则flatten后求，带上axis则求某一维的最大值，例如np.max(x3d, axis=0) 则x3d[max][i][j] =x2d
np.argmax(a, axis=1) # 上面是求最大值，这个是取最大值所在的index

new_a = a[np.newaxis, :] # 在最前面增加一个纬度。new_a=[[1,2,3]]
new_b = a[:, np.newaxis] # 最后面增加一维，=[[1],[2],[3]]






