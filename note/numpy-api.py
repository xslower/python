import numpy as np

# 貌似是批装数组的
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b)) # array([1, 2, 3, 2, 3, 4])
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.hstack((a,b)) # array([[1, 2],[2, 3],[3, 4]])

# 生成一个指定范围的数组
np.arange(3,7) #array([3, 4, 5, 6])

np.amax(a, axis=1) # 求数组中最大的值，不带axis则flatten后求，带上axis则求某一维最大值







