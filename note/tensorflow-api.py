"""
tensorflow api文档理解+记录:
ps: 感觉还是把代码相关的东西写在代码里方便
"""

import tensorflow as tf

# 逻辑op：
x = tf.constant([1, 2])
y = tf.constant([2, 3])
tf.greater(x, y)
tf.less(x, y)
tf.logical_and(x, y)

tf.where(tf.greater(x, y), x, y)
# tf.while_loop()

"""
数据处理：
"""
x = tensor = tf.Variable()
y = y_label = tf.Variable()
inputs = tf.Variable()
weights = tf.Variable()
bias = tf.Variable()

tf.assign(x, y) # 把y赋值给x
tf.multiply(x, y) # x与y相乘；矩阵相乘时，只能有一个为矩阵，按元素相乘
tf.matmul(x, y) # 矩阵相乘，x与y的shape必须匹配
tf.argmax(y_label, axis=1)  # y_label是2维的，axis是操作哪一维。[[0,1,0],[1,0,0],[0,0,1]->[1,0,2]
tf.expand_dims(tensor, 1)  # 把一个tensor扩充一维，例如[1,2,3]->[[1],[2],[3]] -> [[[1]],[[2]],[[3]]]。后面的参数不能随意取
tf.squeeze(x) # 去掉长度=1的那些纬度，例：tf.shape(t) = [1,2,1,3,1,1] tf.shape(tf.squeeze(t))=[2,3]
tf.reduce_mean(x, axis=0) # 延着指定纬度计算平均值x[mean][i][j]，不指定纬度则计算全部值的平均值

tf.range(0, 10, 0)  # 跟python 的range一样生成一个0到9的list
tf.concat([x, y], 1)  # 把两个同rank的tensor合并到一起，[[0],[1]] + [[2],[3]] = [[0,2],[1,3]]
tf.slice(tensor, begin=0, size=2)  #
tf.strided_slice(tensor, [1], [6])  # = labels[1:6]
tf.nn.lrn(input)  # 卷积层之后使用的一种数据归一化方法，可以把大的值变得相对更大，小值相对更小，防止在层数增加时权重衰减
tf.nn.xw_plus_b(x, weights, bias)  # 计算x * w + b
tf.transpose(tensor, perm=[1, 2, 0])  # 例如：[depth, height, width] to [height, width, depth]。perm指定哪些纬度交换，[1, 2, 0]=第一维成为最后一维二三维成为一二维。矩阵转秩
batch_x, batch_y = tf.train.batch([x, y], batch_size=10)  # 貌似专门用来打包输入数据的，一般x,y为两个queue，此方法会把打包为每次输出batch_size大小的x,y

tf.layers.batch_normalization(tensor)  # 貌似是在每层激活函数之前，给增加每一维增加一个sub学习率，来归一化数据，让其分布保持不变，以加速训练速度

with tf.control_dependencies([x, y]):  # 梳理op运行关系的，必须先运行一些op，才能运行后面的。经常用于先计算summary
    pass

from tensorflow.python.ops import array_ops

array_ops.split(tensor, num_or_size_splits=2, axis=0)  # 把tensor延axis方向上切为小块的tensor，split是一个scalar则平均分，否则每片多少由split指定
array_ops.concat(tensor, axis=1)  # 把tensor里的张量延方向合并。例：([[1,2,3],[4,5,6]],[[3,2,1],[6,5,4]])，延0合并=[[1,2,3],[4,5,6],[3,2,1],[6,5,4]]，延1合并=[[1,2,3,3,2,1],[4,5,6,6,5,4]]。tf里经常延1合并，是在每一个batch上合并成一个长向量。

from tensorflow.python.util import nest

nest.flatten(x)  # 内部用递归的方式把N维输入，转为1维的list输出
nest.is_sequence(x)  # isinstance(x, collections.Sequence) 1维以上的数组都=True

'''概率'''
mean = 0.0
stddev = 1.0
norm = tf.distributions.Normal(loc=mean, scale=stddev) # 正态分布，loc=平均值，scale=标准差
norm.sample(sample_shape=1) # 基于此分布生成一个目标shape的样本集
norm.log_prob(value=1.1) # 貌似是计算此分布某点的概率密度
norm.entropy() # Shannon entropy

"""
训练、建模：
"""
# 损失函数：
tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=y)
# 这个主要是面向2分类的

tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y)
# labels和logits必须是相同的shape

tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_label, 1), logits=y)
# 与上面不同。估计是为了维数多时方便接收数据的。
# labels必须比logits少一维。一般为[0,1,2,1]每一位表示一个样本的类别
# logits则是计算出来的输出。一般为[[1,0,0],[0,1,0],[0,0,1],[0,1,0]]。

class _LoggerHook:
    pass


tf.train.MonitoredTrainingSession(checkpoint_dir='dir', hooks=[_LoggerHook()], config=tf.ConfigProto(log_device_placement=True))
# 根据参数创建一个MonitoredSession对象。其是个很好用的工具，可以自行把summary和checkpoing记录到指定的目录下，而且可以自行维护所有线程。
# 并且，这个session貌似会恢复上次的训练的结果后继续训练
# tf.layers  # 这是一个高层api，提供了一些网络结构定义的方法
# tf.estimator  # 这是一个高层api，提供了训练评估等方法的封装 # todo 有空研究下，还有keras可以减少很多代码。
tf.nn.embedding_lookup(tensor, ids=[1, 2, 3])  # 这是根据ids里的索引idx，获取params里相应索引的值，
tf.nn.nce_loss(weights=x, biases=y, labels=y, inputs=x, num_sampled=10, num_classes=50000)  # cbow和skip-gram训练时打包的一个损失函数(目标函数)，详见word2vec理解

""" cnn相关："""
# 卷积层
tf.nn.conv2d(inputs, weights, strides=[1, 2, 2, 1], padding='SAME')  # stride=在input各个纬度上的步长，[不同的样本, 宽, 高, 深]，第一纬为不同的样本，步长只能=1，最后一纬=深度，也只能=1
# 池化层
tf.nn.max_pool(inputs, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')  # 其它参数同上，ksize的格式很奇葩与stride相同[1,2,2,1]。[不同的样本, 宽, 高, 深]

"""rnn相关："""
last_state = tf.Variable()
tf.nn.rnn_cell.RNNCell()  # rnn的抽象类

# 最基本的rnn 单元。其实内部就是一个全链接网络，只是输入合并input+last_state, 输出一样的output和new_state。
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10, activation=tf.nn.relu)  # num_units=单元数量=全链接层输出数量=input输入的纬度=state的长度
output1, new_state1 = cell(inputs=inputs, state=last_state)

# gru单元。内部实现与[https://www.jianshu.com/p/9dc9f41f0b29]里最后的gru图完全一样，只有在最后z和(1-z)反了一下，本质还是相同
cell = tf.nn.rnn_cell.GRUCell(num_units=10, activation=tf.nn.tanh, reuse=True)  # num_units同样=input/state的纬度
output2, new_state2 = cell(input=inputs, state=last_state)  # 用法基本与rnncell相同
tf.contrib.rnn.GRUBlockCell()  # 貌似与上面的基本相同

# 基本lstm单元。内部实现与[https://www.jianshu.com/p/9dc9f41f0b29]里主力讲解的lstm完全一样。
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10, state_is_tuple=True, activation=tf.nn.tanh)  # num_units同样=每个状态c/h和input的纬度
output3, new_state3 = cell(inputs, state=last_state)  # 这里state推荐是一个tuple，也就是两个状态(c,h)

# 高级lstm单元。
# use_peepholes=False时，跟BasicLSTMCell结构基本相同。区别可以用num_proj单独指定结果和状态m输出的纬度，但中间计算依然全部使用num_units维，所以最后需要单独为m和output做一个线性变换到num_proj维。
# use_peepholes=True时，就是在三次计算gate值时，都外加一个weight向量 X c，作用是啥等以后看论文吧
cell = tf.nn.rnn_cell.LSTMCell(num_units=10, use_peepholes=True, num_proj=20, proj_clip=1e-1, cell_clip=0.2)  # clip proj_clip是限制下最终m和out的值的上下限。cell_clip是限制输出的c的每个值的上下限
o4, ns4 = cell(inputs, last_state)
tf.contrib.rnn.LSTMBlockCell()  # 看代码+注释，貌似跟上面的实现基本相同，只是少了几个参数，例如不能指定num_proj。
# 发现contrib下面的实现经常与nn下的差不多。

# 多层rnn封装，把多个cell连接到一起，形成深层rnn
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3)  # 先定义好每层的cell传入
output4, all_state = multi_cell(inputs, (new_state1, new_state2, new_state3))  # 状态是并行的，只在对应的层中流动；input->output是穿过所有层

# cudnn封装，多层lstm，上面的可以手动指定每一层的cell具体实现，这里只能使用标准lstm
cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=3, num_units=90, input_size=95)  # input_size是每个x的纬度，一般跟num_units相同，不然估计需要做线性变换

# rnn打包。上面只是定义了网络，但是rnn调用时并不是直入直出的，而是按照顺序挨个输入Xi，同时输入X(i-1)的state，计算后输出Yi。最后把1-n的Yi打包在一起形成最终输出。
# 下面就是自动挨个调用rnn_cell的打包方法。
outputs1, final_states1 = tf.nn.static_rnn(cell, inputs, initial_state=last_state)  #
outputs2, final_states2 = tf.nn.dynamic_rnn(cell, inputs, initial_state=last_state)  # 跟static的区别貌似是接收的inputs形状不同，static接收的必须是相同batch_size的输入，而dynamic可以不同。

"""
tensorboard使用:
"""

tf.summary.scalar(name='', tensor=tensor)  # 收集一个标量的值
tf.summary.histogram(name='', values=tensor)  # 收集一个张量的值
# tf.nn.zero_fraction(tensor)  # 用于判断一个张量的稀松情况的
tf.summary.image(name='', tensor=tensor)  # 收集input图片用于查看input是否正确
