"""
keras相关api
"""
'''tf内的keras封装'''
import tensorflow as tf

x = tf.Variable()

""" tensorflow.keras """

kr.layers.Input(shape=[4,3], tensor=None) # 生成一个placeholder，tensor非None时则生成相应张量

dnn = tf.keras.layers.Dense(units=100, activation='relu')  # 全链接层，units是隐藏单元数
# 虽然tf自己也封装了很多的方法，但显然不如keras封装的易用易理解。
y1 = dnn(x)
kr.layers.Dense(units=100, activation=None)

cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, input_shape=(128, 10), padding='valid')  # keras封装的卷积方法。filters=输出单元数=全链接层的隐藏单元数；kernel_size就是卷积的窗口大小，1维的只对输入元素的第一维(128)进行卷积过滤，第2维(10)与2D的channels层一样不过滤；strides=移动步长；padding='same'or'valid'；第一层需要input_shape
y2 = cnn(x)

cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first', input_shape=(1, 128, 128))  # 基本同上，只是2维的,channels_first输入为channel维在前[batches, channels, rows, cols]；第一层也需要input_shape，要与channels_first匹配
y3 = cnn(x)

pool = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')  # pool_size是窗口大小，strides步长
y4 = pool(x)

tf.keras.layers.MaxPooling2D()  # 与Conv2D类似

'''和原生keras'''

import keras as kr
from keras.layers import Input,Dense,Embedding,Permute
from keras.models import Model,Sequential

# 标准调用
model = Sequential([Dense(100, input_dim=(784,))]) # 第一层需要指定输入的数据shape
model.add(Embedding(input_dim=784, output_dim=50)) # 把词idx转为相应的embedding向量
model.add(kr.layers.Bidirectional(kr.layers.CuDNNGRU(units=64)))
model.add(kr.layers.Dense(1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y1)

# 函数式调用
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs) # 这个全链接层适应力较强，如果input大于2维，内部则会reshape后计算，之后再reshape回来
x = Permute([2,1])(x) # =tf.transpose(x, [0,2,1]) 交换第一维和第二维数据
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y1)  # starts training

