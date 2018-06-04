"""
keras相关api
"""
'''tf内的keras封装''''''和原生keras'''
import tensorflow as tf
import keras as kr

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

kr.layers.Embedding(input_dim=1024, output_dim=50) # 把词idx转为相应的embedding向量
kr.layers.LSTM(units=64)
kr.layers.GRU()
kr.layers.Dropout()


