"""
keras相关api
"""
'''tf内的keras封装'''
import tensorflow as tf
import keras as kr

x = tf.Variable()

""" tensorflow.keras """

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

from keras.layers import *
from keras.models import Model,Sequential
from keras import optimizers
from keras.preprocessing import sequence
# 标准调用
model = Sequential([Dense(100, input_shape=(784,))]) # 第一层需要指定输入的数据shape
model.add(Embedding(input_dim=784, output_dim=50)) # 把词idx转为相应的embedding向量
model.add(kr.layers.Bidirectional(kr.layers.CuDNNGRU(units=64)))
model.add(kr.layers.Dense(1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y1)
# 使用多分类时，标签需要使用one-hot向量式
from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(y1, num_classes=3)

# 函数式调用
x = sequence.pad_sequences(x, maxlen=28) # 给不一样长的系列填充0向量
kr.layers.Input(shape=[4,3], tensor=None) # 生成一个placeholder，tensor非None时则生成相应张量
inputs = Input(shape=(784,))
x = Masking(mask_value=0.)(inputs) # 跳过值=0的步
x = Dense(64, activation='relu')(inputs) # 这个全链接层适应力较强，如果input大于2维，内部则会reshape后计算，之后再reshape回来
x = Permute([2,1])(x) # =tf.transpose(x, [0,2,1]) 交换第一维和第二维数据
predictions = Dense(1, activation='softmax')(x)
predictions = Reshape([])(predictions) # reshape可以用来降维=sqeeze

model = Model(inputs=inputs, outputs=predictions)
# 后面的编译、训练、预测等都相同

sgd = optimizers.Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mse'])
early = kr.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto') # 训练时的回调
chkp = kr.callbacks.ModelCheckpoint(filepath='data/', monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit(x, y1, callbacks=[early, chkp], validation_split=0.05)
model.predict(x)
model.save(filepath='/abc')
model = kr.models.load_model('/abc', custom_objects={'CuDNNGRU2': CuDNNGRU}) # 有自定义的类或方法需要手动传进去
# 多输入
# 可以定义多个输入holder
x2 = Input()
model = Model(inputs=[x,x2], outputs=y1)
# 多输出
# construct model
main_input = Input((100,), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
aux_out = Dense(1, activation='sigmoid', name='aux_out')(lstm_out)

aux_input = Input((5,), name='aux_input')
x = kr.layers.concatenate([lstm_out, aux_input])
x = Dense(64, activation='relu')(x)
main_out = Dense(1, activation='sigmoid', name='main_out')(x)

model = Model(inputs=[main_input, aux_input], outputs=[main_out, aux_out])
model.compile(optimizer='rmsprop', loss={'main_out': 'binary_crossentropy', 'aux_out': 'mean_absolute_error'}, loss_weights={'main_out': 1., 'aux_out': 0.3})

# train
model.fit(x={'main_input': x, 'aux_input': x2}, y={'main_out': y1, 'aux_out': y2}, batch_size=32, epochs=10, verbose=1)
score = model.evaluate(x={'main_input': x, 'aux_input': x2}, y={'main_out': y1, 'aux_out': y2}, batch_size=10, verbose=1)
print(score)

loss = model.train_on_batch(x=x2[:100], y=y2[:100]) # 适用于强化学习

# 关于权重的获取和设置
# 必须使用model.layers或model.get_layer()获取对应的层进行get_weights/set_weights。不然都是在计算图之外，影响不了模型的计算。
for ly in model.layers: # 可以获取model的所有层
    if ly.name == 'emb':
        pass
ly = model.get_layer(name='emb')
w = ly.get_weights()[0] # 0是weights，1是bias
ly.set_weights([w]) #

'''
mean_squared_error或mse
mean_absolute_error或mae
mean_absolute_percentage_error或mape
mean_squared_logarithmic_error或msle
squared_hinge
hinge
binary_crossentropy（亦称作对数损失，logloss）
categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：np.expand_dims(y,-1)
kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.
poisson：即(predictions - targets * log(predictions))的均值
cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数

metrics:
mse
mae
binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率
categorical_accuracy:对多分类问题,计算再所有预测值上的平均正确率
sparse_categorical_accuracy:与categorical_accuracy相同,在对稀疏的目标值预测时有用
top_k_categorical_accracy: 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
mean_squared_error:计算预测值与真值的均方差
mean_absolute_error:计算预测值与真值的平均绝对误差
mean_absolute_percentage_error:计算预测值与真值的平均绝对误差率
mean_squared_logarithmic_error:计算预测值与真值的平均指数误差
hinge:计算预测值与真值的hinge loss
squared_hinge:计算预测值与真值的平方hinge loss
categorical_crossentropy:计算预测值与真值的多类交叉熵(输入值为二值矩阵,而不是向量)
sparse_categorical_crossentropy:与多类交叉熵相同,适用于稀疏情况
binary_crossentropy:计算预测值与真值的交叉熵
poisson:计算预测值与真值的泊松函数值
cosine_proximity:计算预测值与真值的余弦相似性
matthews_correlation:计算预测值与真值的马氏距离
precision：计算精确度，注意percision跟accuracy是不同的。percision用于评价多标签分类中有多少个选中的项是正确的
recall：召回率，计算多标签分类中有多少正确的项被选中
fbeta_score:计算F值,即召回率与准确率的加权调和平均,该函数在多标签分类(一个样本有多个标签)时有用,如果只使用准确率作为度量,模型只要把所有输入分类为"所有类别"就可以获得完美的准确率,为了避免这种情况,度量指标应该对错误的选择进行惩罚. F-beta分值(0到1之间)通过准确率和召回率的加权调和平均来更好的度量.当beta为1时,该指标等价于F-measure,beta<1时,模型选对正确的标签更加重要,而beta>1时,模型对选错标签有更大的惩罚.
fmeasure：计算f-measure，即percision和recall的调和平均
'''

"""关于keras内部运行状况的监控，非常困难。例如要监控loss是如何计算的，callbacks没有办法"""