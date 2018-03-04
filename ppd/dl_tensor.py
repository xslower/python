# coding=utf-8
import sys

sys.path.append('../lib')
from header import *
import tensorflow as tf
import tf_framework as fw
import pickle
import svc
from sklearn import preprocessing


def train_test_split(dx, dy):
    # x = np.array(dx)
    # y = np.array(dy)
    x = dx
    y = dy
    pos = int(len(x) / 10 * 8)
    return x[:pos], x[pos:], y[:pos], y[pos:]


def reshape_y(y):
    reshaped = []
    for i in range(len(y)):
        row = [0, 0]
        row[y[i]] = 1
        # reshaped[i][y[i]] = 1
        reshaped.append(row)
    return reshaped


def shape_back(y):
    shaped = [0] * len(y)
    for i in range(len(y)):
        if y[i][1] == 1:
            shaped[i] = 1
    return shaped


def infer(x, regular = None):
    conv1 = fw.conv2d_layer(x, [1, 3, None, 16], 1)
    pool1 = fw.pool_layer(conv1, [1, 2])

    layer1 = fw.full_layer(pool1, [None, 256], 3, regular)
    out = fw.full_layer(layer1, [None, 2], 4, regular)
    return out


class net_define:
    reshape = True
    output_num = 2
    batch_size = 200
    learning_rate_base = 0.8
    learning_rate_decay = 0.99
    regular_rate = 0.0001
    training_steps = 5001
    moving_avaerage_decay = 0.99

    save_path = 'data/model/'
    model_name = 'model.ckpt'


def train(dx, dy):
    scaler = preprocessing.StandardScaler()
    scaler.fit(dx)
    dx = scaler.transform(dx)
    x_train, x_test, y_train, y_test = train_test_split(dx, dy)
    # print(y_train)
    y_train = reshape_y(y_train)
    # print(y_train)
    # exit()
    nnm = fw.NnModel(inference=infer, input_shape=[1, len(dx[0]), 1])
    nnm.set_attr(net_define)
    nnm.train((x_train, y_train), {1: 200})
    # nnm.train((x_train, y_train))
    y_pred = nnm.predict(x_test)
    # print(y_pred)
    svc.precise(y_test, y_pred)


def test():
    a = [[0, 1], [0, 1], [0, 1]]
    b = [[0, 1], [0.5, 0, 5], [0, 0]]
    a = [0, 1]
    b = [0.0, 1.0]
    tb = tf.Variable(b)
    sm_b = tf.nn.softmax(b)
    sm_b2 = tf.clip_by_value(sm_b, 1e-10, 1.0)
    log_b = tf.log(sm_b2)

    loss = (a * tf.log(tf.clip_by_value(tf.nn.softmax(b), 1e-10, 1.0)))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        v = sess.run([loss, log_b, sm_b2, sm_b])
        tv = tb.eval()
        print(v, type(tv))


def kk(x, y, x_test, y_test):
    batch_size = 100
    learning_rate = 0.01
    training_epochs = 10

    x_holder = tf.placeholder(tf.float32, shape=(None, len(x[0])))
    labels = tf.placeholder(tf.int32, shape=(None,))

    h = tf.keras.layers.Dense(10, activation='relu')(x_holder)
    h = tf.keras.layers.Dense(8, activation='relu')(h)
    h = tf.keras.layers.Dense(6, activation='relu')(h)
    y_pred = tf.keras.layers.Dense(3, activation='relu')(h)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels, logits=y_pred, pos_weight=[1, 2, 4]))
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))
    train_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epoch_num = len(x) // batch_size
        for epoch in range(training_epochs):
            for i in range(epoch_num):
                start = (i * batch_size)
                end = start + batch_size
                batch_x, batch_y = x[start:end], y[start:end]
                sess.run(train_optim, feed_dict={x_holder: batch_x, labels: batch_y})

                acc_pred = tf.keras.metrics.categorical_accuracy(labels, y_pred)
                pred = sess.run(acc_pred, feed_dict={labels: y_test, x_holder: x_test})
                svc.precise(y_test, pred, 3)
                print('accuracy: %.3f' % (sum(pred) / len(x_test)))


def autoEncoder(x, x_test, y_test):
    import matplotlib.pyplot as plt

    learning_rate = 0.01
    training_epochs = 10
    batch_size = 100
    display_step = 1
    examples_to_show = 10
    x_dim = len(x[0])
    x_holder = tf.placeholder(tf.float32, [None, x_dim])
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # 构建编码器
    def encoder(input):
        l1 = tf.keras.layers.Dense(8, activation='relu')(input)
        l2 = tf.keras.layers.Dense(2, activation='relu')(l1)
        return l2

    # def encoder(x):
    #     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    #     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    #     return layer_2
    # 构建解码器
    def decoder(input):
        l1 = tf.keras.layers.Dense(8, activation='relu')(input)
        l2 = tf.keras.layers.Dense(x_dim, activation='relu')(l1)
        return l2

    #     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    #     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    #     return layer_2

    # 构建模型
    encoded = encoder(x_holder)
    y_pred = decoder(encoded)

    # 预测
    y_true = x_holder

    # 定义代价函数和优化器
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # 最小二乘法
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    min_cost = 100
    bt_step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # sess.run(init)
        # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
        total_batch = len(x) // batch_size  # 总批数
        for epoch in range(training_epochs):
            for i in range(total_batch):
                start = (i * batch_size)
                end = start + batch_size
                batch_xs = x[start:end]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cv = sess.run([optimizer, cost], feed_dict={x_holder: batch_xs})
                if i % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", cv)
                if cv < min_cost:
                    min_cost = cv
                    bt_step = 0
                else:
                    bt_step +=1
            # if bt_step > 10:
            #     break
        print("Optimization Finished!")

        edd = sess.run(encoded, feed_dict={x_holder: x_test})
        plt.scatter(edd[:, 0], edd[:, 1], c=y_test)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    dx, dy = svc.init_data()
    # svc.save_file(dx, dy)
    # dx, dy = svc.load_file()
    x_train, x_test, y_train, y_test = svc.train_test_split(dx, dy, 6)
    kk(x_train, y_train, x_test, y_test)
