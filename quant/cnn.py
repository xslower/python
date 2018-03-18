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


def infer(x, regular=None):
    tf.nn.
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


if __name__ == '__main__':
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(message)s')
    # dx, dy = svc.init_data()
    # svc.save_file(dx, dy)
    dx, dy = svc.load_file()
    train(dx, dy)
