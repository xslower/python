import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"

# 总迭代次数500
max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256


# generate (model 1)
def build_generator(z_prior, c_gen):
    # w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
    # b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
    # h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
    # w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
    # b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    # h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    # w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
    # b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
    # h3 = tf.matmul(h2, w3) + b3
    # x_generate = tf.nn.tanh(h3)
    # g_params = [w1, b1, w2, b2, w3, b3]
    c_in = tf.one_hot(c_gen, depth=10, dtype=tf.float32)
    z_prior = tf.concat([z_prior, c_in], axis=-1)
    scope = 'generator'
    with tf.variable_scope(scope):
        # l1y = tf.layers.dense(y_in, units=)
        l1 = tf.layers.dense(z_prior, units=h1_size, activation=tf.nn.relu)
        l2 = tf.layers.dense(l1, units=h2_size, activation=tf.nn.relu)
        x_generate = tf.layers.dense(l2, units=img_size, activation=tf.nn.tanh)
    g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return x_generate, g_params


# discriminator (model 2)
def build_discriminator(x_data, c_data, x_gen, c_gen, keep_prob):
    # c_data = tf.one_hot(c_data, depth=10, dtype=tf.float32)
    x_data = tf.concat([x_data, c_data], axis=-1)
    c_gen = tf.one_hot(c_gen, depth=10, dtype=tf.float32)
    x_gen = tf.concat([x_gen, c_gen], axis=-1)
    x_in = tf.concat([x_data, x_gen], 0)
    # w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
    # b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
    # h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
    # w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
    # b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
    # h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
    # w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    # b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    # h3 = tf.matmul(h2, w3) + b3
    # y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    # y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    # d_params = [w1, b1, w2, b2, w3, b3]
    scope = 'discriminator'
    with tf.variable_scope(scope):
        l1 = tf.layers.dense(x_in, units=h2_size, activation=tf.nn.relu)
        l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
        l2 = tf.layers.dense(l1, units=h1_size, activation=tf.nn.relu)
        l2 = tf.nn.dropout(l2, keep_prob)
        y = tf.layers.dense(l2, units=1, activation=tf.nn.sigmoid)
        c_data = tf.slice(y, [0, 0], [batch_size, -1])
        y_generated = tf.slice(y, [batch_size, 0], [-1, -1])
    d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return c_data, y_generated, d_params


# 
def show_result(batch_res, fname, grid_size = (8, 8), grid_pad = 5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    # load data（mnist手写数据集）
    mnist = input_data.read_data_sets('data', one_hot=True)

    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    c_data = tf.placeholder(tf.float32, [batch_size, 10])
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    c_gen = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # 创建生成模型
    x_gen, g_params = build_generator(z_prior, c_gen)
    # 创建判别模型
    y_data, y_generated, d_params = build_discriminator(x_data, c_data, x_gen, c_gen, keep_prob)

    # 损失函数的设置
    # 这里用1-,就是让d网默认认为gen出来的都是错的，所以让网络学习如何把gen判别为0
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    # 这是让d网尽量把gen判别为正确的
    g_loss = - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size))
    c_sample_val = np.zeros((batch_size))
    for i in range(batch_size):
        c_sample_val[i] = i % 10

    steps = 60000 / batch_size
    for i in range(sess.run(global_step), max_epoch):
        for j in np.arange(steps):
            #         for j in range(steps):
            print("epoch:%s, iter:%s" % (i, j))
            # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
            x_value, c_value = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            z_sample = np.random.normal(0, 1, size=(batch_size, z_size))
            c_sample = np.random.randint(0, 9, size=(batch_size))
            # 执行生成
            fd = {x_data: x_value, c_data: c_value, z_prior: z_sample, c_gen: c_sample, keep_prob: 0.7}
            sess.run(d_trainer, feed_dict=fd)
            # 执行判别
            if j % 1 == 0:
                sess.run(g_trainer, feed_dict=fd)
        sess.run(tf.assign(global_step, i + 1))
        if i % 10 == 0:
            x_gen_val = sess.run(x_gen, feed_dict={z_prior: z_sample_val, c_gen: c_sample_val})
            show_result(x_gen_val, "output/sample{0}.jpg".format(i))
            z_rand_val = np.random.normal(0, 1, size=(batch_size, z_size))
            x_gen_val = sess.run(x_gen, feed_dict={z_prior: z_rand_val, c_gen: c_sample_val})
            show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
            saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()
