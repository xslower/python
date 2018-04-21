import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
import prettytensor as pt
# %matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

batch_size = 64
n_latent = 8
n_class = 10

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28], name='X')
Y = tf.placeholder(dtype=tf.int32, shape=[None], name='Y')

keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
Z_in = tf.placeholder(dtype=tf.float32, shape=[None, n_latent])
dec_in_channels = 1

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels // 2


def lrelu(x, alpha = 0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def vae_encoder(X_in, keep_prob = 1.0):
    activation = lrelu
    with tf.variable_scope("enc", reuse=None):
        x = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        # x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        # x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        # x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd


# def ae_encoder(X_in):
#     activation = lrelu
#     with tf.variable_scope("enc", reuse=None):
#         x = tf.reshape(X_in, shape=[-1, 28, 28, 1])
#         x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
#         x = tf.nn.dropout(x, keep_prob)
#         x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
#         x = tf.nn.dropout(x, keep_prob)
#         x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
#         x = tf.nn.dropout(x, keep_prob)
#         x = tf.contrib.layers.flatten(x)
#         z = tf.layers.dense(x, n_latent)
#         return z



def decoder(z_gen, keep_prob = 1.0, z_label = None):
    if z_label is not None:
        z_label = tf.one_hot(z_label, depth=n_class, dtype=tf.float32)
        z_gen = tf.concat([z_gen, z_label], axis=-1)
    with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(z_gen, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        # x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        # x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img


def dicriminator(x, keep_prob):
    act = lrelu
    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
        # x = tf.concat([x, x_gen], axis=0)
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=act)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=act)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=act)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.flatten(x)
        l = tf.layers.dense(x, units=256, activation=act)
        d = tf.layers.dense(l, units=1, activation=tf.nn.sigmoid)
        # y_real, y_gen = tf.split(x, 2, axis=0)

        return d, l


y_real, l_real = dicriminator(X_in, keep_prob)

z_gened, mn, sd = vae_encoder(X_in, keep_prob)
# z_gened = vae_encoder(X_in)
x_rebuild = decoder(z_gened)
y_rebuild, l_rebuild = dicriminator(x_rebuild, keep_prob)

x_gen = decoder(Z_in)
y_gen, _ = dicriminator(x_gen, keep_prob)

x_gen_flat = tf.reshape(x_rebuild, [-1, 28 * 28])
X_flat = tf.reshape(X_in, shape=[-1, 28 * 28])
img_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_gen_flat, X_flat), 1))
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
# loss = tf.reduce_mean(img_loss + latent_loss)
d_loss = -(tf.log(y_real) + tf.log(1 - y_gen))
g_loss = -(tf.log(y_gen))
ll_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(l_rebuild, l_real), 1))
# loss = img_loss + d_loss + g_loss
enc_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'enc')
dec_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dec')
dis_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dis')
optimizer = tf.train.AdamOptimizer(0.0003)
enc_trainer = optimizer.minimize(latent_loss, var_list=enc_para)
d_trainer = optimizer.minimize(d_loss, var_list=dis_para)
g_trainer = optimizer.minimize(g_loss, var_list=dec_para)
ll_trainer = optimizer.minimize(ll_loss, var_list=enc_para + dec_para)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def show_result(batch_res, fname, grid_size = (8, 8), grid_pad = 5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], 28, 28)) + 0.5
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


mnist = input_data.read_data_sets('data')
c_sample_val = np.zeros(batch_size)
for i in range(batch_size):
    c_sample_val[i] = i % 10

for i in range(3000):
    # batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    x_value, c_value = mnist.train.next_batch(batch_size)
    # sess.run(ae_trainer, feed_dict={X_in: x_value, Y: c_value, keep_prob: 0.8})
    z_samples = np.random.normal(0, 1, size=(batch_size, n_latent))
    fd = {X_in: x_value, Y: c_value, Z_in: z_samples, keep_prob: 0.8}
    sess.run([d_trainer], feed_dict=fd)
    sess.run(g_trainer, feed_dict=fd)

    if i % 500 == 499:
        # ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})
        # plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        # plt.show()
        # plt.imshow(d[0], cmap='gray')
        # plt.show()
        # print(i, ls, np.mean(i_ls), np.mean(d_ls))

        imgs = sess.run(x_gen, feed_dict={Z_in: z_samples, Y: c_sample_val, keep_prob: 1.0})
        show_result(imgs, 'output/sample{0}.jpg'.format(i))
        # imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

# for img in imgs:
#     plt.figure(figsize=(1, 1))
#     plt.axis('off')
#     plt.imshow(img, cmap='gray')
