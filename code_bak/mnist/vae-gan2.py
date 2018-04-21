import os
import numpy as np
import prettytensor as pt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from skimage.io import imsave
# import IPython.display
import math
import tqdm  # making loops prettier
import h5py  # for reading our dataset

# import ipywidgets as widgets
# from ipywidgets import interact, interactive, fixed

# %matplotlib inline

# dim1 = 64  # first dimension of input data
# dim2 = 64  # second dimension of input data
# dim3 = 3  # third dimension of input data (colors)
dim1 = 28
dim2 = 28
dim_in_channels = 1
batch_size = 32  # size of batches to use (per GPU)
hidden_size = 32  # size of hidden (z) layer to use
num_examples = 60000  # how many examples are in your training set
num_epochs = 10000  # number of epochs to run
### we can train our different networks  with different learning rates if we want to
e_learning_rate = 1e-3
g_learning_rate = 1e-3
d_learning_rate = 1e-3
num_expressed = 24
n_class = 10
reshaped_dim = [-1, 7, 7, dim_in_channels]

# grab the faces back out after we've flattened them
def create_image(im):
    return np.reshape(im, (dim1, dim2, dim_in_channels))

def lrelu(x, alpha = 0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X):
    '''Create encoder network.
    Args:
        x: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
            # The transformation is parametrized and can be learned.
            # returns network output, mean, setd
    '''
    lay_end = (pt.wrap(X).reshape([batch_size, dim1, dim2, dim_in_channels]).conv2d(4, 64, stride=2).conv2d(4, 64, stride=2).conv2d(4, 64, stride=1).flatten())
    z_mean = lay_end.fully_connected(hidden_size, activation_fn=None)
    z_log_sigma_sq = lay_end.fully_connected(hidden_size, activation_fn=None)
    return z_mean, z_log_sigma_sq


def generator(Z):
    '''Create generator network.
        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        x: a batch of vectors to decode
    Returns:
        A tensor that expresses the generator network
    '''
    return (pt.wrap(Z).fully_connected(49).reshape([batch_size, 7, 7, 1]).deconv2d(4, 64, stride=2).deconv2d(4, 64, stride=1).deconv2d(4, 32, stride=1).flatten().fully_connected(28 * 28, activation_fn=tf.sigmoid))


def decoder(z_gen, keep_prob = 1.0, z_label = None):
    if z_label is not None:
        z_label = tf.one_hot(z_label, depth=n_class, dtype=tf.float32)
        z_gen = tf.concat([z_gen, z_label], axis=-1)
    with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(z_gen, units=num_expressed, activation=lrelu)
        x = tf.layers.dense(x, units=num_expressed * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img


def discriminator(D_I):
    ''' A encodes
    Create a network that discriminates between images from a dataset and
    generated ones.
    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    descrim_conv = (pt.wrap(D_I).  # This is what we're descriminating
        reshape([batch_size, dim1, dim2, dim_in_channels]).conv2d(4, 32, stride=1).conv2d(4, 64, stride=2).conv2d(4, 64, stride=2).flatten())
    lth_layer = descrim_conv.fully_connected(1024, activation_fn=tf.nn.elu)  # this is the lth layer
    D = lth_layer.fully_connected(1, activation_fn=tf.nn.sigmoid)  # this is the actual discrimination
    return D, lth_layer


def inference(x):
    """
    Run the models. Called inference because it does the same thing as tensorflow's cifar tutorial
    """
    z_p = tf.random_normal((batch_size, hidden_size), 0, 1)  # normal dist for GAN
    eps = tf.random_normal((batch_size, hidden_size), 0, 1)  # normal dist for VAE

    with pt.defaults_scope(activation_fn=tf.nn.elu, batch_normalize=True, learned_moments_update_rate=0.0003, variance_epsilon=0.001, scale_after_normalization=True):
        with tf.variable_scope("enc"):
            z_x_mean, z_x_log_sigma_sq = encoder(x)  # get z from the input
        with tf.variable_scope("gen"):
            z_x = tf.add(z_x_mean, tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps))  # grab our actual z
            # x_tilde = generator(z_x)
            x_tilde = decoder(z_x)
        with tf.variable_scope("dis"):
            _, l_x_tilde = discriminator(x_tilde)
        with tf.variable_scope("gen", reuse=True):
            # x_p = generator(z_p)
            x_p = decoder(z_p)
        with tf.variable_scope("dis", reuse=True):
            d_x, l_x = discriminator(x)  # positive examples
        with tf.variable_scope("dis", reuse=True):
            d_x_p, _ = discriminator(x_p)
        return z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p


def loss(x, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, dim1, dim2, dim3):
    """
    Loss functions for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    ### We don't actually use SSE (MSE) loss for anything (but maybe pretraining)
    # SSE_loss = tf.reduce_mean(tf.square(x - x_tilde))  # This is what a normal VAE uses
    SSE_loss = 0
    # We clip gradients of KL divergence to prevent NANs
    KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0) - tf.square(tf.clip_by_value(z_x_mean, -10.0, 10.0)) - tf.exp(tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0)), 1)) / dim1 / dim2 / dim3
    # Discriminator Loss
    D_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(d_x, 1e-5, 1.0)) + tf.log(tf.clip_by_value(1.0 - d_x_p, 1e-5, 1.0))))
    # Generator Loss
    G_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(d_x_p, 1e-5, 1.0))))  # +
    # tf.log(tf.clip_by_value(1.0 - d_x,1e-5,1.0))))
    # Lth Layer Loss - the 'learned similarity measure'
    LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde)) / dim1 / dim2 / dim3
    return SSE_loss, KL_loss, D_loss, G_loss, LL_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.


    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


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


def sigmoid(x, shift, mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + math.exp(-(x + shift) * mult))


graph = tf.Graph()
# Make lists to save the losses to
# You should probably just be using tensorboard to do any visualization(or just use tensorboard...)
G_loss_list = []
D_loss_list = []
SSE_loss_list = []
KL_loss_list = []
LL_loss_list = []
dxp_list = []
dx_list = []

with graph.as_default():
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count number of train calls
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # different optimizers are needed for different learning rates (using the same learning rate seems to work fine though)
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])
    lr_E = tf.placeholder(tf.float32, shape=[])
    opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
    opt_G = tf.train.AdamOptimizer(lr_G, epsilon=1.0)
    opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)

    # These are the lists of gradients for each tower
    tower_grads_e = []
    tower_grads_g = []
    tower_grads_d = []

    x_in = tf.placeholder(tf.float32, [batch_size, dim1 * dim2 * dim_in_channels])
    KL_param = tf.placeholder(tf.float32)
    LL_param = tf.placeholder(tf.float32)
    G_param = tf.placeholder(tf.float32)  # Construct the model
    z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p = inference(x_in)

    # Calculate the loss for this tower
    SSE_loss, KL_loss, D_loss, G_loss, LL_loss = loss(x_in, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, dim1, dim2, dim_in_channels)

    # specify loss to parameters
    params = tf.trainable_variables()
    E_params = [i for i in params if 'enc' in i.name]
    G_params = [i for i in params if 'gen' in i.name]
    D_params = [i for i in params if 'dis' in i.name]

    # Calculate the losses specific to encoder, generator, decoder
    L_e = tf.clip_by_value(KL_loss * KL_param + LL_loss, -100, 100)
    L_g = tf.clip_by_value(LL_loss * LL_param + G_loss * G_param, -100, 100)
    L_d = tf.clip_by_value(D_loss, -100, 100)

    # Reuse variables for the next tower.
    tf.get_variable_scope().reuse_variables()

    # Calculate the gradients for the batch of data on this CIFAR tower.
    grads_e = opt_E.compute_gradients(L_e, var_list=E_params)
    grads_g = opt_G.compute_gradients(L_g, var_list=G_params)
    grads_d = opt_D.compute_gradients(L_d, var_list=D_params)

    # Keep track of the gradients across all towers.
    tower_grads_e.append(grads_e)
    tower_grads_g.append(grads_g)
    tower_grads_d.append(grads_d)

    # Average the gradients
    # grads_e = average_gradients(tower_grads_e)
    # grads_g = average_gradients(tower_grads_g)
    # grads_d = average_gradients(tower_grads_d)

    # apply the gradients with our optimizers
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    train_G = opt_G.apply_gradients(grads_g, global_step=global_step)
    train_D = opt_D.apply_gradients(grads_d, global_step=global_step)

    # Start the Session
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()  # initialize network saver
    sess = tf.InteractiveSession(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(init)

    epoch = 0
    d_real = 0
    d_fake = 0
    mnist = input_data.read_data_sets('data', one_hot=True)

    while epoch < num_epochs:
        # balence gen and descrim
        e_current_lr = e_learning_rate * sigmoid(np.mean(d_real), -.5, 15)
        g_current_lr = g_learning_rate * sigmoid(np.mean(d_real), -.5, 15)
        d_current_lr = d_learning_rate * sigmoid(np.mean(d_fake), -.5, 15)
        next_batches, _ = mnist.train.next_batch(batch_size)
        fd = {lr_E: e_current_lr, lr_G: g_current_lr, lr_D: d_current_lr, x_in: next_batches, KL_param: 1, G_param: 1, LL_param: 1}
        _, _, _, D_err, G_err, KL_err, SSE_err, LL_err, d_fake, d_real = sess.run([train_E, train_G, train_D, D_loss, G_loss, KL_loss, SSE_loss, LL_loss, d_x_p, d_x,

        ], fd)
        # KL_err= SSE_err= LL_err = 1
        # Save our lists
        dxp_list.append(d_fake)
        dx_list.append(d_real)
        G_loss_list.append(G_err)
        D_loss_list.append(D_err)
        KL_loss_list.append(KL_err)
        SSE_loss_list.append(SSE_err)
        LL_loss_list.append(LL_err)

        if epoch % 100 == 90:
            imgs = sess.run(x_p, feed_dict=fd)
            show_result(imgs, 'output/sample{0}.jpg'.format(epoch))
            rz = np.random.normal(0, 1, (batch_size, hidden_size))
            rx = sess.run((x_tilde), {z_x: rz})
            show_result(rx, 'output/rx{0}.jpg'.format(epoch))
        epoch += 1

