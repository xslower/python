# coding=utf-8
import sys

sys.path.append('../../lib')
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf_framework as fw
import time
from datetime import datetime

class conf:
    input_shape = [28, 28, 1]
    output_num = 10
    batch_size = 100
    # training_steps = 30001
    learning_rate_base = 0.4
    learning_rate_decay = 0.98

    max_steps = 20001
    moving_average_decay = 0.99
    epoch_size_for_train = 50000
    total_num = 50000
    log_frequency = 500
    data_dir = 'data/'
    train_dir = 'data/train/'
    eval_dir = 'data/eval/'
    model_name = 'm.ckpt'
    timeline_dir = 'data/timeline/'
    log_device_placement = False

# 定义了前向传播计算结果。1层隐藏层
def infer(x, is_train=True):
    c1 = fw.conv2d_layer(x, [5, 5, None, 16], 1)
    p1 = fw.pool_layer(c1)
    c2 = fw.conv2d_layer(p1, [5, 5, None, 32], 2)
    p2 = fw.pool_layer(c2)
    f1 = fw.full_layer(p2, [None, 256], 0.0001, 3)
    if is_train:
        f1 = tf.nn.dropout(f1, 0.5)
    y = fw.full_layer(f1, [None, 10], 0.0001, 4)
    return y


# def _train(mnist):
#     nnm = fw.NnModel(inference=infer)
#     nnm.set_attr(conf)
#     data = (mnist.train.images, mnist.train.labels)
#     print(len(data[0]))
#     nnm.train(data)
#     nnm.evaluate((mnist.test.images, mnist.test.labels))

def train(mnist):
    # global_step = tf.train.get_or_create_global_step()
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/cpu:0'):
        images, labels = mnist.train.images, mnist.train.labels
        images = np.reshape(images, [len(images), *conf.input_shape])
    # q = tf.FIFOQueue(capacity=55000,dtypes=tf.float32)
    # enqueue_op = q.enqueue_many(images)
    # qr = tf.train.QueueRunner(q, [enqueue_op])
    # tf.train.add_queue_runner(qr)
    # x = q.dequeue()
    # x = tf.train.batch()
    images, labels = fw.data_producer(conf, images, labels)
    logits = infer(images)
    loss = fw.loss(logits, labels)
    train_op = fw.train(conf, loss, global_step)
    fw.run_sess(conf, train_op, loss, global_step)
    # class _LoggerHook(tf.train.SessionRunHook):
    #     def begin(self):
    #         self._step = -1
    #         self._start_time = time.time()
    #
    #     def before_run(self, run_context):
    #         self._step += 1
    #         return tf.train.SessionRunArgs(loss)  # Asks for loss value.
    #
    #     def after_run(self, run_context, run_values):
    #         if self._step % conf.log_frequency == 0:
    #             current_time = time.time()
    #             duration = current_time - self._start_time
    #             self._start_time = current_time
    #
    #             loss_value = run_values.results
    #             examples_per_sec = conf.log_frequency * conf.batch_size / duration
    #             sec_per_batch = float(duration / conf.log_frequency)
    #
    #             format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.4f '
    #                           'sec/batch)')
    #             print(format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
    #
    # # with tf.train.MonitoredTrainingSession(checkpoint_dir=conf.train_dir, hooks=[tf.train.StopAtStepHook(num_steps=conf.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()], config=tf.ConfigProto(log_device_placement=conf.log_device_placement)) as mon_sess:
    # #     while not mon_sess.should_stop():
    # #         mon_sess.run(train_op)

def evaluate(mnist):
    images, labels = mnist.test.images, mnist.test.labels
    images = np.reshape(images, [len(images), *conf.input_shape])
    fw.evaluate(conf, images, labels, infer)


def main():
    mnist = input_data.read_data_sets('data', one_hot=True)
    is_eval = False
    if len(sys.argv) > 1:
        is_eval = True
    if is_eval:
        evaluate(mnist)
    else:
        train(mnist)
        evaluate(mnist)


if __name__ == '__main__':
    main()