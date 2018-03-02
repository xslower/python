# coding=utf-8
import time
import time_plus
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


def get_weights_bias(shape, idx = 1, regular = None):
    # w = tf.get_variable('weight-' + str(idx), shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    w = tf.get_variable('weight-' + str(idx), shape, initializer=tf.constant_initializer(0.0))
    b = tf.get_variable('bias-' + str(idx), shape[-1], initializer=tf.constant_initializer(0.1))
    if regular is not None:
        tf.add_to_collection('losses', regular(w))
    return w, b


# x必须为匹配全链接格式
def full_layer(x, w_shape, idx = 1, regular = None):
    with tf.variable_scope('full-' + str(idx)):
        if len(w_shape) != 2:
            raise Exception('w_shape must be rank 2 list, such as [None, 256]')

        shape_x = x.get_shape().as_list()
        if len(shape_x) > 2:  # 自动把输入转为扁平
            num = 1
            for i in range(1, len(shape_x)):
                num *= shape_x[i]
            x = tf.reshape(x, [-1, num])
        else:
            num = shape_x[-1]
        if w_shape[0] is None:
            w_shape[0] = num
        w, b = get_weights_bias(w_shape, idx, regular)
        layer = tf.nn.relu(tf.matmul(x, w) + b)
        return layer


def conv2d_layer(x, filter_shape, idx = 1, stride = None):
    with tf.variable_scope('conv2d-' + str(idx)):
        if len(filter_shape) != 4:
            raise Exception('filter_shape must be rank 4 list, such as [3, 3, None, 32]')
        shape_x = x.get_shape().as_list()
        if filter_shape[2] is None:
            filter_shape[2] = shape_x[-1]
        for i in range(2):  # filter的宽高不能大于输入数据的宽高
            if filter_shape[i] > shape_x[i + 1]:
                filter_shape[i] = shape_x[i + 1]
        w, b = get_weights_bias(filter_shape, idx)
        if stride is None:
            stride = [1, 1, 1, 1]
        conv = tf.nn.conv2d(x, w, strides=stride, padding='SAME')
        layer = tf.nn.relu(tf.nn.bias_add(conv, b))
        return layer


# pool层ksize参数格式与conv不同，1、4都必须=1, 2、3=宽高的窗口大小
def pool_layer(x, filter_shape = None):
    if filter_shape is None:
        filter_shape = [1, 2, 2, 1]
    elif len(filter_shape) == 2:
        filter_shape = [1, *filter_shape, 1]
    layer = tf.nn.max_pool(x, ksize=filter_shape, strides=[1, 2, 2, 1], padding='SAME')
    return layer


# neural network model
class NnModel:
    inference = None
    # loss = None
    # input_shape = []
    reshape = False
    # all_shape = []
    input_shape = None
    output_num = 2
    batch_size = 100
    learning_rate_base = 0.5
    learning_rate_decay = 0.97
    regular_rate = 0.0001
    training_steps = 5000
    moving_avaerage_decay = 0.99
    save_path = 'data/model/'
    model_name = 'm.ckpt'
    summary_dir = 'summary/'
    optimize = False

    def __init__(self, **kv):
        for k, v in kv.items():
            setattr(self, k, v)

    def set_attr(self, obj_config):
        for k, v in obj_config.__dict__.items():
            if '__' in k:
                continue
            setattr(self, k, v)

    def x_holder_shape(self):
        if isinstance(self.input_shape, int):
            x_shape = [None, self.input_shape]
        elif isinstance(self.input_shape, list):
            x_shape = [None, *self.input_shape]
        else:
            raise Exception('input_shape not defined or format error')
        return x_shape

    def shaped_x(self, x_input):
        if self.reshape:
            x_input = np.reshape(x_input, [len(x_input), *self.input_shape])
        return x_input

    def dropout_infer(self, x, regular = None, train = False):
        if self.inference is None:
            raise Exception('not defined inference')
        mid = self.inference(x, regular)
        if train:
            mid = tf.nn.dropout(mid, 0.5)
        y = full_layer(mid, [None, self.output_num], 999, regular)
        return y

    # 返回一个出队op， x,y
    def data_producer(self, data):
        x_input, y_input = data
        total = len(y_input)
        if total == 0:
            raise Exception('no data for train')
        x_input = self.shaped_x(x_input)
        x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
        y_input = tf.convert_to_tensor(y_input, dtype=tf.float32)
        epoch = total // self.batch_size
        epoch = tf.identity(epoch, name='epoch_size')
        i = tf.train.range_input_producer(epoch + 1, shuffle=False).dequeue()
        start = (i * self.batch_size) % total
        # end = min(start + self.batch_size, total)
        end = start + self.batch_size
        xs = tf.strided_slice(x_input, [start], [end])
        ys = tf.strided_slice(y_input, [start], [end])
        # xs, ys = x_input[start:end], y_input[start:end]
        return xs, ys

    # rate可以指定某个类别的权重。例如{1:90}表示类别1的权重为90，默认权重为1
    def train(self, data: tuple, rate: dict = None) -> None:

        with tf.name_scope('input'):
            # x_holder = tf.placeholder(tf.float32, self.x_holder_shape(), name='x-holder')
            # y_holder = tf.placeholder(tf.float32, [None, self.output_num], name='y-holder')
            x_holder, y_holder = self.data_producer(data)

        regular = tf.contrib.layers.l2_regularizer(self.regular_rate)
        y_train = self.dropout_infer(x_holder, regular, True)
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope('moving_average'):
            va = tf.train.ExponentialMovingAverage(self.moving_avaerage_decay, global_step)
            va_op = va.apply(tf.trainable_variables())

        with tf.name_scope('loss'):
            if rate is not None:
                m = [1] * self.output_num
                for k, v in rate.items():
                    m[k] = v
                y_mholder = y_holder * m
                print('self defined')
                _log = -tf.log(tf.clip_by_value(tf.nn.softmax(y_train), 1e-10, 1.0))
                cross_entropy = y_mholder * _log
            else:
                # y_mholder = y_holder
                print('sparse_softmax')
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_train, labels=tf.argmax(y_holder, 1))
            loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('loss-summary', loss)
        batch = self.batch_size
        total = len(data[1])
        with tf.name_scope('train_step'):
            learn_rate = tf.train.exponential_decay(self.learning_rate_base, global_step, total / batch, self.learning_rate_decay, True)
            train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step)
            train_op = tf.group(train_step, va_op)

        start_time = int(time.time())
        print('start', time_plus.std_datetime(start_time))
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.save_path + self.summary_dir, tf.get_default_graph())
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_meta = tf.RunMetadata()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(self.training_steps):
                if i % 1000 == 0:
                    now_time = int(time.time())
                    print('now:', now_time, 'elapse:', now_time - start_time)
                    loss_v, step_v = sess.run([loss, global_step])
                    print(step_v, loss_v)
                    if self.optimize:
                        summary_op = tf.summary.merge_all()
                        writer.add_summary(sess.run(summary_op), global_step)
                        sess.run(train_op, options=run_options, run_metadata=run_meta)
                        # writer.add_run_metadata(run_meta, 'step-%d' % i)
                        # timeline
                        fetched_timeline = timeline.Timeline(run_meta.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open('timeline/step_%d.json' % i, 'w') as f:
                            f.write(chrome_trace)
                    else:
                        sess.run(train_op)
                    saver.save(sess, self.save_path + self.model_name, global_step=global_step)
                else:
                    sess.run(train_op)
            coord.request_stop()
            coord.join(threads)
        end_time = int(time.time())
        print('now:', end_time, 'total elapse:', end_time - start_time)
        writer.close()

    def predict(self, x_input):
        with tf.Graph().as_default():
            x_holder = tf.placeholder(tf.float32, self.x_holder_shape(), name='x-input')
            va_feed = {x_holder: self.shaped_x(x_input)}
            predict_y = self.dropout_infer(x_holder)
            # acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict_y, 1), tf.argmax(ty, 1)), tf.float32))
            va = tf.train.ExponentialMovingAverage(self.moving_avaerage_decay)
            var_restore = va.variables_to_restore()
            saver = tf.train.Saver(var_restore)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    yv = sess.run(predict_y, feed_dict=va_feed)
                    ret = tf.argmax(yv, 1).eval()
                    return ret
                else:
                    print('no checkpoint file')
                    return

    def evaluate(self, data):
        with tf.Graph().as_default():
            x_holder = tf.placeholder(tf.float32, self.x_holder_shape(), name='x-input')
            y_holder = tf.placeholder(tf.float32, [None, self.output_num], name='y-input')
            x_input, y_input = data
            va_feed = {x_holder: self.shaped_x(x_input), y_holder: y_input}
            predict_y = self.dropout_infer(x_holder)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1)), tf.float32))
            va = tf.train.ExponentialMovingAverage(self.moving_avaerage_decay)
            var_restore = va.variables_to_restore()
            saver = tf.train.Saver(var_restore)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    acc_score = sess.run(acc, feed_dict=va_feed)
                    print(acc_score)
                else:
                    print('no checkpoint file')
                    return
