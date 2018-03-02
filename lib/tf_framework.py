# coding=utf-8
import time
import time_plus
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


def data_type():
    return tf.float32


def get_weights_bias(shape, wd = None):
    w = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    # w = tf.get_variable('weight', shape, dtype=data_type(), initializer=tf.constant_initializer(0.0))
    b = tf.get_variable('bias', shape[-1], dtype=data_type(), initializer=tf.constant_initializer(0.1))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_regularizered')
        # weight_decay = tf.contrib.layers.l2_regularizer(0.0001)(w)
        tf.add_to_collection('losses', weight_decay)
    return w, b


# x必须为匹配全链接格式
def full_layer(x, w_shape, wd, idx = 1):
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
        w, b = get_weights_bias(w_shape, wd)
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
        w, b = get_weights_bias(filter_shape)
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


def loss(logits, labels, weight = None):
    with tf.name_scope('loss'):
        # if weight is not None:
        #     m = [1] * self.output_num
        #     for k, v in weight.items():
        #         m[k] = v
        #     wted_labels = labels * m
        #     print('self defined')
        #     _log = -tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-10, 1.0))
        #     cross_entropy = wted_labels * _log
        # else:
        print('sparse_softmax')
        if type(labels).__name__ == 'Tensor':
            shape = labels.get_shape().as_list()
            if len(shape) == 2:
                matched_labels = tf.argmax(labels, 1)
            else:
                matched_labels = labels
        else:
            matched_labels = tf.argmax(labels, 1)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=matched_labels, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', loss)
        loss = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss-summary', loss)
        return loss


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(conf, total_loss, global_step):
    decay_steps = conf.epoch_size_for_train // conf.batch_size * conf.num_epochs_per_decay

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(conf.learning_rate_base, global_step, decay_steps, conf.learning_rate_decay, staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    # 这里只是给可训练的变量加上移动平均，并未给loss加
    variable_averages = tf.train.ExponentialMovingAverage(conf.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


# 返回一个出队op， x,y
def data_producer(conf, x_input, y_input):
    total = len(y_input)
    if total == 0:
        raise Exception('no data for train')
    x_input = tf.convert_to_tensor(x_input, dtype=data_type())
    y_input = tf.convert_to_tensor(y_input, dtype=tf.int32)
    epoch = total // conf.batch_size
    epoch = tf.identity(epoch, name='epoch_size')
    i = tf.train.range_input_producer(epoch + 1, shuffle=False).dequeue()
    start = (i * conf.batch_size) % total
    # end = min(start + self.batch_size, total)
    end = start + conf.batch_size
    xs = tf.strided_slice(x_input, [start], [end])
    ys = tf.strided_slice(y_input, [start], [end])
    # xs, ys = x_input[start:end], y_input[start:end]
    return xs, ys


def rnn_producer(raw_x, raw_y, batch_size, num_steps, name = None):
    with tf.name_scope(name, "rnn_producer", [raw_x, raw_y, batch_size, num_steps]):
        raw_x = tf.convert_to_tensor(raw_x, name="raw_x", dtype=tf.float32)
        raw_y = tf.convert_to_tensor(raw_y, name='raw_y', dtype=tf.int32)
        shape = raw_x.get_shape().as_list()
        data_len = shape[0]
        dims = shape[1]
        batch_len = data_len // batch_size
        data_x = tf.reshape(raw_x[0: batch_size * batch_len], [batch_size, batch_len, dims])
        data_y = tf.reshape(raw_y[0: batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data_x, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps, dims])
        y = tf.strided_slice(data_y, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        y.set_shape([batch_size, num_steps])
        return x, y


def run_sess(conf, train_op, loss, global_step):
    # global_step = tf.Variable(0, trainable=False)

    start_time = int(time.time())
    print('start', time_plus.std_datetime(start_time))
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(conf.train_dir, tf.get_default_graph())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_meta = tf.RunMetadata()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        for i in range(conf.max_steps):
            if i % conf.log_frequency == 0:
                now_time = int(time.time())
                print('now:', now_time, 'elapse:', now_time - start_time)
                loss_v, step_v = sess.run([loss, global_step])
                print(step_v, loss_v)
                summary_op = tf.summary.merge_all()
                writer.add_summary(sess.run(summary_op), step_v)
                sess.run(train_op, options=run_options, run_metadata=run_meta)
                writer.add_run_metadata(run_meta, 'step-%d' % i)
                # timeline
                fetched_timeline = timeline.Timeline(run_meta.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(conf.timeline_dir + 'step_%d.json' % i, 'w') as f:
                    f.write(chrome_trace)
                saver.save(sess, conf.train_dir + conf.model_name, global_step=global_step)
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


def evaluate(conf, saver, logits = None, labels = None):
    # acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_input, 1)), tf.float32))


    with tf.Session() as sess:
        # tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        ckpt = tf.train.get_checkpoint_state(conf.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('no checkpoint file')
            return
        num_iter = conf.test_num // conf.batch_size
        total_num = num_iter * conf.batch_size
        acc_op = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), labels)
        i = 0
        true_count = 0
        while i < num_iter and not coord.should_stop():
            i += 1
            acc_v = sess.run([acc_op])
            true_count += np.sum(acc_v)
            # print(acc_v)
        real_acc = true_count / total_num
        print(real_acc)
        coord.request_stop()
        coord.join(threads)
