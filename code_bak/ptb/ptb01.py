import sys

sys.path.append('./')

import numpy as np
import tensorflow as tf
import reader


class ptb_model(object):
    data_path = 'data/data'
    hidden_size = 200
    num_layers = 2
    vocab_size = 10000
    learning_rate = 1.0

    num_epoch = 2
    keep_prob = 0.5
    max_grad_norm = 5

    def __init__(self, train, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.inputs, self.targets = reader.ptb_producer(train, batch_size, num_steps)
        # 定义lstm网络，并使用dropout
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        if train:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
        # 初始化全零向量
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        # 将单词id转为单词向量
        embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size])
        #
        inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        if train:
            inputs = tf.nn.dropout(inputs, self.keep_prob)
        # 定义输出列表，把lstm中间时刻的输出收集，再通过一个全链接层得到最终输出
        outputs = []
        # 存储lstm的中间状态
        state = self.initial_state
        with tf.variable_scope('rnn'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                #
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_size])
        # 经过全链接层
        w = tf.get_variable('weight', [self.hidden_size, self.vocab_size])
        b = tf.get_variable('bias', [self.vocab_size])
        logits = tf.matmul(output, w) + b

        # 定义损失函数
        loss = tf.contrib.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])], [tf.ones([batch_size * num_steps], tf.float32)])

        #
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not train:
            return
        trainable_var = tf.trainable_variables()
        # 通过clip_by_global_norm控制梯度的大小
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_var), self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_var))


# 训练
def run_epoch(sess, model, data, train_op, output_log):
    #
    total_costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)
    # for step, (x,y) in enumerate()
    cost, state, _ = sess.run([model.cost, model.final_state, train_op])
    total_costs += cost
    iters += model.num_steps
    return np.exp(total_costs / iters)


def main():
    train_batch_size = 20
    train_num_step = 35

    test_batch_size = 1
    test_num_step = 1

    train_data, valid_data, test_data, _ = reader.ptb_raw_data('data/data')
    # 初始化
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('language-model', reuse=None, initializer=initializer):
        train_model = ptb_model(True, train_batch_size, train_num_step)
    with tf.variable_scope('language-model', reuse=True, initializer=initializer):
        eval_model = ptb_model(False, test_batch_size, test_num_step)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord=coord)
        tf.global_variables_initializer().run()
        for i in range(1000):
            run_epoch(sess, train_model, train_data, train_model.train_op, True)

            valid_perpelexity = run_epoch(sess, eval_model, valid_data, tf.no_op(), False)
            print('epoch: %d validation : %.3f' % (i + 1, valid_perpelexity))

        coord.request_stop()
        coord.join()


if __name__ == '__main__':
    # tf.app.run()
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = tf.constant([[3, 2, 1], [6, 5, 4],[9,8,7]])
    inputs = tf.concat((a,b), 1)
    # tf.nn.rnn_cell.BasicRNNCell
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(inputs.eval())
