import sys
sys.path.append('../../lib')

import stock_data
import tf_framework as fw
import tensorflow as tf

id = '000001.XSHE'
x,y = stock_data.prepare_single(id)
x,y = fw.rnn_producer(x, y, 2, 10)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    ths = tf.train.start_queue_runners(sess, coord=coord)
    print(x.eval(), y.eval())
    coord.request_stop()
    coord.join(ths)

