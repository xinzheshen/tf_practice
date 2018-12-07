#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf


# 利用张量
a = tf.constant([5, 3], name='input_a')
# 当以张量形式输入时， reduce_xxx 函数会接收其所有分量，再进行xxx运算
c = tf.reduce_sum(a, name='mul_c')
d = tf.reduce_prod(a, name='add_d')
e = tf.add(c, d, name='add_e')

shape = tf.shape(a, name='a_shape')

sess = tf.Session()
output = sess.run(e)
print(output)
print('shape', sess.run(shape))

# writer = tf.summary.FileWriter('./my_graph', sess.graph)
# # 在对应的虚拟环境下运行 tensorboard --logdir=.\src\my_graph ，在浏览器中访问 6006端口可查看数据流图
# writer.close()
sess.close()
