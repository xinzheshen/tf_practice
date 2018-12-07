#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf

'''run() 的输入参数'''

a = tf.constant([5, 3], name='input_a')
b = tf.reduce_sum(a, name='add_b')
c = tf.reduce_prod(a, name='mul_c')
d = tf.add(b, c, name='add_d')

sess = tf.Session()

# fetches参数可接收 Op或Tensor对象，后者则输出一个Numpy数组， 前者输出为None
# 接收Tensor对象
print('cal d : ', sess.run(d))
print('cal b, c, d : ', sess.run([b, c, d]))
# # 接收Op句柄
# print('Op', sess.run(tf.initialize_all_variables()))

# feed_dict参数 用于覆盖数据流图中的Tensor对象值
replace_dict = {b: 15}
print('cal d afer replaced : ', sess.run(d, feed_dict=replace_dict))

# writer = tf.summary.FileWriter('./my_graph', sess.graph)
# # 在对应的虚拟环境下运行 tensorboard --logdir=.\src\my_graph ，在浏览器中访问 6006端口可查看数据流图
# writer.close()

sess.close()
