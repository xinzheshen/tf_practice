#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

'''常用张量的创建 及 variable对象'''

# -----------------一些常用张量的创建方法-----------------
# 2x2的零矩阵
zeros = tf.zeros([2, 2])

# 长度为6的全1矩阵
ones = tf.ones([6])

# 3x3x3的张量，其元素服从0~10的均分分布
union = tf.random_uniform([3, 3, 3], minval=0, maxval=10)

# 3x3x3的张量，其元素服从0均值，标准差为2的正态分布
normal = tf.random_normal([3, 3, 3], mean=0, stddev=2.0)

# truncated_normal能保证不会创建任何偏离均值超过2倍标准差的值，
# 即下面的Tensor对象不会返回任何小于3.0或大于7.0的值
tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)
# 默认的均值为0，标准差为1.0
tf.truncated_normal([2, 2])

# --------------variable对象 (Tensor对象和Op对象都是不可变的（immutable）)--------------
my_var = tf.Variable(3, name='my_variable')
a = tf.add(5, my_var)
# 初始化全部variable对象
init = tf.initialize_all_variables()
# # 初始化指定的对象
# tf.initialize_variables([my_var], name='init_my_var')
sess = tf.Session()
sess.run(init)
print('cal a : ', sess.run(a))

# 修改variable对象
my_var = my_var.assign(my_var * 2)
print('cal my_var_times_two : ', sess.run(my_var))
print('cal my_var_times_two : ', sess.run(my_var))
print('cal a after updated : ', sess.run(a))

# 如果想重置variable对象为初始值，可再次调用初始化方法
sess.run(init)
print('cal a after init again : ', sess.run(a))

# 在训练机器学习模型的Optimizer类中，会自动修改variable对象的值，如果只允许手动修改，可设置trainable参数为false
my_var = tf.Variable(3, name='my_variable', trainable=False)

writer = tf.summary.FileWriter('./my_graph', sess.graph)
# 在对应的虚拟环境下运行 tensorboard --logdir=.\src\my_graph ，在浏览器中访问 6006端口可查看数据流图
writer.close()

sess.close()
