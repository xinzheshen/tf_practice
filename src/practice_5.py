#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

'''占位符'''

# 创建一个长度为2，数据类型为int32的占位向量(shape 参数默认为None，表示可接收任意形状的Tensor对象）
a = tf.placeholder(tf.int32, shape=[2], name='my_input')
# 将占位向量视为其他任意Tensor对象，加以使用
b = tf.reduce_prod(a, name='prod_b')
c = tf.reduce_sum(a, name='sum_c')
d = tf.add(b, c, name='add_d')

# 可以用with 语句打开会话
with tf.Session() as sess:
    # 利用feed_dict 参数传入一个实际值，用以测试
    input_dict = {a: np.array([5, 3], dtype=np.int32)}
    print('cal d with placeholder : ', sess.run(d, feed_dict=input_dict))
