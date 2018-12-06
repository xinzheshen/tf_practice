#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf


'''name scope'''

with tf.name_scope('scope_add'):
    a = tf.add(1, 2, name='add_a')
    b = tf.add(a, 2, name='add_b')

with tf.name_scope('scope_multiply'):
    c = tf.multiply(3, 2, name='multiply_c')
    d = tf.multiply(c, 2, name='multiply_d')

e = tf.add(b, d, name='output')

with tf.Session() as sess:
    print('cal e : ', sess.run(e))
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    writer.close()









