import tensorflow as tf

# # tensorflow Operation 也称Op， 是一些对（或利用）Tensor对象执行运算的节点，计算后返回0个或多个张量
# # 通过给Op提供name参数，可用描述性字符来指代某个特定的Op
# a = tf.constant(5, name='input_a')
# b = tf.constant(3, name='input_b')
# c = tf.multiply(a, b, name='mul_c')
# d = tf.add(a, b, name='add_d')
# e = tf.add(c, d, name='add_e')
#
# sess = tf.Session()
# output = sess.run(e)
# print(output)
#
# writer = tf.summary.FileWriter('./my_graph', sess.graph)
# # 在对应的虚拟环境下运行 tensorboard --logdir=.\src\my_graph ，在浏览器中访问 6006端口可查看数据流图
# writer.close()
# sess.close()

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



