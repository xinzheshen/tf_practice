
import tensorflow as tf


input_batch = tf.constant([
        [
          # 第一个输入（如果是一幅2D图像，它的维数包括其宽度、高度和通道数，即3维，
          # 所以第一个输入可以代表一个高2个像素，宽2个像素，深度为一个通道的图像）
            [[0.0], [1.0]],
            [[2.0], [3.0]]
        ],
        [
            # 第二个输入
            [[2.0], [4.0]],
            [[6.0], [8.0]]
        ]
    ])

kernel = tf.constant([
        [
            [[1.0, 5.0]]
        ]
    ])
print('input_shape', input_batch.shape)
# shape 同时刻画了张量的维（阶）数 以及 每一维的长度， 张量的形状可以是包含有序整数集的俩报表或元组
# 列表中元素的数量与维数一致，且每个元素描述了相应维数上的长度
# input_shape (2, 2, 2, 1)
print('kernel_shape', kernel.shape)
# kernel_shape (1, 1, 1, 2)
conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.Session()
output = sess.run(conv2d)
# 该输出是另一个与input_batch同秩的张量，但其维数与卷积核相同
print('output', output)
print('output_shape', output.shape)
# output_shape (2, 2, 2, 2)

kernel_shape = tf.shape(kernel)
print('kernel_shape2', sess.run(kernel_shape))

