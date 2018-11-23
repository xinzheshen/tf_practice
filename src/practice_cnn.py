
import tensorflow as tf


input_batch = tf.constant([
        [ # 第一个输入
            [[0.0], [1.0]],
            [[2.0], [3.0]]
        ],
        [ #第二个输入
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
print('kernel_shape', kernel.shape)
conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.Session()
output = sess.run(conv2d)
print('output', output)
print('output_shape', output.shape)

