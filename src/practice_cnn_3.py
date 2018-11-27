'''
 卷积神经网络
 数据集： http://vision.stanford.edu/aditya86/ImageNetDogs/
'''

import tensorflow as tf

# ---------------加载图像-------------------------------

source = '../output/training-images/*.tfrecords'
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(source))
reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)

features = tf.parse_single_example(serialized, features={
    'label': tf.FixedLenFeature([], tf.string),
    'image': tf.FixedLenFeature([], tf.string),
    })

record_image = tf.decode_raw(features['image'], tf.uint8)

# 修改图像的形状有助于训练和输出的可视化
image = tf.reshape(record_image, [250, 151, 1])

label = tf.cast(features['label'], tf.string)
min_after_dequeue = 10
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                  capacity=capacity, min_after_dequeue=min_after_dequeue)

# -------------------模型-----------------
# 将图像转换为灰度值位于[0,1)的浮点类型，以与convolution2d期望的输入匹配
float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

conv2d_layer_one = tf.layers.conv2d(float_image_batch, filters=32, kernel_size=(5, 5), activation=tf.nn.relu,
                                    strides=(2, 2), trainable=True)

pool_layer_one = tf.nn.max_pool(conv2d_layer_one, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积输出的第1维和最后一维未发生改变，但中间的两维发生了变化
print('conv2d_layer_one', conv2d_layer_one.get_shape())
print('pool_layer_one', pool_layer_one.get_shape())

conv2d_layer_two = tf.layers.conv2d(pool_layer_one, filters=64, kernel_size=(5, 5), activation=tf.nn.relu,
                                    strides=(1, 1), trainable=True)

pool_layer_two = tf.nn.max_pool(conv2d_layer_two, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积输出的第1维和最后一维未发生改变，但中间的两维发生了变化
print('conv2d_layer_two', conv2d_layer_two.get_shape())
print('pool_layer_two', pool_layer_two.get_shape())


# 将图像中的每个点都与输出神经元建立全连接。由于本例中，后面要是使用softmax，因此全连接层需改为二阶张量。
# 张量的第1维用于区分每幅图像，第2维对应于每个输入张量的秩1张量

flattened_layer_two = tf.reshape(pool_layer_two,
                                 [batch_size, # image_batch中的每幅图像
                                  -1] # 输入的其他所有维
                                 )
print('flattened_layer_two', flattened_layer_two.get_shape())

# 池化层展开后，便可与将网络当前状态与所预测的狗的品种关联的两个全连接层进行整合

hidden_layer_three = tf.layers.dense(flattened_layer_two, 512, activation=tf.nn.relu)

# 对一些神经元进行dropout处理，削减它们在模型中的重要性
hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

# 输出是前面的参与训练中可用的120个不同的狗的品种的全连接
final_fully_connected = tf.layers.dense(hidden_layer_three, 5)
