
'''
    sorfmax 练习
    MNIST 入门
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''准备数据，'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。

# 通过为输入图像和目标输出类别创建节点，来开始构建计算图。
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''建立模型'''
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''训练模型'''
# 计算交叉熵， 成本函数或损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# 训练1000次，每次都随机抓取训练数据集中的100个批处理数据点
# 每次训练后会自动进行反向传播，优化参数
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''评估模型'''
# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 将布尔值转为浮点值，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print('accuracy', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
