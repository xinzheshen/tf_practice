'''
 线性回归练习
 数据集： http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
'''

import tensorflow as tf

# 通用代码框架

# 初始化变量和模型参数，定义训练闭环中的运算

# 设置权值为2x1的列向量
w = tf.Variable(tf.zeros([2, 1]), name='weights')
b = tf.Variable(0., name='bias')


# 计算推断模型在数据X上的输出，并将结果返回
def inference(X):
    with tf.name_scope('inference'):
        # 线性回归
        return tf.matmul(X, w) + b


# 依据训练数据X及其期望输出Y计算损失
def loss(X, Y):
    Y_predicted = inference(X)
    with tf.name_scope('loss'):
        # 计算总平方误差，即每个预测值与期望输出只差的平方的总和
        return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


# 读取或生成训练数据X及其期望输出Y
def inputs():
    with tf.name_scope('inputs'):
        weight_age = [[84, 46],
                      [73, 20],
                      [65, 52],
                      [70, 30],
                      [76, 57],
                      [69, 25],
                      [63, 28],
                      [72, 36],
                      [79, 57],
                      [75, 44],
                      [27, 24],
                      [89, 31],
                      [65, 52],
                      [57, 23],
                      [59, 60],
                      [69, 48],
                      [60, 34],
                      [79, 51],
                      [75, 50],
                      [82, 34],
                      [59, 46],
                      [67, 23],
                      [85, 37],
                      [55, 40],
                      [63, 30]]
        blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395,
                             434, 220, 374, 308, 220, 311, 181, 274, 303, 244]

        return tf.to_float(weight_age), tf.to_float(blood_fat_content)


# 依据计算的总损失训练或调整模型参数
def train(total_loss):
    with tf.name_scope('train'):
        learning_rate = 0.0000001
        # 采用梯度下降算法对模型参数进行优化
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


# 对训练得到的模型进行评估
def evaluate(sess, X, Y):
    with tf.name_scope('evaluate'):
        print('80kg: ', sess.run(inference([[80., 25.]])))
        print('65kg: ', sess.run(inference([[65., 25.]])))


with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)

    train_op = train(total_loss)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    histogram = tf.summary.histogram('weights', w)
    writer = tf.summary.FileWriter('D:\\sxz\\practice\\other\\log', sess.graph)
    summaries = tf.summary.merge_all()

    # 实际训练迭代次数
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])

        train_summary = sess.run(histogram)
        writer.add_summary(train_summary, global_step=step)
        if step % 10 == 0:
            print('loss', sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    writer.close()







