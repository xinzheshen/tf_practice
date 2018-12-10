'''
 线性回归练习
 数据集： http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 通用代码框架

# 计算推断模型在数据X上的输出，并将结果返回
def inference(X):
    with tf.name_scope('inference'):
        # 线性回归
        return tf.matmul(X, w) + b


# 依据训练数据X及其期望输出Y计算损失
def loss(X, Y):
    Y_predicted = inference(X)
    with tf.name_scope('loss'):
        # 计算总平方误差，即每个预测值与期望输出之差的平方的总和
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


def plot_figure(data_X, data_Y, data_Y_cal):
    fig = plt.figure(1, figsize=(8, 4))
    ax = Axes3D(fig)

    print('data_x shape', data_X.shape)
    print('data_Y shape', data_Y.shape)
    print('data_Y_cal shape', data_Y_cal.shape)
    # 坐标轴
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    ax.scatter(data_X[:, 0], data_X[:, 1], data_Y, c='r', s=1)
    ax.scatter(data_X[:, 0], data_X[:, 1], data_Y_cal, 'b--', s=1)

    plt.show()


if __name__ == '__main__':
    # 设置权值为2x1的列向量
    w = tf.Variable(tf.zeros([2, 1]), name='weights')
    b = tf.Variable(0., name='bias')

    X, Y = inputs()

    total_loss = loss(X, Y)

    train_op = train(total_loss)

    scalar_loss = tf.summary.scalar('myloss', total_loss)
    hisrogram_weights = tf.summary.histogram('myweights', w)
    hisrogram_bias = tf.summary.histogram('mybias', b)
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:

        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        writer = tf.summary.FileWriter('D:\\sxz\\practice\\other\\log', sess.graph)

        # 实际训练迭代次数
        training_steps = 1000
        for step in range(training_steps):
            sess.run([train_op])

            # weights_summary, bias_summary, loss_summary = sess.run([hisrogram_weights, hisrogram_bias, scalar_loss])
            # writer.add_summary(weights_summary, global_step=step)
            # writer.add_summary(bias_summary, global_step=step)
            # writer.add_summary(loss_summary, global_step=step)

            train_summary = sess.run(summaries)
            writer.add_summary(train_summary, global_step=step)

            if step % 10 == 0:
                print('loss', sess.run([total_loss]))

        weight = sess.run(w)
        bia = sess.run(b)

        evaluate(sess, X, Y)

        coord.request_stop()
        coord.join(threads)

        writer.close()

        #计算预测的结果
        Y_cal = bia + weight[0]*X[:, 0] + weight[1]*X[:, 1]

        plot_figure(X.eval(), Y.eval(), Y_cal.eval())









