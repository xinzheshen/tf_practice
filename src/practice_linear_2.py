'''
    二元线性回归案例
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D


'''
准备数据，并设置目标函数
'''
# 产生测试数据
def genDate(numPoints,bias,variance):
    '''
    :param numPoints : 实例个数 两维数据
    :param bias : 偏向值
    :param variance : 变化
    返回得到的x和y数据
    '''
    # 产生numPoints*2的零矩阵
    x = np.zeros(shape=(numPoints, 2))
    # 产生一维数组
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        # x赋值  x = [[1,0],[1,1],[1,2]...]
        x[i][0] = 1
        x[i][1] = i
        # y赋值  正太分布
        y[i] = i + np.random.normal(loc=bias, scale=variance, size=None)
    return np.array(x,dtype='float32').reshape(numPoints,2),np.array(y,dtype='float32').reshape(numPoints,1)


# 生成数据
x_data, y_data = genDate(100,25,3)

print(x_data.shape, y_data.shape)

# print(training_x,training_y)


fig = plt.figure(1, figsize=(8,4))
# http://blog.csdn.net/eddy_zheng/article/details/48713449
# ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
ax = Axes3D(fig)

# 坐标轴
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')

ax.scatter(x_data[:,0],x_data[:,1],y_data,c='r',s=1)         #绘制数据点


# 构造二元线性回归模型
b = tf.Variable(1.0)
w = tf.Variable(tf.ones([2, 1]))
y = tf.matmul(x_data,w) + b

# 设置均方差损失函数，在使用梯度下架法的时候学习率不能选择太多，不然会震荡，不会收敛
cost = tf.reduce_mean(tf.square(y - y_data))  # 拟合效果更好

# 选择绝对损失函数可以拟合很好
# cost = tf.reduce_mean(tf.abs(y - y_data))

# 选择梯度下降的方法 传入学习率
optimizer = tf.train.GradientDescentOptimizer(0.0001)   # 学习率不能选择过大，不然会震荡
# 迭代的目标，最小化损失函数
train = optimizer.minimize(cost)

'''
#开始求解
'''
# 初始化变量：tf的准备工作，主要声明了变量，就必须先初始化才可以使用
init = tf.global_variables_initializer()

#设置tensorflow对GPU使用按需分配
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#使用会话执行图
with tf.Session(config=config) as sess:
    sess.run(init)
    #迭代，重复执行最下化损失函数这一步骤
    for step in range(100000):
        sess.run(train)
        if step % 10000 == 0:
            print('迭代次数{0}:W->{1},b->{2},{3}'.format(step,sess.run(w),sess.run(b),sess.run(cost)))
    #保存最后结果
    rw = sess.run(w)
    rb = sess.run(b)


#计算预测的结果
X, Y = np.meshgrid(x_data[:,0], x_data[:,1])
Z = rb + rw[0]*X + rw[1]*Y
#绘制数据点
ax.scatter(X, Y, Z, 'b--', s=1)
plt.show()
print('end')
