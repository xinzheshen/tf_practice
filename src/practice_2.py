import tensorflow as tf
import numpy as np


# 创建一个新的数据流图，Graph 对象
g = tf.Graph()
# Graph.as_default()方法访问其上下文管理器，为其添加Op
with g.as_default():
    a = tf.add(2, 3)
    c = tf.multiply(a, 3)

# 如果不手动创建数据流图， tensorflow库被加载时会自动创建一个Graph对象，并将其作为默认的数据流图
g_default = tf.get_default_graph()
with g_default.as_default():
    b = tf.multiply(2, 3)

# 指定数据流图
sess = tf.Session(graph=g)
# -----------------run() 的输入参数--------------------------------------
# fetches参数可接收 Op或Tensor对象，后者则输出一个Numpy数组， 前者输出为None
# 接收Tensor对象
print(sess.run(c))
print(sess.run([a, c]))
# # 接收Op句柄
# print('Op', sess.run(tf.initialize_all_variables()))

# feed_dict参数 用于覆盖数据流图中的Tensor对象值
replace_dict = {a: 15}
print('replace', sess.run(c))
sess.close()

# ------------------占位符 ----------------------------------------
# 创建一个长度为2，数据类型为int32的占位向量(shape 参数默认为None，表示可接收任意形状的Tensor对象）
a = tf.placeholder(tf.int32, shape=[2], name='my_input')
# 将占位向量视为其他任意Tensor对象，加以使用
b = tf.reduce_prod(a, name='prod_b')
c = tf.reduce_sum(a, name='sum_c')
# 完成数据流图的定义
d = tf.add(b, c, name='add_d')

with tf.Session() as sess:
    # 利用feed_dict 参数传入一个实际值，用以测试
    input_dict = {a: np.array([5, 3], dtype=np.int32)}
    print('placeholder', sess.run(d, feed_dict=input_dict))

# -----------------一些常用张量的创建方法-----------------
# 2x2的零矩阵
zeros = tf.zeros([2, 2])

# 长度为6的全1矩阵
ones = tf.ones([6])

# 3x3x3的张量，其元素服从0~10的均分分布
union = tf.random_uniform([3, 3, 3], minval=0, maxval=10)

# 3x3x3的张量，其元素服从0均值，标准差为2的正态分布
normal = tf.random_normal([3, 3, 3], mean=0, stddev=2.0)

# truncated_normal能保证不会创建任何偏离均值超过2倍标准差的值， 即下面的Tensor对象不会返回任何小于3.0或大于7.0的值
tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)
# 默认的均值为0，标准差为1.0
tf.truncated_normal([2, 2])

# --------------variable对象---------------------------------------
# Tensor对象和Op对象都是不可变的（immutable）
my_var = tf.Variable(3, name='my_variable')
a = tf.add(5, my_var)
# 初始化全部variable对象
init = tf.initialize_all_variables()
# # 初始化指定的对象
# tf.initialize_variables([my_var], name='init_my_var')

with tf.Session() as sess:
    sess.run(init)
# 修改variable对象
my_var_update = my_var.assign(my_var * 2)
# 如果想重置variable对象为初始值，可再次调用初始化方法

# 在训练机器学习模型的Optimizer类中，会自动修改variable对象的值，如果只允许手动修改，可设置trainable参数为false
my_var = tf.Variable(3, name='my_variable', trainable=False)


# ----------------name scope------------------------------
with tf.name_scope('scope_A'):
    a = tf.add(1, 2, name='add_a')
    b = tf.add(2, 2, name='add_b')

with tf.name_scope('scope_B'):
    c = tf.add(1, 2, name='add_c')
    d = tf.add(2, 2, name='add_d')

e = tf.add(b, d, name='output')

with tf.Session() as sess:
    sess.run(e)
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    writer.close()









