#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf

'''
在大多数TensorFlow程序中，只使用默认数据流图就足够了。
然而，如果需要定义多个相互之间不存在依赖关系的模型，则创建多个Graph对象十分有用。
当需要在单个文件中定义多个数据流图时，最佳实践是不使用默认数据流图，或为其立即分配句柄。
这样可以保证各节点按照一致的方式添加到每个数据流图中。
'''

# 创建一个新的数据流图，Graph 对象
g = tf.Graph()
# Graph.as_default()方法访问其上下文管理器，为其添加Op
with g.as_default():
    a = tf.add(2, 3)
    b = tf.multiply(a, 3)

# 如果不手动创建数据流图， tensorflow库被加载时会自动创建一个Graph对象，并将其作为默认的数据流图
g_default = tf.get_default_graph()
with g_default.as_default():
    c = tf.multiply(2, 3)


# 指定数据流图
sess = tf.Session(graph=g_default)
output = sess.run(c)
print(output)

# writer = tf.summary.FileWriter('./my_graph', sess.graph)
# # 在对应的虚拟环境下运行 tensorboard --logdir=.\src\my_graph ，在浏览器中访问 6006端口可查看数据流图
# writer.close()
sess.close()
