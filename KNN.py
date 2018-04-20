#from __future__ import print_function
import numpy as np
import tensorflow as tf

#输入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#限制MNIST数据，训练集5000，测试集200
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

#tensorflow graph 输入
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

#采用L1曼哈顿距离，计算曼哈顿距离
#distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
#采用L2距离，欧几里得距离
distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)), 2), reduction_indices=1))
#得到最小距离的编号
pred = tf.argmin(distance, 0)

accuracy = 0.
#初始化变量
init = tf.global_variables_initializer()

#开始测试
with tf.Session() as sess:

    sess.run(init)

    for i in range(len(Xte)):
        #获取最近邻
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        #将得到的最近邻标签与真实值比较
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))
        #计算准确率
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
