#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang
"""
三层神经网络，输入层、隐含层、输出层
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)  # Data Set
sess = tf.InteractiveSession()  # Session

in_units = 784  # input neuron dimen
h1_units = 300  # hide level neuron dimen

# hide level para
# truncated_normal 重新选择超过两个标准差的值
# 矩阵用大写字母，向量用小写字母
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))

# output para
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])  # 任意个in_units维的数
keep_prob = tf.placeholder(tf.float32)  # dropout的保留比率

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # 隐含层，校正线性单元：Rectified Linear Unit
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)  # 隐含层的dropout
y = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)  # 输出层

y_ = tf.placeholder(tf.float32, [None, 10])  # ground truth
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()  # 初始化全部变量

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 随机采样
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})  # Feed数据，并且训练

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))  # 评估数据
"""
输出结果：0.9811
"""