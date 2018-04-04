#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang
"""
简易的卷积神经网络（CNN）
"""
import os
import sys

import tensorflow as tf

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from tensorflow.examples.tutorials.mnist import input_data

from root_dir import ROOT_DIR, MNIST_DIR


def test_simple_cnn():
    """
    测试简易的CNN
    :return: None
    """
    mnist_path = os.path.join(ROOT_DIR, MNIST_DIR)
    mnist = input_data.read_data_sets(mnist_path, one_hot=True)  # 读取MNIST数据
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784])  # 28*28维的输入图像
    y_ = tf.placeholder(tf.float32, [None, 10])  # 10维的输入标签
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 将784维数据转换为28*28维的图片, -1表示数量不固定, 1表示图片的维度为1

    # print mnist.test.images[0]  # 测试读取数据

    # ************ 第一层卷积操作 ************
    W_conv1 = weight_var([5, 5, 1, 32])  # 5*5的卷积核, 1个颜色通道, 32个不同的核
    b_conv1 = bias_var([32])  # 32维的偏移
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 在x_image中执行卷积操作, 再使用ReLU的激活函数
    h_pool1 = max_pool_2x2(h_conv1)  # 执行最大的池化操作

    # ************ 第二层卷积操作 ************
    W_conv2 = weight_var([5, 5, 32, 64])  # 第二层卷积, 核5*5, 32个通道(与第一层对应), 64个不同核
    b_conv2 = bias_var([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二层卷积
    h_pool2 = max_pool_2x2(h_conv2)  # 执行最大的池化操作

    # ************ 全连接层 ************
    W_fc1 = weight_var([7 * 7 * 64, 1024])  # 隐含节点, 1024个
    b_fc1 = bias_var([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 展开
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 全连接层

    # ************ Dropout ************
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)  # 随机遗忘

    # ************ SoftMax ************
    W_fc2 = weight_var([1024, 10])
    b_fc2 = weight_var([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # ************ 训练步骤, 最小化交差熵 ************
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)  # 最小化交差熵

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 平均值

    tf.global_variables_initializer().run()  # 初始化变量

    # ************ Feed数据, 训练模型, 验证 ************
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


def weight_var(shape):
    """
    wx + b 的w, 权重
    :param shape: w的维度
    :return: tf的var
    """
    initial = tf.truncated_normal(shape=shape, stddev=0.1)  # 截断的正态分布
    return tf.Variable(initial_value=initial)


def bias_var(shape):
    """
    wx + b 的b, 偏移
    :param shape: b的维度
    :return: tf的var
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


def conv2d(x, W):
    """
    二维卷积操作, 步长为1, 维度不变
    :param x: 输入矩阵
    :param W: 卷积参数, 如[5, 5, 1, 32], 卷积维度5*5, 1维, 32个卷积核 
    :return: 卷积之后的矩阵
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    2x2的核维度, 步长是横向和竖向2步, 全部维度
    :param x: 输入矩阵
    :return: 池化之后的矩阵
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    test_simple_cnn()  # 测试简易的CNN
