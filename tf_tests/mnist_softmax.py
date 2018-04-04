#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners

回归分析：https://zh.wikipedia.org/wiki/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90
反向传播：
https://zhuanlan.zhihu.com/p/25081671
https://zhuanlan.zhihu.com/p/25416673
随机梯度下降：
https://www.zhihu.com/question/28728418
https://www.zhihu.com/question/27012077
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None  # 全局变量


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)  # 加载数据源

    x = tf.placeholder(tf.float32, [None, 784])  # 数据输入，没有
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b  # Softmax Regression，softmax(y)

    y_ = tf.placeholder(tf.float32, [None, 10])  # 标签输入

    # cross_entropy = tf.reduce_mean(tf.reduce_sum(-1 * (y_ * tf.log(tf.nn.softmax(y))), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # 损失函数
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 优化器

    sess = tf.InteractiveSession()  # 交互会话
    tf.global_variables_initializer().run()  # 初始化变量

    # 训练模型
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 验证模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 设置参数data_dir
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
