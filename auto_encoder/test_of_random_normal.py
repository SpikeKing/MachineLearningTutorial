#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import tensorflow as tf

rn = tf.random_normal((100000,))  # 一行，指定seed，防止均值的时候随机
mean, variance = tf.nn.moments(rn, 0)  # 计算均值和方差，预期均值约等于是0，方差是1
print tf.Session().run(tf.nn.moments(rn, 0))
