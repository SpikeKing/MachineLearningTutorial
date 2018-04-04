#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

mat = [[1., 2.], [3., 4.]]

rm = tf.reduce_mean(mat)
rm_col = tf.reduce_mean(mat, 0)  # 保留行
rm_row = tf.reduce_mean(mat, 1)  # 保留列

sess = tf.Session()
print sess.run(rm)
print sess.run(rm_col)
print sess.run(rm_row)

# 2.5
# [ 2.  3.]
# [ 1.5  3.5]
