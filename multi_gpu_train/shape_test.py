#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import tensorflow as tf

var_1 = tf.get_variable(name='var_1', shape=[], initializer=tf.constant_initializer(0), trainable=False)
ph_1 = tf.placeholder(name='ph_1', shape=(1, None), dtype=tf.float32)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
rand_array = [[1, 2, 3]]

print sess.run(var_1, feed_dict={var_1: 2})  # 测试变量
print sess.run(ph_1, feed_dict={ph_1: [[1, 2, 3]]})  # 测试PlaceHolder
