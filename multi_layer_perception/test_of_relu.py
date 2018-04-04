#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import tensorflow as tf

in_mat = [2, 3, 0, -1, -2]
relu = tf.nn.relu(in_mat)
print tf.Session().run(relu)
# 输出 [2 3 0 0 0]
# 将全部负数都转换为0
