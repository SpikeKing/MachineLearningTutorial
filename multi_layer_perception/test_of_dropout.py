#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import tensorflow as tf

in_mat = [2., 3., 2., 1., 4., 2., 3.]
keep_prop = 0.4
print tf.Session().run(tf.nn.dropout(in_mat, keep_prop))
"""
输出：[  5.    7.5   0.    2.5  10.    0.    7.5]
keep_prop 随机保留40%的数据，并且将所有值增加2.5倍，2.5*0.4=1；
如果keep_prop是0.5，则所有值增加2倍；
"""
