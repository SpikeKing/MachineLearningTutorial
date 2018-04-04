#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import os  # 避免Warning

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mat = tf.zeros([2, 3])  # 第一维表示行，第二维表示列
print tf.Session().run(mat)