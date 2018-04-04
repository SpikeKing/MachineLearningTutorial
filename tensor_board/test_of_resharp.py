#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import tensorflow as tf

in_mat = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
r_mat = tf.reshape(in_mat, [-1, 3, 3, 1])
print tf.Session().run(r_mat)
