#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import tensorflow as tf

mat = [1., 2., 3.]  # 需要使用小数
# softplus: [ln(e^1 + 1), ln(e^2 + 1), ln(e^3 + 1)]
print tf.Session().run(tf.nn.softplus(mat))
