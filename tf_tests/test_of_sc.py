# -*-coding: utf-8-*-#

# Created by C.L.Wang

import os  # 避免Warning

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

labels_ = [[0., 1.], [1., 0.]]
labels = [[0.1, 0.9], [0.8, 0.2]]

sc = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=labels)

sess = tf.Session()

print sess.run(sc)

sc2 = tf.reduce_sum(-1 * (labels_ * tf.log(tf.nn.softmax(labels))), reduction_indices=[1])

print sess.run(sc2)

# [ 0.37110069  0.43748799]
# [ 0.37110075  0.43748796]
