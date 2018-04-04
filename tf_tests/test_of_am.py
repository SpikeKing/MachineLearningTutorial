# -*-coding: utf-8-*-#

# Created by C.L.Wang

import os  # 避免Warning

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
2
1
False
True
0.0, 1.0
'''

mat = [1, 2, 3, 2, 1]
mat2 = [1, 3, 2, 2, 1]
mat3 = [1, 3, 2, 2, 1]

am1 = tf.argmax(mat)
am2 = tf.argmax(mat2)
am3 = tf.argmax(mat3)

equal = tf.equal(am1, am2)
equal2 = tf.equal(am2, am3)

val = tf.cast(equal, tf.float32)
val2 = tf.cast(equal2, tf.float32)

sess = tf.Session()

print sess.run(am1)
print sess.run(am2)
print sess.run(equal)
print sess.run(equal2)
print '%s, %s' % (sess.run(val), sess.run(val2))
