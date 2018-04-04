#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2.cv2 as cv2
# Created by C.L.Wang
#
import matplotlib
import scipy.misc as misc
from PIL import Image

matplotlib.use('TkAgg')

from project_utils import *
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
images = mnist.test.images  # 图片
labels = mnist.test.labels  # 标签

# 存储图片
size = len(labels)
for i in range(size):
    pxl = np.array(images[i])  # 像素
    img = pxl.reshape((28, 28))  # 图片
    lbl = np.argmax(labels[i])  # 标签
    misc.imsave('./IMAGE_data/test/' + str(i) + '_' + str(lbl) + '.png', img)  # scipy的存储模式
    if i == 100:
        break

# 合并图片
large_size = 28 * 10
large_img = Image.new('RGBA', (large_size, large_size))
paths_list, _, __ = listdir_files('./IMAGE_data/test/')
for i in range(100):
    img = Image.open(paths_list[i])
    loc = ((int(i / 10) * 28), (i % 10) * 28)
    large_img.paste(img, loc)
large_img.save('./IMAGE_data/merged.png')

# 其他的图片存储方式
pixel = np.array(images[0])  # 784维的数据
label = np.argmax(labels[0])  # 找到标签
image = pixel.reshape((28, 28))  # 转换成28*28维的矩阵

# -------------------- scipy模式 -------------------- #
misc.imsave('./IMAGE_data/scipy.png', image)  # scipy的存储模式
# -------------------- scipy模式 -------------------- #

# -------------------- matplotlib模式 -------------------- #
plt.gray()  # 转变为灰度图片
plt.imshow(image)
plt.savefig("./IMAGE_data/plt.png")
# plt.show()
# -------------------- matplotlib模式 -------------------- #

# -------------------- opencv模式 -------------------- #
image = image * 255  # 数据是0~1的浮点数
cv2.imwrite("./IMAGE_data/opencv.png", image)
# cv2.imshow('hah', pixels)
# cv2.waitKey(0)
# -------------------- opencv模式 -------------------- #
