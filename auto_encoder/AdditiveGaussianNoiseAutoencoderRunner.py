#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from PIL import Image
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data

from autoencoder_models.DenoisingAutoencoder import AdditiveGaussianNoiseAutoencoder
from project_utils import listdir_files

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)  # 随机获取区块
    return data[start_index:(start_index + batch_size)]  # batch_size大小的区块
pi

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(
    n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

en_images = autoencoder.reconstruct(mnist.test.images)
labels = mnist.test.labels  # 标签

for i in range(len(en_images)):
    img = np.array(en_images[i]).reshape((28, 28))
    lbl = np.argmax(labels[i])
    misc.imsave('./IMAGE_data/en_test/' + str(i) + '_' + str(lbl) + '.png', img)  # scipy的存储模式
    if i == 100:
        break

paths_list, _, __ = listdir_files('./IMAGE_data/en_test/')
large_size = 28 * 10
large_img = Image.new('RGBA', (large_size, large_size))
for i in range(100):
    img = Image.open(paths_list[i])
    loc = ((int(i / 10) * 28), (i % 10) * 28)
    large_img.paste(img, loc)
large_img.save('./IMAGE_data/en_merged.png')
