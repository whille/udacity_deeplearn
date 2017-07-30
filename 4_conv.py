#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from utils import load_pickle, image_size, num_channels, num_labels
import trainer

pickle_file = 'notMNIST.pickle'
dic = load_pickle(pickle_file, twoD=True)

depth = 16
num_hidden = 64


graph = tf.Graph()
with graph.as_default():
    # Variables.
    W1 = tf.Variable(tf.truncated_normal([5, 5, num_channels, depth], stddev=0.1))
    B1 = tf.Variable(tf.zeros([depth]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, depth, depth], stddev=0.1))
    B2 = tf.Variable(tf.constant(1.0, shape=[depth]))
    W3 = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden],
        stddev=0.1))
    B3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    W4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    B4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, W1, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + B1)
        print hidden.shape
        hidden = tf.nn.max_pool(hidden,
                                [1, 3, 3, 1],
                                [1, 2, 2, 1],
                                padding='SAME')
        print hidden.shape

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]
                                      ])
        hidden = tf.nn.relu(tf.matmul(reshape, W3) + B3)
        print hidden.shape
        return tf.matmul(hidden, W4) + B4

trainer.run(dic, graph, model)
