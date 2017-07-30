#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from utils import load_pickle, num_channels, num_labels
import trainer

pickle_file = 'notMNIST.pickle'
dic = load_pickle(pickle_file, twoD=True)

num_hidden = 96


graph = tf.Graph()
with graph.as_default():
    # Variables.
    W1 = tf.Variable(tf.truncated_normal([5, 5, num_channels, 16], stddev=0.1))
    B1 = tf.Variable(tf.zeros([16]))
    W21 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    W22 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    W222 = tf.Variable(tf.truncated_normal([3, 3, 8, 16], stddev=0.1))
    W23 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    W232 = tf.Variable(tf.truncated_normal([5, 5, 8, 16], stddev=0.1))
    W242 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    B2 = tf.Variable(tf.zeros([48]))
    W3 = tf.Variable(tf.truncated_normal(
        [4 * 4 * 48, num_hidden],
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
        conv21 = tf.nn.conv2d(hidden, W21, [1, 2, 2, 1], padding='SAME')
        conv22 = tf.nn.conv2d(hidden, W22, [1, 1, 1, 1], padding='SAME')
        conv222 = tf.nn.conv2d(conv22, W222, [1, 2, 2, 1], padding='SAME')
        conv23 = tf.nn.conv2d(hidden, W23, [1, 1, 1, 1], padding='SAME')
        conv232 = tf.nn.conv2d(conv23, W232, [1, 2, 2, 1], padding='SAME')
        pool2 = tf.nn.max_pool(hidden,
                                [1, 3, 3, 1],
                                [1, 2, 2, 1],
                                padding='SAME')
        conv242 = tf.nn.conv2d(pool2, W242, [1, 1, 1, 1], padding='SAME')
        hidden = tf.concat([conv21, conv222, conv232, conv242], axis=3)
        hidden = tf.nn.relu(hidden + B2)
        print hidden.shape
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]
                                      ])
        hidden = tf.nn.relu(tf.matmul(reshape, W3) + B3)
        print hidden.shape
        return tf.matmul(hidden, W4) + B4

trainer.run(dic, graph, model, num_steps=2001)

