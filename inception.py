#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from utils import num_channels, num_labels
import trainer

num_hidden = 96
keep_prob = 0.5

graph = tf.Graph()
with graph.as_default():
    # Variables.
    W1 = tf.Variable(tf.truncated_normal([5, 5, num_channels, 16], stddev=0.1))
    B1 = tf.Variable(tf.zeros([16]))
    W21 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    W22 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    W222 = tf.Variable(tf.truncated_normal([1, 3, 8, 8], stddev=0.1))
    W223 = tf.Variable(tf.truncated_normal([3, 1, 8, 16], stddev=0.1))
    W23 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    W232 = tf.Variable(tf.truncated_normal([1, 3, 8, 8], stddev=0.1))
    W233 = tf.Variable(tf.truncated_normal([3, 1, 8, 8], stddev=0.1))
    W234 = tf.Variable(tf.truncated_normal([1, 3, 8, 8], stddev=0.1))
    W235 = tf.Variable(tf.truncated_normal([3, 1, 8, 16], stddev=0.1))
    W242 = tf.Variable(tf.truncated_normal([1, 1, 16, 8], stddev=0.1))
    B2 = tf.Variable(tf.zeros([48]))
    W3 = tf.Variable(tf.truncated_normal(
        [4 * 4 * 48, num_hidden],
        stddev=0.1))
    B3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    W4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    B4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data, training=True):
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
        conv22 = tf.nn.conv2d(conv22, W222, [1, 1, 1, 1], padding='SAME')
        conv22 = tf.nn.conv2d(conv22, W223, [1, 2, 2, 1], padding='SAME')
        conv23 = tf.nn.conv2d(hidden, W23, [1, 1, 1, 1], padding='SAME')
        conv23 = tf.nn.conv2d(conv23, W232, [1, 1, 1, 1], padding='SAME')
        conv23 = tf.nn.conv2d(conv23, W233, [1, 1, 1, 1], padding='SAME')
        conv23 = tf.nn.conv2d(conv23, W234, [1, 1, 1, 1], padding='SAME')
        conv23 = tf.nn.conv2d(conv23, W235, [1, 2, 2, 1], padding='SAME')
        pool2 = tf.nn.max_pool(hidden,
                                [1, 3, 3, 1],
                                [1, 2, 2, 1],
                                padding='SAME')
        conv242 = tf.nn.conv2d(pool2, W242, [1, 1, 1, 1], padding='SAME')
        hidden = tf.concat([conv21, conv22, conv23, conv242], axis=3)
        hidden = tf.nn.relu(hidden + B2)
        print hidden.shape
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]
                                      ])
        hidden = tf.nn.relu(tf.matmul(reshape, W3) + B3)
        print hidden.shape
        if training:
            hidden = tf.nn.dropout(hidden, keep_prob)
        return tf.matmul(hidden, W4) + B4

# Test accuracy: 93.2% after 1001 steps
trainer.run(graph, model, num_steps=1001)

