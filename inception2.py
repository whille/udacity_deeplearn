#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from utils import num_channels, num_labels, image_size
import trainer

keep_prob = 0.5
reduce1 = 8
reduce2 = 16
map1 = 32
map2 = 64
num_fc1 = 512


def create_weight(size):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1))


def create_bias(size):
    return tf.Variable(tf.constant(0.1, shape=size))


def conv2d_s(x, W, s=1):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def max_pool_3x3_s(x, s=1):
    return tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, s, s, 1],
                          padding='SAME')


def create_inception_vars(in_num, reduce1x1, mapn):
    W1x1_1 = create_weight([1, 1, in_num, mapn])
    W1x1_2 = create_weight([1, 1, in_num, reduce1x1])
    W1x1_3 = create_weight([1, 1, in_num, reduce1x1])
    W1x1_4 = create_weight([1, 1, in_num, mapn])
    W1x3 = create_weight([1, 3, reduce1x1, reduce1x1])
    W3x1 = create_weight([3, 1, reduce1x1, mapn])
    W1x5 = create_weight([1, 5, reduce1x1, reduce1x1])
    W5x1 = create_weight([5, 1, reduce1x1, mapn])
    B1x1 = tf.Variable(tf.zeros([mapn]))
    B1x1_2 = tf.Variable(tf.zeros([reduce1x1]))
    B1x1_3 = tf.Variable(tf.zeros([reduce1x1]))
    B3x3 = tf.Variable(tf.zeros([mapn]))
    B5x5 = tf.Variable(tf.zeros([mapn]))
    B1x1_4 = tf.Variable(tf.zeros([mapn]))
    lst = (W1x1_1, W1x1_2, W1x1_3, W1x1_4, W1x3, W3x1, W1x5, W5x1, B1x1,
           B1x1_2, B1x1_3, B1x1_4, B3x3, B5x5)
    return lst


def model_inception(data, *args):
    (W1x1_1, W1x1_2, W1x1_3, W1x1_4, W1x3, W3x1, W1x5, W5x1, B1x1, B1x1_2,
     B1x1_3, B1x1_4, B3x3, B5x5) = args
    conv1 = conv2d_s(data, W1x1_1, s=2) + B1x1
    conv2 = tf.nn.relu(conv2d_s(data, W1x1_2, s=2) + B1x1_2)
    conv3x3 = conv2d_s(conv2d_s(conv2, W1x3, s=1), W3x1, s=1) + B3x3
    conv3 = tf.nn.relu(conv2d_s(data, W1x1_3, s=2) + B1x1_3)
    conv5x5 = conv2d_s(conv2d_s(conv3, W1x5, s=1), W5x1, s=1) + B5x5
    conv4 = conv2d_s(max_pool_3x3_s(data, s=2), W1x1_4, s=1) + B1x1_4
    incept = tf.nn.relu(tf.concat([conv1, conv3x3, conv5x5, conv4], axis=3))
    print incept.shape
    return incept



graph = tf.Graph()
with graph.as_default():
    # Model.
    def model(data, training=True):
        incept1 = model_inception(data, *incept_lst1)
        incept2 = model_inception(incept1, *incept_lst2)
        #flatten features for fully connected layer
        shape = incept2.get_shape().as_list()
        flat = tf.reshape(incept2, [-1, shape[1] * shape[2] * shape[3]])
        #Fully connected layers
        fc1 = tf.nn.relu(tf.matmul(flat, Wfc_1) + Bfc_1)
        print fc1.shape
        if training:
            fc1 = tf.nn.dropout(fc1, keep_prob)
        return tf.matmul(fc1, Wfc_2) + Bfc_2

    incept_lst1 = create_inception_vars(num_channels, reduce1, map1)
    incept_lst2 = create_inception_vars(4 * map1, reduce2, map2)
    Wfc_1 = create_weight([image_size * image_size * map2 / 4, num_fc1])
    Bfc_1 = create_bias([num_fc1])
    Wfc_2 = create_weight([image_size, image_size, map2])
    Wfc_2 = tf.Variable(tf.truncated_normal([num_fc1, num_labels], stddev=0.1))
    Bfc_2 = create_bias([num_labels])

# Test accuracy: 93.2% after 1001 steps
trainer.run(graph, model, batch_size=50, num_steps=1001)
