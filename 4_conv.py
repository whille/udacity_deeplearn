#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from utils import accuracy, load_pickle

pickle_file = 'notMNIST.pickle'
dic = load_pickle(pickle_file, twoD=True)


image_size = 28
num_labels = 10
num_channels = 1 # grayscale
batch_size = 16
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(dic['valid_dataset'])
  tf_test_dataset = tf.constant(dic['test_dataset'])

  # Variables.
  W1 = tf.Variable(tf.truncated_normal([5, 5, num_channels, depth], stddev=0.1))
  B1 = tf.Variable(tf.zeros([depth]))
  W2 = tf.Variable(tf.truncated_normal([5, 5, depth, depth], stddev=0.1))
  B2 = tf.Variable(tf.constant(1.0, shape=[depth]))
  W3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  B3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  W4 = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  B4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, W1, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + B1)
    print hidden.shape

    #conv = tf.nn.conv2d(hidden, W2, [1, 2, 2, 1], padding='SAME')
    #hidden = tf.nn.relu(conv + layer2_biases)
    # use max_pool instead
    hidden = tf.nn.max_pool(hidden, [1,3,3,1], [1,2,2,1], padding='SAME')
    print hidden.shape

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, W3) + B3)

    print hidden.shape
    return tf.matmul(hidden, W4) + B4

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

  num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (dic['train_labels'].shape[0] - batch_size)
    #print step, batch_size, offset
    batch_data = dic['train_dataset'][offset:(offset + batch_size), :, :, :]
    batch_labels = dic['train_labels'][offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), dic['valid_labels']))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), dic['test_labels']))
