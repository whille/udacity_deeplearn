#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
from utils import load_pickle

image_size = 28
num_hidden = 1024
num_labels = 10

pickle_file = 'notMNIST.pickle'

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


"""
Turn the logistic regression example with SGD into a 1-hidden layer neural
network with rectified linear units nn.relu() and 1024 hidden nodes. This
model should improve your validation / test accuracy.
"""
def calc(X, w1, b1, w2, b2):
    logits_1 = tf.matmul(X, w1) +b1
    hiddens = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(hiddens, w2) + b2
    return logits_2


def main():
    dic = load_pickle(pickle_file)
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(dic['train_dataset'][:train_subset, :])
        tf_train_labels = tf.constant(dic['train_labels'][:train_subset])
        tf_valid_dataset = tf.constant(dic['valid_dataset'])
        tf_test_dataset = tf.constant(dic['test_dataset'])

        # Variables.
        # These are the parameters that we are going to be training.
        # The weight # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden]))
        biases_1 = tf.Variable(tf.zeros([num_hidden]))
        weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits_2 = calc(tf_train_dataset, weights_1, biases_1, weights_2, biases_2)
        beta_1 = 0.1
        beta_2 = 0.1
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2)) + \
                                    beta_1 * tf.nn.l2_loss(weights_1) + beta_2 * tf.nn.l2_loss(weights_2)

        # learning rate
        starter_learning_rate = 0.5
        global_step = tf.Variable(0, trainable=False)
        decay_step = 10
        decay_rate = 0.96
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        # Passing global_step to minimize() will increment it at each step.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits_2)
        valid_prediction = tf.nn.softmax(calc(tf_valid_dataset, weights_1, biases_1, weights_2, biases_2))
        test_prediction = tf.nn.softmax(calc(tf_test_dataset, weights_1, biases_1, weights_2, biases_2))

    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if (step % 10 == 0):
                print('Loss at step %d: %f' % (step, l))
                dic_out = {'train': predictions,
                    'valid':valid_prediction.eval(),
                    'test': test_prediction.eval()}
                for k in ('train', 'valid', 'test'):
                    print('%s accuracy: %.1f%%' %(k, accuracy(dic_out[k], dic['%s_labels' %k][:train_subset, :])))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.

if __name__ == '__main__':
    main()
