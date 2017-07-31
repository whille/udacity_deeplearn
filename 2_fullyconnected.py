#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from utils import load_pickle, accuracy


image_size = 28
num_hidden = 1024
num_labels = 10

pickle_file = 'notMNIST.pickle'

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
num_steps = 801

def main():
    dic = load_pickle(pickle_file)
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are attached to the graph.
        train_dataset = tf.constant(dic['train_dataset'][:train_subset, :])
        train_labels = tf.constant(dic['train_labels'][:train_subset])
        valid_dataset = tf.constant(dic['valid_dataset'])
        test_dataset = tf.constant(dic['test_dataset'])

        # Variables.
        # These are the parameters that we are going to be training.
        # The weight # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        W1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden]))
        B1 = tf.Variable(tf.zeros([num_hidden]))
        W2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
        B2 = tf.Variable(tf.zeros([num_labels]))

        """
        Turn the logistic regression example with SGD into a 1-hidden layer neural
        network with rectified linear units nn.relu() and 1024 hidden nodes. This
        model should improve your validation / test accuracy.
        """
        def model(X, dropout=True, keep_prob=0.5):
            logits= tf.matmul(X, W1) +B1
            hiddens = tf.nn.relu(logits)
            if dropout:
                hiddens = tf.nn.dropout(hiddens, keep_prob)
            logits= tf.matmul(hiddens, W2) + B2
            return logits

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = model(train_dataset)

        beta_1 = 0.1
        beta_2 = 0.1
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits)) + \
                                    beta_1 * tf.nn.l2_loss(W1) + beta_2 * tf.nn.l2_loss(W2)

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
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(valid_dataset, dropout=False))
        test_prediction = tf.nn.softmax(model(test_dataset, dropout=False))

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
