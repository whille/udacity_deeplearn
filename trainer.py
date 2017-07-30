#!/usr/bin/env python
# encoding: utf-8

from utils import accuracy, image_size, num_channels, num_labels


def run(dic, graph, model, batch_size=16, num_steps=1001):
    import tensorflow as tf

    with graph.as_default():
        # Input data.
        train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        valid_dataset = tf.constant(dic['valid_dataset'])
        test_dataset = tf.constant(dic['test_dataset'])
        # Training computation.
        logits = model(train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(valid_dataset))
        test_prediction = tf.nn.softmax(model(test_dataset))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (
                dic['train_labels'].shape[0] - batch_size)
            batch_data = dic['train_dataset'][offset:(offset + batch_size), :, :, :]
            batch_labels = dic['train_labels'][offset:(offset + batch_size), :]
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction],
                feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), dic['valid_labels']))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), dic['test_labels']))
