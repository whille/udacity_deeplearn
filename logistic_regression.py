#!/usr/bin/env python
# encoding: utf-8

#import tensorflow as tf
#import numpy as np
from utils import load_pickle


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 1000

def main():
    pickle_file = 'notMNIST.pickle'
    dic = load_pickle(pickle_file, one_hot=False)

    from sklearn import linear_model
    logreg = linear_model.LogisticRegression()

    logreg.fit(dic['train_dataset'][:train_subset, :],
            dic['train_labels'][:train_subset])

    for k in ('valid', 'test'):
        print '%s score: %s' % (k, logreg.score(dic['%s_dataset' % k], dic[
            '%s_labels' % k]))
