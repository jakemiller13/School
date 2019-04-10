# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:33:38 2019

@author: jmiller
"""

import gzip, os
import sklearn
from sklearn.datasets import load_digits
from sklearn import svm
import numpy as np
import sys
import warnings

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve
    
#################################
# Utility functions from week 3 #
#################################

# Function that downloads a specified MNIST data file from Yann Le Cun's website
def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

# Invokes download() if necessary, then reads in images
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# Load the training set
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')

# Load the testing set
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

# Create and train Linear SVM Classifier
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
iterations = 1000
for value in C_values:
    warnings.filterwarnings('ignore')
    
    svm_clf = svm.LinearSVC(loss = 'hinge',
                            C = value,
                            max_iter = iterations,
                            random_state = 42)
    svm_clf.fit(train_data, train_labels)
    
    svc_clf = svm.SVC(kernel = 'poly',
                      degree = 2,
                      C = value,
                      gamma = 'auto',
                      max_iter = iterations,
                      random_state = 42)
    svc_clf.fit(train_data, train_labels)
    
    print('\n-- TRAIN Error Using C-Value of [{}] --'.format(value))
    print('Linear SVM classifier: {}'.format(svm_clf.score(train_data,
                                                           train_labels)))
    print('Kernel SVM classifier: {}'.format(svc_clf.score(train_data,
                                                           train_labels)))
    
    print('\n-- TEST Error Using C-Value of [{}] --'.format(value))
    print('Linear SVM classifier: {}'.format(svm_clf.score(test_data,
                                                           test_labels)))
    print('Kernel SVM classifier: {}'.format(svc_clf.score(test_data,
                                                           test_labels)))
    
    warnings.filterwarnings('default')