#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 06:58:26 2018

@author: Jake
"""

# import NumPy into Python
import numpy as np

# Create a 1000 x 20 ndarray with random integers in the half-open interval [0, 5001).
X = np.random.randint(5001, size = (1000, 20))

# print the shape of X
print('X:', X)

# Average of the values in each column of X
ave_cols = np.average(X, axis = 0)

# Standard Deviation of the values in each column of X
std_cols = np.std(X, axis = 0)

# Print the shape of ave_cols
print('ave_cols shape:', ave_cols.shape)

# Print the shape of std_cols
print('std_cols shape:', std_cols.shape)

# Mean normalize X
X_norm = (X - ave_cols) / std_cols

# Print the average of all the values of X_norm
print('Average of X_norm: ', round(np.average(X_norm), 4))

# Print the average of the minimum value in each column of X_norm
print('Average of Min:', np.amin(X_norm, axis = 0) / X.shape[0])

# Print the average of the maximum value in each column of X_norm
print('Average of Max:', np.amax(X_norm, axis = 0) / X.shape[0])

#########################
#### Data Separation ####
#########################

# Create a rank 1 ndarray that contains a random permutation of the row indices of `X_norm`
row_indices = np.random.permutation(X.shape[0])

# Create a Training Set
X_train = X[row_indices[0 : int(X.shape[0] * 0.6)]]

# Create a Cross Validation Set
X_crossVal = X[row_indices[int(X.shape[0] * 0.6) : int(X.shape[0] * 0.8)]]

# Create a Test Set
X_test = X[row_indices[int(X.shape[0] * 0.8) : X.shape[0]]]

# Print the shape of X_train
print('X_train shape:', X_train.shape)

# Print the shape of X_crossVal
print('X_crossVal shape:', X_crossVal.shape)

# Print the shape of X_test
print('X_test shape:', X_test.shape)