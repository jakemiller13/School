#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 06:36:32 2018

@author: Jake
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some helper functions for plotting and drawing lines

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

# Reading and plotting the data

data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
plot_points(X,y)
plt.show()

##### Implement the following functions #####

# Activation (sigmoid) function
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Output (prediction) formula y^=σ(w1x1+w2x2+b)
def output_formula(features, weights, bias):
    ##### delete print statements #####
    print('features:\n', features)
    print('weights:\n', weights)
    output = np.matmul(features, weights.T) + bias
    print('output =\n', output)
    if output >= 1:
        print('output: 1')
        return 1
    print('output: 0')
    return 0

# Error (log-loss) formula  Error(y,ŷ )=−ylog(ŷ )−(1−y)log(1−ŷ)
def error_formula(y, output):
    return (-y * np.log(output) - (1 - y) * np.log(1 - output))

# Gradient descent step
# wi⟶wi+α(y−y^)xi
# b⟶b+α(y−ŷ 
def update_weights(x, y, weights, bias, learnrate):
    ##### delete counter #####
    counter3 = 0
    for i in range(len(x)):
        ##### delete counter #####
        print('counter3:', counter3)
        counter3 += 1
        y_hat = output_formula(x, weights, bias) # deleted x[o] here
        ##### delete print statement #####
        print('y:', y)
        if y - y_hat == 1:
            weights[0] += x[0] * learnrate
            weights[1] += x[1] * learnrate
            bias += learnrate
        elif y - y_hat == -1:
            weights[0] -= x[0] * learnrate
            weights[1] -= x[1] * learnrate
            bias -= learnrate
        return (weights, bias)

##########

np.random.seed(44)

epochs = 100
learnrate = 0.01

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    ##### delete counter #####
    counter1 = 0
    for e in range(epochs):
        ##### delete counter #####
        print('counter1:', counter1)
        counter1 += 1
        del_w = np.zeros(weights.shape)
        
        ##### delete counter #####
        counter2 = 0
        for x, y in zip(features, targets):
            ##### delete counter #####
            print('counter2:', counter2)
            counter2 += 1
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()

##########

train(X, y, epochs, learnrate, True)