#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:12:43 2018

@author: Jake
"""

'''
Exercise 4.3.3 from edX course DL0110EN:
Deep Learning with Python and PyTorch

Using the Sequential Constructor to test the Sigmoid, Tanh and Relu activation 
functions on the MNIST Dataset
'''

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# Load data
train_dataset = dsets.MNIST(root = './data',
                            train = True,
                            download = True,
                            transform = transforms.ToTensor())
validation_dataset = dsets.MNIST(root = './data',
                                 train = False,
                                 download = True,
                                 transform = transforms.ToTensor())
train_loader  = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = 2000,
                                            shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset,
                                                batch_size = 5000,
                                                shuffle = False)

# Loss function
criterion = nn.CrossEntropyLoss()

def train(model, criterion, train_loader, validation_loader,
          optimizer, epochs = 100):
    '''
    Neural network custom module and training
    '''
    i = 0
    useful_stuff = {'training_loss' : [], 'validation_accuracy' : []}
    
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
            
        correct = 0
        
        for x, y in validation_loader:
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()
        
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff

# Network dimensions, learning rate, optimizer
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
learning_rate = 0.01

# Sigmoid activation function
model_sig = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                          nn.Sigmoid(),
                          nn.Linear(hidden_dim, output_dim))

optimizer_sig = torch.optim.SGD(model_sig.parameters(), lr = learning_rate)
training_results_sig = train(model_sig, criterion, train_loader,
                             validation_loader, optimizer_sig, epochs = 30)

# Tanh activation function
model_tanh = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                           nn.Tanh(),
                           nn.Linear(hidden_dim, output_dim))

optimizer_tanh = torch.optim.SGD(model_tanh.parameters(), lr = learning_rate)
training_results_tanh = train(model_tanh, criterion, train_loader,
                              validation_loader, optimizer_tanh, epochs = 30)

# Relu activation function
model_relu = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                           nn.ReLU(),
                           nn.Linear(hidden_dim, output_dim))

optimizer_relu = torch.optim.SGD(model_relu.parameters(), lr = learning_rate)
training_results_relu = train(model_relu, criterion, train_loader,
                              validation_loader, optimizer_relu, epochs = 30)

# Training loss - copied directly from lab
plt.plot(training_results_tanh['training_loss'],label='tanh')
plt.plot(training_results_sig['training_loss'],label='sigmoid')
plt.plot(training_results_relu['training_loss'],label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
plt.show()

# Validation loss - copied directly from lab
plt.plot(training_results_tanh['validation_accuracy'],label='tanh')
plt.plot(training_results_sig['validation_accuracy'],label='sigmoid')
plt.plot(training_results_relu['validation_accuracy'],label='relu') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs')
plt.title('validation loss iterations')
plt.legend()
plt.show()