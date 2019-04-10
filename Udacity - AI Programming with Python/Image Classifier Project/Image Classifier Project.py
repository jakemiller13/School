#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:04:20 2018

@author: Jake
"""

# Import packages
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
from collections import OrderedDict

# Load data
data_dir = '/Users/Jake/Documents/School Stuff/Udacity - AI Programming with Python/Image Classifier Project/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transformations
# TODO: Define your transforms for the training, VALIDATION, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load datasets
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# Define dataloaders
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = DataLoader(valid_data, batch_size = 32)
testloader = DataLoader(test_data, batch_size = 32)

# Load json and label map
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build and train network
model = models.vgg19(pretrained = True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
# Replace model classifier with self-defined classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 12000)),
                                        ('relu', nn.ReLU(inplace = True)),
                                        ('dropout', nn.Dropout())
                                        ('fc2', nn.Linear(12000, 102)),
                                        ('output', nn.LogSoftmax(dim = 1))]))
model.classifier = classifier

# Define constants
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
epochs = 3
print_every = 40
steps = 0

# Define functions
def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device = 'cpu'):
    
    steps = 0
#    model.to('cuda')
    
    for e in range(epochs):
        running_loss = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
#            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            
            if steps % print_every == 0:
                print('Epoch: {}/{}...'.format(e+1, epochs), 'Loss: {:.4f}'.format(running_loss/print_every))

def check_accuracy_on_test(model, testloader):
    correct = 0
    total = 0
#    model.to('cuda')
    
    with torch.no_grad():
        for inputs, labels in testloader:
#            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs) #should this be model.forward(inputs)?
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().items()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# TODO: Do validation on the test set


# Run functions
do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device = 'cpu')
print('--- Validation Set ---')
check_accuracy_on_test(model, validloader)
print()
print('--- Test Set ---')
check_accuracy_on_test(model, testloader)