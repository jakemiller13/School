#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:39:00 2018

@author: Jake
"""

# Imports packages and modules
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import argparse

# Create parser
parser = argparse.ArgumentParser(description = 'Image Classifier')
parser.add_argument('--data_dir', default = 'flowers', type = str, help = 'Directory of training/validation/test images')
parser.add_argument('--save_dir', default = 'checkpoint.pth', type = str, help = 'Location to save checkpoint to')
parser.add_argument('--arch', default = 'vgg19', type = str, help = 'Torchvision model')
parser.add_argument('--learning_rate', default = 0.001, type = float, help = 'Learning rate for model')
parser.add_argument('--hidden_units', default = 1000, help = 'Number of hidden layers. Default is 1000. Can be list or int')
parser.add_argument('--epochs', default = 3, type = int, help = 'Number of epochs to run training on model')
parser.add_argument('--device', default = 'cuda', type = str, help = 'Device to train model on')
args = parser.parse_args()
data_dir, save_dir, arch, learning_rate, hidden_units, epochs, device = args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.device

# Define image directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms
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

# Load images/datasets
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

# Define dataloaders
trainloader = DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = DataLoader(valid_data, batch_size = 32)
testloader = DataLoader(test_data, batch_size = 32)

# Load category mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build model based on pretrained network
model = models.vgg19(pretrained = True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False
    
# Replace model classifier with self-defined classifier
classifier_dictionary = OrderedDict([('fc1', nn.Linear(25088, 12000)),
                                     ('relu', nn.ReLU(inplace = True)),
                                     ('dropout', nn.Dropout()),
                                     ('fc2', nn.Linear(12000, 102)),
                                     ('output', nn.LogSoftmax(dim = 1))])

classifier = nn.Sequential(classifier_dictionary)

model.classifier = classifier

# Constants
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
print_every = 20
steps = 0

# Learning function
def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    '''
    Does deep learning based on model type, criterion, optimizer defined above
    Returns nothing
    '''
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    train_loss, train_accuracy = validation(model, trainloader, criterion)
                    test_loss, test_accuracy = validation(model, validloader, criterion)
                    
                print('Epoch: {}/{}...'.format(e+1, epochs),
                      'Training Loss: {:.4f}'.format(train_loss/len(trainloader)),
                      'Training Accuracy: {:.4f}'.format(train_accuracy/len(trainloader)),
                      'Validation Loss: {:.4f}'.format(test_loss/len(validloader)),
                      'Validation Accuracy: {:.4f}'.format(test_accuracy/len(validloader)))
                
                running_loss = 0
                model.train()
            
    print('Training Complete')

# Accuracy function
def check_accuracy_on_test(model, testloader):
    '''
    Checks accuracy of trained model on testloader
    Returns nothing
    '''
    correct = 0
    total = 0
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save checkpoint function 
def save_checkpoint(input_size, output_size, classifier_dict, state_dict, class_to_idx):
    '''
    Saves checkpoint of trained model so training does not have to be repeated
    Returns nothing
    '''
    checkpoint = {'Input Size' : input_size,
                  'Output Size' : output_size,
                  'Classifier Dictionary' : classifier_dict,
                  'State Dict' : state_dict,
                  'Class to Index' : class_to_idx}
    
    torch.save(checkpoint, 'checkpoint.pth')

# Validation function
def validation(model, testloader, criterion):
    '''
    Performs training validation on testloader (validloader) set
    Returns test_loss, accuracy - for use in training
    '''
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        probs = torch.exp(output)
        equality = (labels.data == probs.max(dim = 1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

# Train model
do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device)

# Do validation on the test set
check_accuracy_on_test(model, validloader)

# Save checkpoint
save_checkpoint(input_size = 25088, output_size = 102, classifier_dict = classifier_dictionary, state_dict = model.state_dict(), class_to_idx = train_data.class_to_idx)