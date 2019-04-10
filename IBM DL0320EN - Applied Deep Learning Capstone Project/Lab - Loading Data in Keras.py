#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 07:29:21 2018

@author: Jake
"""
import wget
import tarfile
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from PIL import Image
import keras
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model

# Training Images
train_url = 'https://cocl.us/DL0320EN_TRAIN_TAR_KERAS'
train_file = 'train_data_keras.tar.gz'

# Validation Images
valid_url = 'https://cocl.us/DL0320EN_VALID_TAR_KERAS'
valid_file = 'validation_data_keras.tar.gz'

# Test Images
test_url = 'https://cocl.us/DL0320EN_TEST_TAR_KERAS'
test_file = 'test_data_keras.tar.gz'

def download_images(url, file_name):
    '''
    Downloads and extracts file if it is not in current directory
    '''
    if file_name not in os.listdir():
        print('Downloading and extracting: ' + file_name)
        wget.download(url, file_name)
        tar = tarfile.open(file_name, mode = 'r:gz')
        for member in tar.getmembers():
            if '._' in member.name:
                continue
            else:
                tar.extract(member)
        tar.close()

def plot_images(set_type, euro, image_number):
    '''
    set_type (str): "train", "test", "validation"
    euro (int): value of euro, given in problem
    image_number (int): is given in problem
    '''
    img = './' + set_type + '_data_keras/' + str(euro) + '/' +\
          str(image_number) + '.jpeg'
    plt.imshow(Image.open(img))
    plt.title(set_type.title() + ', \u20ac' + str(euro) + ', Image number: '\
              + str(image_number))
    plt.show()

def generator(set_type, TARGET_SIZE, BATCH_SIZE, CLASSES, RANDOM_SEED,
              SHUFFLE = True):
    '''
    set_type (str): "train", "test", "validation"
    returns: ImageDataGenerator().flow_from_directory
    '''
    data_dir = './' + set_type + '_data_keras'
    return ImageDataGenerator().flow_from_directory(data_dir,
                                                    target_size = TARGET_SIZE,
                                                    batch_size = BATCH_SIZE,
                                                    classes = CLASSES,
                                                    seed = RANDOM_SEED,
                                                    shuffle = SHUFFLE)

def plot_generated_images(generator_batch):
    '''
    Plots images generated from generator
    set_type (str): "train", "test", "validation"
    '''
    name_split = re.split('[^A-Za-z]', generator_batch.directory)
    title = ' '.join([i for i in name_split
                     if i != '' and i.lower() != 'keras']).title()
    for i in range(generator_batch.batch_size):
        plt.imshow(generator_batch.next()[0][i].astype(np.uint8))
        plt.title(title)
        plt.show()

def train_model(model, train_gen, valid_gen, optimizer, loss,
                metrics, n_epochs, steps, model_name = None):
    '''
    Checks if "model_name" exists
    If model DOES NOT exists, trains model and saves as "model_name"
    If model DOES exists, asks if you would like to load saved model
    "model_name" must end it ".pt"
    '''
    if model_name not in os.listdir():
        print('Training [' + model_name + ']...')
        model.compile(optimizer = optimizer,
                      loss = loss,
                      metrics = metrics)
        history = model.fit_generator(generator = train_gen,
                                      validation_data = valid_gen,
                                      steps_per_epoch = STEPS,
                                      epochs = N_EPOCHS,
                                      verbose = 2)
        model.save(model_name)
        return history
    else:
        print('Model [' + model_name + '] already exists. '\
              'Run "load_saved_model(' + model_name + ')"')

def load_saved_model(model_name):
    '''
    Returns model with "model_name.pt"
    '''
    return load_model(model_name)

def class_to_index(dictionary, generator_batch):
    '''
    Returns a dictionary with swapped keys/values
    '''
    class_to_idx = {}
    for index in dictionary.keys():
        class_to_idx[generator_batch.class_indices[index]] = index
    return class_to_idx

def make_prediction(predictions, generator_batch, image_number, classes):
    '''
    Returns a predicted Euro value
    '''
    prediction_index = np.argmax(predictions[image_number])
    prediction = class_to_index(generator_batch.class_indices,\
                                generator_batch)[prediction_index]
    compare_prediction(prediction, generator_batch, image_number, classes)
    return prediction

def compare_prediction(prediction, generator_batch, image_number, classes):
    '''
    Checks if prediction is in correct folder, i.e. if prediction is
    correctly classified.
    Returns True if correct
    '''
    for file in generator_batch.filenames:
        if file.split('/')[1][:-5] == str(image_number):
            correct = file.split('/')[0]
    plot_images('validation', correct, image_number)
    print('Predicted: \u20ac' + prediction)
    if prediction == correct:
        print('Correctly Classified!!')
    else:
        print('Misclassified...')

# Download images if not already downloaded
download_images(train_url, train_file)
download_images(valid_url, valid_file)
download_images(test_url, test_file)

'''
# Plot select images
plot_images('train', 5, 0)
plot_images('train', 200, 52)
plot_images('validation', 5, 0)
plot_images('validation', 50, 36)
'''

# Parameters for image dataset generators
TARGET_SIZE = (224, 224)
CLASSES = ['5', '10', '20', '50', '100', '200', '500']
RANDOM_SEED = 0

# Generate image datasets
BATCH_SIZE = 10
train_generator = generator('train', TARGET_SIZE, BATCH_SIZE,
                            CLASSES, RANDOM_SEED)
BATCH_SIZE = 5
validation_generator = generator('validation', TARGET_SIZE, BATCH_SIZE,
                                 CLASSES, RANDOM_SEED)

'''
# Plot resized images
plot_generated_images(train_generator)
plot_generated_images(validation_generator)
'''

# Load ResNet50, set all layer to non-trainable
base = ResNet50(weights = 'imagenet')
for layer in base.layers:
    layer.trainable = False

# Replace 1000 output classification layer with 7 outputs
sec_last_base = base.layers[-2].output
connected_model = Dense(len(CLASSES), activation = 'softmax')(sec_last_base)
base_input = base.input
model = Model(inputs = base_input, outputs = connected_model)

# Train model
N_EPOCHS = 20
STEPS = train_generator.n // train_generator.batch_size
history = train_model(model,
                      train_gen = train_generator,
                      valid_gen = validation_generator,
                      optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'],
                      n_epochs = N_EPOCHS,
                      steps = STEPS,
                      model_name = 'resnet50_keras.pt')

# Get training history
train_history = model.history.history

# Training/Validation Loss/Accuracy
x_range = [x + 1 for x in range(N_EPOCHS)]

# Training/Validation Loss
plt.plot(x_range, train_history['loss'], label = 'Loss')
plt.plot(x_range, train_history['val_loss'], label = 'Validation Loss')
plt.title('Training and Validation Loss vs. Epoch')
plt.xlabel('Epoch')
plt.xticks(x_range)
plt.ylabel('Loss')
plt.legend()
plt.show()

# Training/Validation Accuracy
plt.plot(x_range, train_history['acc'], label = 'Accuracy')
plt.plot(x_range, train_history['val_acc'], label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.xticks(x_range)
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Use validation data set with NO SHUFFLE
BATCH_SIZE = 5
test_valid_generator = generator('validation', TARGET_SIZE, BATCH_SIZE,
                                 CLASSES, RANDOM_SEED, SHUFFLE = False)

# Set up constants for question
random.seed(0)
numbers = [random.randint(0, 69) for i in range(0, 5)]

# Make prediction
predictions = model.predict_generator(test_valid_generator, verbose = 1)

for image in range(len(numbers)):
    make_prediction(predictions, test_valid_generator, 
                    numbers[image], CLASSES)