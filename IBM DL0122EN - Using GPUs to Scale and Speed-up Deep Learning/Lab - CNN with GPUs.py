#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 07:19:01 2018

@author: Jake
"""

import tensorflow as tf

import os
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
tf.logging.set_verbosity(old_v)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config = config)

# Input images
width = 28
height = 28
flat = width * height
class_output = 10

x  = tf.placeholder(tf.float32, shape = [None, flat])
y_ = tf.placeholder(tf.float32, shape = [None, class_output])

x_image = tf.reshape(x, [-1,28,28,1])  

# Convolutional Layer 1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = 0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape = [32]))
convolve1 = tf.nn.conv2d(x_image,
                         W_conv1,
                         strides = [1, 1, 1, 1],
                         padding = 'SAME') + b_conv1