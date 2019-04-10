#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 06:08:55 2018

@author: Jake
"""

##############################################################################
# IMPORTANT NOTE:                                                            #
# This code is aborting at runtime and crashing kernal with TensorFlow 1.12  #
# By downgrading to 1.10 it is able to run with many deprecation warnings    #
##############################################################################

# Import packages
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from utils import tile_raster_images
import matplotlib.pyplot as plt

# Load utility functions
import urllib.request
with urllib.request.urlopen("http://deeplearning.net/tutorial/code/utils.py") as url:
    response = url.read()
target = open('utils.py', 'w')
target.write(response.decode('utf-8'))
target.close()

# Biases - [v]isibile and [h]idden
v_bias = tf.placeholder("float", [7])
h_bias = tf.placeholder("float", [2])

# Weights between neurons
W = tf.constant(np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(7, 2)).astype(np.float32))

# Toy example
# Forward Pass
print('-- Toy Example Start --\n')
sess = tf.Session()
X = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
v_state = X
print ("Input: ", sess.run(v_state))

h_bias = tf.constant([0.1, 0.1])
print ("hb: ", sess.run(h_bias))
print ("w: ", sess.run(W))

# Calculate the probabilities of turning the hidden units on:
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)
print ("p(h|v): ", sess.run(h_prob))

# Draw samples from the distribution:
h_state = tf.nn.relu(tf.sign(h_prob - tf.random_uniform(tf.shape(h_prob))))
print ("h0 states:", sess.run(h_state))

# Backward Pass
vb = tf.constant([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
print ("b: ", sess.run(vb))
v_prob = sess.run(tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(W)) + vb))
print ("p(vi∣h): ", v_prob)
v_state = tf.nn.relu(tf.sign(v_prob - tf.random_uniform(tf.shape(v_prob))))
print ("v probability states: ", sess.run(v_state))

inp = sess.run(X)
print('Input:', inp)
print(v_prob[0])
v_probability = 1
for elm, p in zip(inp[0],v_prob[0]) :
    if elm ==1:
        v_probability *= p
    else:
        v_probability *= (1-p)
print('Probability Distribution:', v_probability)

print('\n-- Toy Example End --')
print('\n\n\n')
print('-- MNIST Start --')

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
                     mnist.test.images,mnist.test.labels

# Visible layer (784 pixels), hidden layer (50 neurons)
W = tf.placeholder("float", [784, 50])
vb = tf.placeholder("float", [784])
hb = tf.placeholder("float", [50])
v0_state = tf.placeholder("float", [None, 784])

# Hidden layer
h0_prob = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)
h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random_uniform(tf.shape(h0_prob))))

# Reconstruction
v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + vb) 
v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random_uniform(tf.shape(v1_prob))))

# Mean squared error
err = tf.reduce_mean(tf.square(v0_state - v1_state))

# The following is assuming k=1, i.e. last step of training
h1_prob = tf.nn.sigmoid(tf.matmul(v1_state, W) + hb)
h1_state = tf.nn.relu(tf.sign(h1_prob - tf.random_uniform(tf.shape(h1_prob))))

alpha = 0.01
W_Delta = tf.matmul(tf.transpose(v0_state), h0_prob) -\
                    tf.matmul(tf.transpose(v1_state), h1_prob)
update_w = W + alpha * W_Delta
update_vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
update_hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0)

# Initialize variables
cur_w = np.zeros([784, 50], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([50], np.float32)
prv_w = np.zeros([784, 50], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([50], np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Error of first run
print('\nError:',
      sess.run(err, feed_dict = {v0_state: trX, W: prv_w,
                                 vb: prv_vb, hb: prv_hb}))

#Parameters
epochs = 5
batchsize = 100
weights = []
errors = []

for epoch in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize),
                          range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict = {v0_state: batch, W: prv_w,
                                                vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict = {v0_state: batch, W: prv_w,
                                                  vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict = {v0_state: batch, W: prv_w,
                                                  vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(err, feed_dict = {v0_state: trX, W: cur_w,
                                                     vb: cur_vb, hb: cur_hb}))
            weights.append(cur_w)
    print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()

uw = weights[-1].T
print(uw)

tile_raster_images(X = cur_w.T, img_shape = (28, 28), tile_shape = (5, 10),\
                   tile_spacing = (1, 1))
image = Image.fromarray(tile_raster_images(X = cur_w.T,\
                                           img_shape = (28, 28),
                                           tile_shape = (5, 10),
                                           tile_spacing = (1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

image = Image.fromarray(tile_raster_images(X = cur_w.T[10:11],
                                           img_shape = (28, 28),
                                           tile_shape = (1, 1),
                                           tile_spacing = (1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

# Import destructed "3"
destructed_3 = urllib.request.urlretrieve(
    'https://ibm.box.com/shared/static/vvm1b63uvuxq88vbw9znpwu5ol380mco.jpg',
    'destructed_3')
img = Image.open('destructed_3')

# convert the image to a 1d numpy array
sample_case = np.array(img.convert('I').resize((28,28))).\
              ravel().reshape((1, -1))/255.0
              
hh0_p = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)
hh0_s = tf.round(hh0_p)
hh0_p_val,hh0_s_val  = sess.run((hh0_p, hh0_s),\
                                feed_dict = {v0_state: sample_case,
                                             W: prv_w,
                                             hb: prv_hb})
print("Probability nodes in hidden layer:" , hh0_p_val)
print("activated nodes in hidden layer:" , hh0_s_val)

# reconstruct
vv1_p = tf.nn.sigmoid(tf.matmul(hh0_s_val, tf.transpose(W)) + vb)
rec_prob = sess.run(vv1_p, feed_dict={ hh0_s: hh0_s_val, W: prv_w, vb: prv_vb})

img = Image.fromarray(tile_raster_images(X = rec_prob,
                                         img_shape = (28, 28),
                                         tile_shape = (1, 1),
                                         tile_spacing = (1, 1)))
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(img)
imgplot.set_cmap('gray') 