#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:15:59 2018

@author: Jake
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile
import io

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Download and extract zipped movie file
movie_file = requests.get(
             'http://files.grouplens.org/datasets/movielens/ml-1m.zip')
z = zipfile.ZipFile(io.BytesIO(movie_file.content))
z.extractall()

# Load movies and ratings dataframes
movies_df = pd.read_csv('./ml-1m/movies.dat',
                        sep = '::',
                        header = None,
                        engine = 'python')
movies_df.columns = ['MovieID', 'Title', 'Genres']

ratings_df = pd.read_csv('./ml-1m/ratings.dat',
                         sep = '::',
                         header = None,
                         engine = 'python')
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Pivots ratings_df to ['UserID' x 'MovieID']
user_rating_df = ratings_df.pivot(index = 'UserID',
                                  columns = 'MovieID',
                                  values = 'Rating')

# Normalize user ratings
norm_user_rating_df = user_rating_df.fillna(0) / 5.0
trX = norm_user_rating_df.values

# Set model parameters
visibleUnits = len(user_rating_df.columns)
hiddenUnits = 20
vb = tf.placeholder(tf.float32, [visibleUnits])
hb = tf.placeholder(tf.float32, [hiddenUnits])
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])

# Phase 1: Input Processing
v0 = tf.placeholder(tf.float32, [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# Set training parameters
alpha = 1.0

# Contrastive Divergence
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Update weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Error function - MSE
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

# Initialize all variables (CURrent/PReVious, Visible/Hidden, Weights/Biases)
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
cur_vb = np.zeros([visibleUnits], np.float32)
cur_hb = np.zeros([hiddenUnits], np.float32)
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
prv_vb = np.zeros([visibleUnits], np.float32)
prv_hb = np.zeros([hiddenUnits], np.float32)

# Create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train model
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize),
                          range(batchsize, len(trX), batchsize)):
        batch = trX[start : end]
        cur_w = sess.run(update_w, feed_dict = {v0: batch,
                                                W: prv_w,
                                                vb: prv_vb,
                                                hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict = {v0: batch,
                                                  W: prv_w,
                                                  vb: prv_vb,
                                                  hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict = {v0: batch,
                                                  W: prv_w,
                                                  vb: prv_vb,
                                                  hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict = {v0: trX,
                                                 W: cur_w,
                                                 vb: cur_vb,
                                                 hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

# Create prediction
mock_user_id = 215

# Normalized ratings for mock user
inputUser = trX[mock_user_id - 1].reshape(1, -1)

# Feeding in mock user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict = {v0: inputUser,
                                  W: prv_w,
                                  hb: prv_hb})
rec = sess.run(vv1, feed_dict = {hh0: feed,
                                 W: prv_w,
                                 vb: prv_vb})

# Ensure movies df only includes rated movies, adds/sorts RecommendationScore
scored_movies_df_mock = movies_df[movies_df['MovieID'].\
                                  isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.\
                                  assign(RecommendationScore = rec[0])
scored_movies_df_mock.sort_values(["RecommendationScore"],\
                                  ascending = False).head(20)

# Movies mock user has already watched
movies_df_mock = ratings_df[ratings_df['UserID'] == mock_user_id]

# Merge movies already watched with predicted scores
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock,
                                             on = 'MovieID',
                                             how = 'outer')

# Sort based on RecommendationScore
sorted_rec = merged_df_mock.sort_values(["RecommendationScore"],
                                        ascending = False)

# Recommend top 10 movies not seen
print(sorted_rec.loc[sorted_rec['Rating'].isna()].head(10))