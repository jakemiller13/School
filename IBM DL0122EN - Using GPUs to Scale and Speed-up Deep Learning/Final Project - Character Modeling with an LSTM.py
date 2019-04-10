#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 07:00:25 2018

@author: Jake
"""

# NOTE: Most of this is not working locally because no GPU
# The code is copied over from Jupyter notebook running on PowerAI with GPU

import tensorflow as tf
import time
import codecs
import os
import collections
import wget
from six.moves import cPickle
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

data = 'https://ibm.box.com/shared/static/a3f9e9mbpup09toq35ut7ke3l3lf03hg.txt'
file_name = 'input.txt'

def download_data(url, file_name):
    '''
    Downloads file if it is not in current directory and saves as "file_name"
    '''
    if file_name not in os.listdir():
        print('Downloading: ' + file_name)
        wget.download(url, file_name)
    else:
        print('[' + file_name + '] already exists. Skipping...\n')

def peak_at_data(file_name):
    '''
    Prints 500 characters of "file_name"
    '''
    with open('input.txt', 'r') as f:
        read_data = f.read()
        print("-------------Sample text---------------")
        print (read_data[0:500])
        print("---------------------------------------")
    f.closed

class TextLoader():
    '''
    Reads data from input file
    I DID NOT MAKE THIS CLASS
    '''
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size /\
                               (self.batch_size * self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size /\
                               (self.batch_size * self.seq_length))

        # When the data (tensor) is too small, let's give them a better
        # error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and"\
                          "batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * \
                                  self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),\
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),\
                                  self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

# Parameters for converting characters to numbers
seq_length = 50 # RNN sequence length
batch_size = 128  # minibatch size, i.e. size of data in each epoch
num_epochs = 20 # should change to 50 if want to see relatively good results
learning_rate = 0.002
decay_rate = 0.97
rnn_size = 128 # size of RNN hidden state (output dimension)
num_layers = 2 #number of layers in the RNN

# Running commands
download_data(data, file_name)
peak_at_data(file_name)

# Read data in batches. Represent each sequence as vector
data_loader = TextLoader('', batch_size, seq_length)
vocab_size = data_loader.vocab_size
print()
print('Vocabulary Size: ' + str(data_loader.vocab_size))
print('Characters:' + str(data_loader.chars))
print('Vocab Number of "F": ' + str(data_loader.vocab['F']))
print('Character Sequences (first batch): ' + str(data_loader.x_batches[0]))

# Define input/out
x, y = data_loader.next_batch()

# Define LSTM cell and create 2 layer (stacked) LSTM
cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

# Define input/target data
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets = tf.placeholder(tf.int32, [batch_size, seq_length])
initial_state = stacked_cell.zero_state(batch_size, tf.float32) 

# Check values of input_data
session = tf.Session(config = config) # have to remove config to run locally
feed_dict = {input_data: x, targets: y}
session.run(input_data, feed_dict)

# Embedding - [60, 50, 128] - 60 batches, 50 characters, 128-dim vector
# This was originally "reuse = False"
with tf.variable_scope('rnnlm', reuse = True):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size]) #128x65
    softmax_b = tf.get_variable("softmax_b", [vocab_size]) # 1x65)
    #with tf.device("/cpu:0"):
        
    # embedding variable is initialized randomely
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size])  #65x128

    # embedding_lookup goes to each row of input_data, and for each character
    # in the row, finds the correspond vector in embedding
    # it creates a 60*50*[1*128] matrix
    # so, the first element of em is a matrix of 50x128, which each row of it
    # is vector representing that character
    em = tf.nn.embedding_lookup(embedding, input_data) # em is 60x50x[1*128]
    # split: Splits a tensor into sub tensors.
    # syntax:  tf.split(split_dim, num_split, value, name='split')
    # it will split the 60x50x[1x128] matrix into 50 matrix of 60x[1*128]
    inputs = tf.split(em, seq_length, 1)
    # It will convert the list to 50 matrix of [60x128]
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

# Initialize embedding variable
session.run(tf.global_variables_initializer())
session.run(embedding)

em = tf.nn.embedding_lookup(embedding, input_data)
emp = session.run(em,feed_dict = {input_data: x})

inputs = tf.split(em, seq_length, 1)
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
inputs[0:5]

# Feed batch of 50 to RNN
session.run(inputs[0],feed_dict = {input_data: x})
outputs, new_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                     inputs,
                     initial_state,
                     stacked_cell,
                     loop_function = None,
                     scope = 'rnnlm')

# Status check on output of network after first batch
first_output = outputs[0]
session.run(tf.global_variables_initializer())
session.run(first_output,feed_dict = {input_data: x})

# Reshape output vector to [60, 50, 128]
output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])
logits = tf.matmul(output, softmax_w) + softmax_b
probs = tf.nn.softmax(logits)

# Probability of next character in all batches
session.run(tf.global_variables_initializer())
session.run(probs,feed_dict={input_data:x})

# Training variables
grad_clip = 5.
tvars = tf.trainable_variables()

# LSTM Model

class LSTMModel():
    def __init__(self, sample = False, device = '/cpu:0'):
        rnn_size = 128 # size of RNN hidden state vector
        batch_size = 128 # minibatch size, i.e. size of dataset in each epoch
        seq_length = 50 # RNN sequence length
        num_layers = 2 # number of layers in the RNN
        vocab_size = 65
        grad_clip = 5.
        if sample:
            batch_size = 1
            seq_length = 1
        with tf.device(device):
            # The core of the model consists of an LSTM cell that processes
            # one char at a time and computes probabilities of the possible
            # continuations of the char. 
            basic_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
            # model.cell.state_size is (128, 128)
            self.stacked_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * \
                                                            num_layers)

            self.input_data = tf.placeholder(tf.int32,
                                             [batch_size, seq_length],
                                             name = "input_data")
            self.targets = tf.placeholder(tf.int32,
                                          [batch_size, seq_length],
                                          name = "targets")
            # Initial state of the LSTM memory.
            # The memory state of the network is initialized with a vector of
            # zeros and gets updated after reading each char. 
            self.initial_state = stacked_cell.zero_state(batch_size,
                                                         tf.float32) #why batch_size

            with tf.variable_scope('rnnlm_class1'):
                softmax_w = tf.get_variable("softmax_w",
                                            [rnn_size, vocab_size]) #128x65
                softmax_b = tf.get_variable("softmax_b",
                                            [vocab_size]) # 1x65
                embedding = tf.get_variable("embedding",
                                            [vocab_size, rnn_size])  #65x128
                inputs = tf.split(tf.nn.embedding_lookup(\
                                  embedding, self.input_data), seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs] 

            # The value of state is updated after processing each batch
            # of chars.
            outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                                  inputs,
                                  self.initial_state,
                                  self.stacked_cell,
                                  loop_function = None,
                                  scope = 'rnnlm_class1')
            output = tf.reshape(tf.concat(outputs,1), [-1, rnn_size])
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [self.logits],
                    [tf.reshape(self.targets, [-1])],
                    [tf.ones([batch_size * seq_length])],
                    vocab_size)
            self.cost = tf.reduce_sum(loss) / batch_size / seq_length
            self.final_state = last_state
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(
                                              self.cost, tvars), grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num = 200, prime = 'The ',
               sampling_type = 1):
        state = sess.run(self.stacked_cell.zero_state(1, tf.float32))
        #print state
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret