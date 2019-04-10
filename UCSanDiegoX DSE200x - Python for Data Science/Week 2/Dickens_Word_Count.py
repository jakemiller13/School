#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 06:24:05 2019

@author: Jake
"""

import string

text = open('./word_cloud/98-0.txt')

punctuation = set(string.punctuation)
word_cloud = {}

for word in text.read().lower().split():
    word_set = set(word)
    for char in word_set.intersection(string.punctuation):
        word.replace(char, '')
    
    if word.strip() in word_cloud:
        word_cloud[word.strip()] += 1
    else:
        word_cloud[word.strip()] = 1

print(word_cloud)