#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:01:56 2019

@author: Jake
"""

import itertools

set_ = {1, 2, 3, 4, 5}

combo_list = set()

for i in range(len(set_)):
    combos = list(itertools.combinations(set_, i))
    combo_list.update(combos)

#############
# PROBLEM 6 #
#############

# 6.1
count = 0
for j in combo_list:
    for k in combo_list:
        if set(j).intersection(set(k)) == set():
#            print(set(j).intersection(set(k)))
            count += 1

print('Count for problem 6.1: ' + str(count))

# 6.2
count = 0
for j in combo_list:
    for k in combo_list:
        if set(j).intersection(set(k)) == set({1}):
            print(j, k)
            count += 1

print('Count for problem 6.2: ' + str(count))

# 6.3
count = 0
for j in combo_list:
    for k in combo_list:
        if len(set(j).intersection(set(k))) == 1:
            count += 1

print('Count for problem 6.3: ' + str(count))

#############
# PROBLEM 7 #
#############

# 7.1
count = 0
for i in combo_list:
    for j in combo_list:
        if set(i).union(set(j)) == {1, 2, 3, 4, 5}:
#            print(set(i), set(j), set(i).union(set(j)))
            count += 1

print('Count for problem 7.1: ' + str(count))

# 7.2
count = 0
for i in combo_list:
    for j in combo_list:
        if len(set(i).union(set(j))) == 4:
#            print(set(i).union(set(j)))
            count += 1

print('Count for problem 7.2: ' + str(count))