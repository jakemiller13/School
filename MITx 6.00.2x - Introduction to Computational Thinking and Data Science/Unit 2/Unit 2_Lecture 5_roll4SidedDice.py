#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:35:09 2018

@author: Jake
"""

import random

def rollDice(numDice, sides):
    
    total_rolls = 0
    sum2Even = 0
    
    for i in range(1,5):
        for j in range(1,5):
            total_rolls += 1
            
            #Check how many sums are even
            if (i + j) % 2 == 0:
                sum2Even += 1
    
    print('Probability sum of rolls is even = ',str(sum2Even/total_rolls))

rollDice(2,4)