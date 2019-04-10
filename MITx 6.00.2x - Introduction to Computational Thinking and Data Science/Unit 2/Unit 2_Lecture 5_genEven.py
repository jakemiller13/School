#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:46:20 2018

@author: Jake
"""

import random

def genEven(n = 99):
    
    '''
    Returns a random even number x, where 0 <= x < 100
    '''
    num = random.randint(0,n)
    
    if num % 2 == 0:
        return num
    else:
        return genEven(n)

print(genEven())