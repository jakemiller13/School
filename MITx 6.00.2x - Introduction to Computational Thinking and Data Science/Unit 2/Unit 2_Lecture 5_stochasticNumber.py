#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:00:45 2018

@author: Jake
"""

import random

def stochasticNumber():
    '''
    Stochastically generates and returns a uniformly distributed even number between 9 and 21
    '''
    
    num = random.randint(9,21)
    
    if num % 2 == 0:
        return num
    else:
        return stochasticNumber()

print(stochasticNumber())