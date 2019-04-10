#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:28:49 2018

@author: Jake
"""

import random

def rollDie():
    '''
    Returns a random int between 1 and 6
    '''
    
    return random.choice([0,1,2,3,4,5,6])

def testRoll(n):
    '''
    Tests rollDie()
    '''
    
    result = ''
    for i in range(n):
        result = result + str(rollDie())
    print(result)