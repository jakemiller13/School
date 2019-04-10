#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 08:35:02 2017

@author: Jake
"""

def gcdRecur(a, b):
    
    '''
    a, b: positive integers
    
    returns: a positive integer, the greatest common divisor of a & b.
    '''
    
    if b == 0:
        return a
    else:
        return gcdRecur(b, a % b)

#print('GCD of ' + str(a) + ' and ' + str(b) + ' is ' + str(gcdRecur(a,b)))