#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:24:45 2017

@author: Jake
"""

def closest_power(base, num):
    '''
    base: base of the exponential, integer > 1
    num: number you want to be closest to, integer > 0
    Find the integer exponent such that base**exponent is closest to num.
    Note that the base**exponent may be either greater or smaller than num.
    In case of a tie, return the smaller value.
    Returns the exponent.
    '''
    
    y = 0
    
    while abs((base**y - num)) > abs((base**(y+1) - num)):
        y += 1
    
    return y

print(closest_power(3,12)) #returns 2
print(closest_power(4,12)) #returns 2
print(closest_power(4,1)) #returns 0