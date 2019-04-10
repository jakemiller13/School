#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 06:42:09 2017

@author: Jake
"""

def iterPower(base, exp):
    
    '''
    base: int or float.
    exp: int >= 0
 
    returns: int or float, base^exp
    '''
    
    total = 1

    if exp == 0:
        return total
    else:
        while exp > 0:
            total = total * base
            exp -= 1

    return total