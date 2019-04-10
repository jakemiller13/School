#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 06:19:10 2017

@author: Jake
"""

def recurPower(base, exp):
    
    '''
    base: int or float.
    exp: int >= 0
 
    returns: int or float, base^exp
    '''
    
    if exp == 0:
        return 1
    else:
        return base * recurPower(base,exp-1)