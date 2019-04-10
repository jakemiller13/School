#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 06:39:21 2017

@author: Jake
"""

def isIn(char, aStr):
    '''
    char: a single character
    aStr: an alphabetized string
    
    returns: True if char is in aStr; False otherwise
    '''
    #char = str(char)
    if char == aStr:
        return True
    elif aStr == '' or len(aStr) == 1:
        return False
    elif char > aStr[int(len(aStr)/2)]:
        #return aStr w/ aStr being second half
        return isIn(char)