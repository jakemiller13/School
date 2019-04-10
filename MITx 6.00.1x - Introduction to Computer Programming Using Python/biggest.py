#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 06:40:21 2017

@author: Jake
"""

def biggest(aDict):
    
    '''
    aDict: A dictionary, where all the values are lists.

    returns: The key with the largest number of values associated with it
    '''
    
    biggest_key = ''
    num = 0
    
    for i in aDict:
        
        if len(aDict[i]) > num:
            num = len(aDict[i])
            biggest_key = i
        
    return biggest_key

print(biggest(animals))