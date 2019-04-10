#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 06:31:44 2017

@author: Jake
"""

def how_many(aDict):
    
    '''
    aDict: A dictionary, where all the values are lists.

    returns: int, how many values are in the dictionary.
    '''
    
    count = 0
    
    for i in aDict:
        
        count += len(aDict[i])
        
    return count

print(how_many(animals))