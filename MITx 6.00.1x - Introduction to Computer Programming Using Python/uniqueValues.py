#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:23:07 2017

@author: Jake
"""

def uniqueValues(aDict):
    
    '''
    aDict: a dictionary
    returns a list
    '''
    
    aDict_values = list(aDict.values())
    
    for value in aDict_values:
        if aDict_values.count(value) > 1:
            for k,v in list(aDict.items()):
                if v == value:
                    del aDict[k]

    return list(aDict.keys())
            

uniqueValues({1: 1, 2: 2, 3: 3})
uniqueValues({1: 1, 2: 1, 3: 1})
uniqueValues({0: 3, 1: 2, 2: 3, 3: 1, 4: 0, 6: 0, 7: 4, 8: 2, 9: 7, 10: 0})
#3, 7, 9