#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:32:27 2017

@author: Jake
"""

def flatten(aList):
    
    ''' 
    aList: a list 
    Returns a copy of aList, which is a flattened version of aList 
    '''
    
    if aList == []:
        return aList
    if isinstance(aList[0], list):
        return flatten(aList[0]) + flatten(aList[1:])
    return aList[:1] + flatten(aList[1:])
    
    print('here')
    print('newList =',aList)

testList = [[1,'a',['cat'],2],[[[3]],'dog'],4,5]
# [1,'a','cat',2,3,'dog',4,5]

print(flatten(testList))