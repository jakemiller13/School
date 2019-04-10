#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 08:05:05 2017

@author: Jake
"""

def largest_odd_times(L):
    """
    Assumes L is a non-empty list of ints
    Returns the largest element of L that occurs an odd number 
    of times in L. If no such element exists, returns None
    """
    
    #sorts and mutates list. Do here because code references twice
    L.sort()
    
    try:
        
        #Counts number of highest element, checks if odd
        if L.count(L[-1]) % 2 == 1:
            return L[-1]
        
        #returns function, removes all occurances highest number
        else:
            #print('L[-1]:',L[-1])
            return largest_odd_times(L[:-L.count(L[-1])])
    
    except IndexError:
        return
    
    
    # largest_odd_times([2,2,4,4]) returns None
    # largest_odd_times([3,9,5,3,5,3]) returns 9