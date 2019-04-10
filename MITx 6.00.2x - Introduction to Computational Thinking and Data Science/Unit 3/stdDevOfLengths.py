#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:33:37 2018

@author: Jake
"""

def stdDevOfLengths(L):
    """
    L: a list of strings

    returns: float, the standard deviation of the lengths of the strings,
      or NaN if L is empty.
    """
    
    # Return NaN if L is empty
    if L == []:
        return float('NaN')
    
    # Initialize variables
    numerator = 0
    average = 0
    
    # Find average of L
    for phrase in L:
        average += (len(phrase) / len(L))
    
    # Create numerator
    for phrase in L:
        numerator += (len(phrase) - average) ** 2
    
    # Return standard deviation
    return (numerator / len(L)) ** 0.5