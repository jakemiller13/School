#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 06:41:34 2017

@author: Jake
"""

def calculateHandlen(hand):
    
    """ 
    Returns the length (number of letters) in the current hand.
    
    hand: dictionary (string int)
    returns: integer
    """
    
    hand_length = 0
    
    for num in hand.values():
        hand_length += num
    
    return hand_length

hand = {'a':0, 'b':1, 'o':1, 't':1, 'l':1, 'e':1}
print(calculateHandlen(hand))