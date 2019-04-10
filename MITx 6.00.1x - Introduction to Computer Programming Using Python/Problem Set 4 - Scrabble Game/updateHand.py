#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 06:34:20 2017

@author: Jake
"""

def updateHand(hand, word):
    
    """
    Assumes that 'hand' has all the letters in word.
    In other words, this assumes that however many times
    a letter appears in 'word', 'hand' has at least as
    many of that letter in it. 

    Updates the hand: uses up the letters in the given word
    and returns the new hand, without those letters in it.

    Has no side effects: does not modify hand.

    word: string
    hand: dictionary (string -> int)    
    returns: dictionary (string -> int)
    """
    
    new_hand = hand.copy()
    
    for letter in word:
        new_hand[letter] = new_hand.get(letter,0) - 1
    
    return new_hand
    
hand = {'a':1, 'q':1, 'l':2, 'm':1, 'u':1, 'i':1}
### displayHand(hand) # Implemented for you
### a q l l m u i
hand = updateHand(hand, 'quail') # You implement this function!
print(hand)
### {'a':0, 'q':0, 'l':1, 'm':1, 'u':0, 'i':0}
### displayHand(hand)
### l m  