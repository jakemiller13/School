#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:37:43 2017

@author: Jake
"""

def score(word, f):
    
    """
       word, a string of length > 1 of alphabetical 
             characters (upper and lowercase)
       f, a function that takes in two int arguments and returns an int

       Returns the score of word as defined by the method:

    1) Score for each letter is its location in the alphabet (a=1 ... z=26) 
       times its distance from start of word.  
       Ex. the scores for the letters in 'adD' are 1*0, 4*1, and 4*2.
    2) The score for a word is the result of applying f to the
       scores of the word's two highest scoring letters. 
       The first parameter to f is the highest letter score, 
       and the second parameter is the second highest letter score.
       Ex. If f returns the sum of its arguments, then the 
           score for 'adD' is 12 
    """
    
    #import string module
    import string
    
    #initiate variables
    letterList = list(string.ascii_lowercase)
    letterScore = {}
    x = 1
    score = []
    
    #assign value to each letter for letterScore
    for letter in letterList:
        letterScore[letter] = x
        x += 1
    
    #get score for word
    for letterPlace in range(len(word)):
        score.append(letterPlace * letterScore[word.lower()[letterPlace]])
        
    high_int = max(score)
    score.remove(high_int)
    low_int = max(score)
    
    return f(high_int, low_int)

#function f
def f(high_int, low_int):
    return high_int + low_int
        
print(score('finKlEBerry',f))