#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:11:41 2017

@author: Jake
"""

def getAvailableLetters(lettersGuessed):
    
    '''
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters that represents what letters have not
      yet been guessed.
    '''
    
    import string
    remainingLetters = list(string.ascii_lowercase)
    
    for letter in lettersGuessed:

        if letter in remainingLetters:
            remainingLetters.remove(letter)
#            print(''.join(remainingLetters))
        else:
            print("Oops, you've already guessed \"" + letter + "\"")
    
    return "Available letters: " + ''.join(remainingLetters)
        
lettersGuessed = ['e', 'i', 'k', 'p', 'r', 's']
print(getAvailableLetters(lettersGuessed))