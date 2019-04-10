#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 06:35:16 2017

@author: Jake
"""

def isWordGuessed(secretWord, lettersGuessed):
    
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: boolean, True if all the letters of secretWord are in lettersGuessed;
      False otherwise
    '''
    
    secretWordDict = {}
    guessed = {}
    
    for i in secretWord:
        secretWordDict[i] = i
#        print(secretWordDict)
    
    for letter in lettersGuessed:
#        print('letter =',letter)
#        print('guessed =',guessed)

        if letter in secretWord and letter not in guessed:
            secretWordDict.pop(letter)
#            print(secretWordDict)
        elif letter in guessed:
#            print("Oops, you've already guessed",letter)
            pass
            
        guessed[letter] = letter
    
    if secretWordDict == {}:
        return True
    else:
        return False
    
secretWord = 'apple' 
lettersGuessed = ['e', 'i', 'k', 'p', 'r', 's']
print(isWordGuessed('pineapple', ['z', 'x', 'q', 'p', 'i', 'n', 'e', 'a', 'p', 'p', 'l', 'e']))