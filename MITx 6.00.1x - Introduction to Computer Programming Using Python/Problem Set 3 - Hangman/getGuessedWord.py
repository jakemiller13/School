#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 06:23:07 2017

@author: Jake
"""

def getGuessedWord(secretWord, lettersGuessed):
    
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters and underscores that represents
      what letters in secretWord have been guessed so far.
    '''

    secretWordDict = {}
    guessed = {} #will eventually need to change this to lettersGuessed
    
    blankSecretWord = list('_' * len(secretWord))
    
    for i in secretWord:
        secretWordDict[i] = i
#        print(secretWordDict)
    
    for letter in lettersGuessed:
#        print('letter =',letter)
#        print('guessed =',guessed)

        if letter in secretWord and letter not in guessed:
            secretWordDict.pop(letter)
#            print(secretWordDict)
            
            for j in range(len(secretWord)):
                if secretWord[j] == letter:
                    blankSecretWord[j] = letter
#                    print(' '.join(blankSecretWord))
                
        elif letter in guessed:
#            print("Oops, you've already guessed",letter)
            pass
            
        guessed[letter] = letter
    
#    if secretWordDict == {}:
#        return True
#    else:
#        return False

    return ' '.join(blankSecretWord)

secretWord = 'apple' 
lettersGuessed = ['e', 'i', 'k', 'p', 'r', 's']
print(getGuessedWord(secretWord, lettersGuessed))