#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 06:11:35 2017

@author: Jake
"""

def isValidWord(word, hand, wordList):
    
    """
    Returns True if word is in the wordList and is entirely
    composed of letters in the hand. Otherwise, returns False.

    Does not mutate hand or wordList.
   
    word: string
    hand: dictionary (string -> int)
    wordList: list of lowercase strings
    """
    
    hand_copy = hand.copy()
    
    if word not in wordList:
        return False
    
    for letter in word:
        if letter not in hand_copy.keys() or hand_copy[letter] == 0:
            return False
        else:
            hand_copy[letter] -= 1
    
    return True


word = 'bottle'
hand = {'a':1, 'b':1, 'o':1, 't':1, 'l':1, 'e':1}

### DELETE WORD LIST BEFORE SUBMITTING ###
    
print("Loading word list from file...")
# inFile: file
inFile = open(WORDLIST_FILENAME, 'r')
# wordList: list of strings
wordList = []
for line in inFile:
    wordList.append(line.strip().lower())
print("  ", len(wordList), "words loaded.")

### DELETE ABOVE ###

print(isValidWord(word, hand, wordList))