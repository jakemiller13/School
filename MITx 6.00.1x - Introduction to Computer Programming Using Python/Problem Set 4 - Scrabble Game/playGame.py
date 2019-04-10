#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:37:53 2017

@author: Jake
"""

HAND_SIZE = 7
n = HAND_SIZE

def playGame(wordList):
    
    """
    Allow the user to play an arbitrary number of hands.

    1) Asks the user to input 'n' or 'r' or 'e'.
      * If the user inputs 'n', let the user play a new (random) hand.
      * If the user inputs 'r', let the user play the last hand again.
      * If the user inputs 'e', exit the game.
      * If the user inputs anything else, tell them their input was invalid.
 
    2) When done playing the hand, repeat from step 1    
    """
    
    while True:
        
        n = HAND_SIZE ##This is a variable in main function
        
        game_mode = input('Enter n to deal a new hand, r to replay the last hand, or e to end game: ')
    
        if game_mode == 'n':
            hand = dealHand(n)
            playHand(hand, wordList, n)
        elif game_mode == 'r':
            try:
                playHand(hand, wordList, n)
            except UnboundLocalError:
                print('You have not played a hand yet. Please play a new hand first!')
        elif game_mode == 'e':
            break
        else:
            print('Invalid command.')


wordList = loadWords()
playGame(wordList)
#playHand({'p':1, 'z':1, 'u':1, 't':3, 'o':1}, wordList, 7)