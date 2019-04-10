#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:09:37 2017

@author: Jake
"""

high = 100
low = 0
guess = 50

# this is not used in this code, but could be used in the future
#user_number = int(input('Please think of a number between ' + str(low) + ' and ' + str(high) + ': '))

print('Please think of a number between 0 and 100!')

while True:
    
    print('\nIs your secret number ' + str(guess) + '?')
    
    high_low = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low. Enter 'c' to indicate I guessed correctly. ")
    
    if high_low == 'h':
        high = guess
        guess = int((guess + low) / 2)
    elif high_low == 'l':
        low = guess
        guess = int((guess + high)/2)
    elif high_low == 'c':
        print('\nGame over. Your secret number was: ' + str(guess))
        break
    else:
        print('\nSorry, I did not understand your input.')