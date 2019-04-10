#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:35:09 2018

@author: Jake
"""

import random

def rollDice(numDice, sides):
    
    print('You are rolling ' + str(numDice) + ' ' + str(sides) + '-sided dice\n')
    
    #Generates a random dice roll based on number of dice and sides
    randomDiceRoll = []
    
    for i in range(numDice):
        randomDiceRoll.append(random.randint(1,sides))
    
    #Generates a RANDOM list of possible rolls based on number of dice and sides
    setOfRolls = []
    
    for i in range(numDice):
        setOfRolls.append(random.randint(1,sides))
    
    print('Random set of rolls: ',setOfRolls) 
    
    #Generates a set of consecutive numbers based on number of dice and sides
    possibleSets = []
    
    for i in range(numDice):
        possibleSets.append(list(range(1,sides + 1)))
    
    print('\nPossible sets: ' + str(possibleSets))
    print()
    
    total_rolls = 0
    sum2Even = 0
    sum2Odd = 0
    firstEqualsSecond = 0
    firstLarger = 0
    rollIs4 = 0
    rollIsNot4 = 0
    TwoThree = 0
    
    for i in range(1,sides+1):
        for j in range(1,sides+1):
            
            total_rolls += 1
            
            #Sums are even
            if (i + j) % 2 == 0:
                sum2Even += 1
            
            #Sums are odd
            if (i + j) % 2 == 1:
                sum2Odd += 1
            
            #First roll equals second roll
            if i == j:
                firstEqualsSecond += 1
            
            #First roll larger than second
            if i > j:
                firstLarger += 1
            
            #Any roll is a 4
            if i == 4 or j == 4:
                rollIs4 += 1
            
            #No roll is a 4
            if i != 4 and j != 4:
                rollIsNot4 += 1
            
            #Roll 2 followed by 3
            if i == 2 and j == 3:
                TwoThree += 1
                
    

    print('Sample space size = ',str(sides ** numDice))
    print('Probability sum of rolls is EVEN = ',str(sum2Even/total_rolls))
    print('Probability sum of rolls is ODD = ',str(sum2Odd/total_rolls))
    print('Probability first roll EQUALS second = ',str(firstEqualsSecond/total_rolls))
    print('Probability first roll LARGER than second = ',str(firstLarger/total_rolls))
    print('Probability EITHER roll is 4 = ',str(rollIs4/total_rolls))
    print('Probability NEITHER roll is 4 = ',str(rollIsNot4/total_rolls))
    print('Probability of 2 followed by 3 = ',str(TwoThree/total_rolls))

numDice = int(input('How many dice are you rolling? '))
sides = int(input('How many sides does each dice have? '))
print()
rollDice(numDice,sides)