#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 19:50:48 2018

@author: Jake
"""

import random

def noReplacementSimulation(numTrials):
    '''
    Runs numTrials trials of a Monte Carlo simulation
    of drawing 3 balls out of a bucket containing
    3 red and 3 green balls. Balls are not replaced once
    drawn. Returns the a decimal - the fraction of times 3 
    balls of the same color were drawn.
    '''
    
    same_color = 0
    different_color = 0
    
    for num in range(numTrials):
        bucket = ['r', 'r', 'r', 'g', 'g', 'g']
        choice = []
        
        for i in range(3):
            pick = random.choice(bucket)
            choice.append(pick)
            bucket.remove(pick)
            
#            print('choice:',choice)
#            print('bucket:',bucket)
        
        if choice[0] == choice[1] == choice[2]:
            same_color += 1
        else:
            different_color += 1
    
    return same_color/numTrials