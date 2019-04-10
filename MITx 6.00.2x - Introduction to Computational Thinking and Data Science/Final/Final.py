#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 08:12:46 2018

@author: Jake
"""

#import random, pylab
#xVals = []
#yVals = []
#wVals = []
#for i in range(1000):
#    xVals.append(random.random())
#    yVals.append(random.random())
#    wVals.append(random.random())
#xVals = pylab.array(xVals)
#yVals = pylab.array(yVals)
#wVals = pylab.array(wVals)
#xVals = xVals + xVals
#zVals = xVals + yVals
#tVals = xVals + yVals + wVals
#
#t_array = []
#for i in range(1000):
#    t_array.append(i)
#t_array = pylab.array(t_array)
#
##pylab.plot(t_array,xVals,label = 'xVals')
##pylab.plot(t_array,tVals,label = 'tVals')
##pylab.legend(loc = 'best')
#
#pylab.plot(sorted(xVals), sorted(yVals))


#########


#def drawing_without_replacement_sim(numTrials):
#    '''
#    Runs numTrials trials of a Monte Carlo simulation
#    of drawing 3 balls out of a bucket containing
#    4 red and 4 green balls. Balls are not replaced once
#    drawn. Returns a float - the fraction of times 3 
#    balls of the same color were drawn in the first 3 draws.
#    '''
#    
#    import random
#    
#    same_color = 0
#    
#    for trial in range(numTrials):
#        choices = ['r','r','r','g','g','g']
#        instance = []
#        
#        for i in range(3):
#            pick = random.choice(choices)
#            instance.append(pick)
#            choices.remove(pick)
#        
#        if instance[0] == instance[1] and instance[1] == instance[2]:
#            same_color += 1
#    
#    return float(same_color/numTrials)
#
#print(drawing_without_replacement_sim(100000))


#########


#import random, pylab
#
## You are given this function
#def getMeanAndStd(X):
#    mean = sum(X)/float(len(X))
#    tot = 0.0
#    for x in X:
#        tot += (x - mean)**2
#    std = (tot/len(X))**0.5
#    return mean, std
#
## You are given this class
#class Die(object):
#    def __init__(self, valList):
#        """ valList is not empty """
#        self.possibleVals = valList[:]
#    def roll(self):
#        return random.choice(self.possibleVals)
#
## Implement this -- Coding Part 1 of 2
#def makeHistogram(values, numBins, xLabel, yLabel, title=None):
#    """
#      - values, a sequence of numbers
#      - numBins, a positive int
#      - xLabel, yLabel, title, are strings
#      - Produces a histogram of values with numBins bins and the indicated labels
#        for the x and y axis
#      - If title is provided by caller, puts that title on the figure and otherwise
#        does not title the figure
#    """
#    
#    pylab.hist(values, numBins)
#    pylab.xlabel(xLabel)
#    pylab.ylabel(yLabel)
#    if title != None:
#        pylab.title(title)
#    pylab.show()
#                    
## Implement this -- Coding Part 2 of 2
#def getAverage(die, numRolls, numTrials):
#    """
#      - die, a Die
#      - numRolls, numTrials, are positive ints
#      - Calculates the expected mean value of the longest run of a number
#        over numTrials runs of numRolls rolls.
#      - Calls makeHistogram to produce a histogram of the longest runs for all
#        the trials. There should be 10 bins in the histogram
#      - Choose appropriate labels for the x and y axes.
#      - Returns the mean calculated
#    """
#    
#    total_max_runs = []
#    
#    for trial in range(numTrials):
#        rolls = []
#        
#        for i in range(numRolls):
#            rolls.append(die.roll())
#        
#        temp_max_run = current_max_run = 1
#        
#        for i in range(1,numRolls):
#            if rolls[i] == rolls[i-1]:
#                temp_max_run += 1
#            elif temp_max_run > current_max_run:
#                current_max_run = temp_max_run
#                temp_max_run = 1
#            else:
#                temp_max_run = 1
#        
#        total_max_runs.append(current_max_run)
#    
#    makeHistogram(total_max_runs, 10, 'Expected Mean Value', '# Runs')#, title=None)
#    
#    return round(getMeanAndStd(total_max_runs)[0],3)
#          
#
##die = Die([1,2,3,4,5,6])
##numRolls = 100
##numTrials = 1000
##print(getAverage(die,numRolls,numTrials))
#    
## One test case
#print(getAverage(Die([1,2,3,4,5,6,6,6,7]), 500, 10000))
#print(getAverage(Die([1]), 10, 1000))
#print(getAverage(Die([1,2,3,4,5,6]),100,1000))
#print(getAverage(Die([1,1]), 10, 1000))


#########


def find_combination(choices, total):
    """
    choices: a non-empty list of ints
    total: a positive int
 
    Returns result, a numpy.array of length len(choices) 
    such that
        * each element of result is 0 or 1
        * sum(result*choices) == total
        * sum(result) is as small as possible
    In case of ties, returns any result that works.
    If there is no result that gives the exact total, 
    pick the one that gives sum(result*choices) closest 
    to total without going over.
    """
    import numpy as np
    
    bin_choices = []
    differences = []
    
    #A lot happens here. Creates binary representation for all combinations of len(choices). Pads zeroes to the front so all binary strings are the same length. Creates list of binary strings and turns each digit into int. Appends list of binary representation to bin_choices, and then turns the whole thing into an array.
    for i in range(2**len(choices)):
        bin_choices.append([int(x) for x in list(bin(i)[2:].zfill(len(choices)))])
    bin_choices = np.array(bin_choices)
    
    #Creates list of differences between total and sum of bin_choices*choices
    for combo in bin_choices:
        differences.append(total - sum(combo * choices))
    
    sorted_differences = list(np.sort(differences).copy())

    index = 0
    
    while sorted_differences[index] < 0:
#        print(differences[index])
        index += 1

    return bin_choices[differences.index(sorted_differences[index])]


#choices = [1,2,2,3]
#total = 4
## [0 1 1 0] or [1 0 0 1]
#print('1:',find_combination(choices,total))
#
#choices = [1,1,3,5,3]
#total = 5
## [0 0 0 1 0]
#print('2:',find_combination(choices,total))
#
#choices = [1,1,1,9]
#total = 4
## return [1 1 1 0]
#print('3:',find_combination(choices,total))
#
#choices = [10, 100, 1000, 3, 8, 12, 38]
#total = 1171
#print('4:',find_combination(choices,total))

#choices = [1, 81, 3, 102, 450, 10]
#total = 9
#print('5:',find_combination(choices,total))
##array([1, 0, 1, 0, 0, 0])

#choices = [4, 6, 3, 5, 2]
#total = 10
#print('6:',find_combination(choices,total))
#array([1, 1, 0, 0, 0])