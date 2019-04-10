#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 06:50:35 2018

@author: Jake
"""

def yieldAllCombos(items):
    
    """
        Generates all combinations of N items into two bags, whereby each 
        item is in one or zero bags.

        Yields a tuple, (bag1, bag2), where each bag is represented as a list 
        of which item(s) are in each bag.
    """
    
#    print('\nStart')
    N = len(items)
    for i in range(3**N):
#        print('i:',i)
        bag1 = []
        bag2 = []
        for j in range(N):
            if (i // 3**j) % 3 == 1:
                bag1.append(items[j])
            elif (i // 3**j) % 3 == 2:
                bag2.append(items[j])
#        print('Bag 1:',bag1,'Bag 2:',bag2)
        yield (bag1, bag2)
                
items = (1,2)

yieldAllCombos(items)
        
    
def yieldAllCombos_ans(items):
    """
    Generates all combinations of N items into two bags, whereby each item is in one or zero bags.

    Yields a tuple, (bag1, bag2), where each bag is represented as a list of which item(s) are in each bag.
    """
    N = len(items)
    # Enumerate the 3**N possible combinations   
    for i in range(3**N):
        bag1 = []
        bag2 = []
        for j in range(N):
            if (i // (3 ** j)) % 3 == 1:
                bag1.append(items[j])
            elif (i // (3 ** j)) % 3 == 2:
                bag2.append(items[j])
        yield (bag1, bag2)
        
yieldAllCombos_ans(items)