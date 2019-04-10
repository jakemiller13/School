#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 08:42:24 2018

@author: Jake
"""

def max_contig_sum(L):
    """ L, a list of integers, at least one positive
    Returns the maximum sum of a contiguous subsequence in L """
    
    totals = [] 
    
    for i in range(len(L)):
        temp_total = 0 #initiate temporary total to 0 each round
        
        while i < len(L):
            temp_total += L[i]
            totals.append(temp_total) #add temp_total each time we go up a step
            i += 1 #counter
    
    return max(totals)
        

print(max_contig_sum([3,4,-1,5,-4]))
print(max_contig_sum([3,4,-8,15,-1,2]))

#in the list [3, 4, -1, 5, -4], the maximum sum is 3+4-1+5 = 11
#in the list [3, 4, -8, 15, -1, 2], the maximum sum is 15-1+2 = 16