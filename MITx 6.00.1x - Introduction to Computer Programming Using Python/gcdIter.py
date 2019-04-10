#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 07:13:02 2017

@author: Jake
"""

def gcdIter(a, b):
    '''
    a, b: positive integers
    
    returns: a positive integer, the greatest common divisor of a & b.
    '''

    if a >= b:
        divisor = b
#        print("I'm in a > b and divisor =",divisor)
        while a%divisor != 0 or b%divisor != 0:
            divisor -= 1
#            print("I'm in a > b and divisor should be decrementing: ",divisor)
#            print("a%divisor = ",a%divisor,". b%divisor = ",b%divisor,".\n")
        return divisor
    
    elif b > a:
        divisor = a
#        print("I'm in b > a and divisor =",divisor)
        while b%divisor != 0 or a%divisor != 0:
            divisor -= 1
#            print("I'm in b > a and divisor should be decrementing: ",divisor)
#            print("b%divisor = ",b%divisor,". a%divisor = ",a%divisor,".\n")
        return divisor
    
#    else:
#        return 'DNE'
        
#a = float(input('a = '))
#b = float(input('b = '))

print('\nGreatest common denominator =',gcdIter(17,12))