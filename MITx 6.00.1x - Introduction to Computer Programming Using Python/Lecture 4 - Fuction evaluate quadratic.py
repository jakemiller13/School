#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 06:42:55 2017

@author: Jake
"""

def evalQuadratic(a, b, c, x):
    
    '''
    a, b, c: numerical values for the coefficients of a quadratic equation
    x: numerical value at which to evaluate the quadratic.
    '''

    y = a * x**2 + b * x + c
    return y