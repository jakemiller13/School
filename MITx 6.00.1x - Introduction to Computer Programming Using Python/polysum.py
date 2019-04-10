#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:57:40 2017

@author: Jake
"""

import math

def polysum(n,s):
    
    '''
    Takes 2 arguments and returns the area + perimeter^2, rounded to 4 digits
    n = number of sides of polygon
    s = length of side
    '''
    
    area = (0.25 * n * s**2) / (math.tan(math.pi/n))
    perimeter = n * s
    
    return round(area + perimeter**2,4)