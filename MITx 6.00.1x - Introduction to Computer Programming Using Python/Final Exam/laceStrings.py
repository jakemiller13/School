#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 07:25:45 2017

@author: Jake
"""

def laceStrings(s1, s2):
    """
    s1 and s2 are strings.

    Returns a new str with elements of s1 and s2 interlaced,
    beginning with s1. If strings are not of same length, 
    then the extra elements should appear at the end.
    """

    s3 = ''
    
    if len(s1) >= len(s2):
        long_string = s1
    else:
        long_string = s2
    
    for i in range(len(long_string)):
        try:
            s3 = s3 + s1[i] + s2[i]
        except IndexError:
            s3 = s3 + long_string[i:]
            break
    
    return s3