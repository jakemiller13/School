#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 07:47:39 2017

@author: Jake
"""

import string


##DON"T FORGET SELF HERE IF YOU REMOVE IT
#def build_shift_dict(self, shift):

def build_shift_dict(shift):
    '''
    Creates a dictionary that can be used to apply a cipher to a letter.
    The dictionary maps every uppercase and lowercase letter to a
    character shifted down the alphabet by the input shift. The dictionary
    should have 52 keys of all the uppercase letters and all the lowercase
    letters only.        
    
    shift (integer): the amount by which to shift every letter of the 
    alphabet. 0 <= shift < 26

    Returns: a dictionary mapping a letter (string) to 
             another letter (string). 
    '''
    
    long_lower = 2 * string.ascii_lowercase
    long_upper = 2 * string.ascii_uppercase
    
    shift_dict_lower = {long_lower[i]: long_lower[i+shift] for i in range(len(string.ascii_lowercase))}
    
    shift_dict_upper = {long_upper[i]: long_upper[i+shift] for i in range(len(string.ascii_lowercase))}
    
    return dict(shift_dict_lower, **shift_dict_upper)