#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 09:25:59 2017

@author: Jake
"""

import string

message_test = 'testing test'

def apply_shift(shift):
    '''
    Applies the Caesar Cipher to self.message_text with the input shift.
    Creates a new string that is self.message_text shifted down the
    alphabet by some number of characters determined by the input shift        
    
    shift (integer): the shift with which to encrypt the message.
    0 <= shift < 26

    Returns: the message text (string) in which every character is shifted
         down the alphabet by the input shift
    '''
    
    shifted_dict = self.build_shift_dict(shift)
    shifted_message = []
    
    for i in range(len(self.get_message_text())):
        if self.get_message_text()[i] not in string.ascii_letters:
            shifted_message.append(self.get_message_text()[i])
        else:
            shifted_message.append(shifted_dict[self.get_message_text()[i]])
    
    return ''.join(shifted_message)