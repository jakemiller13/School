#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 06:38:32 2018

@author: Jake
"""

import itertools

items = [1, 2, 3, 4]
powerset = [x for length in range(len(items)+1) for x in itertools.combinations(items, length)]




from itertools import chain, combinations

def Powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

test = [1,2,3,4,5]
Powerset(test)