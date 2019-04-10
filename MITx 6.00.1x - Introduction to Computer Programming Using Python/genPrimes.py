#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 06:38:19 2017

@author: Jake
"""

def genPrimes():
    
    count = 1
    primes = []
    
    while True:
        count += 1
        if all(count%i != 0 for i in primes):
            primes.append(count)
            yield count