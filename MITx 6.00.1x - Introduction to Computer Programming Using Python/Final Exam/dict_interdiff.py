#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 09:16:53 2017

@author: Jake
"""

def dict_interdiff(d1, d2):
    
    #create tuple and new empty dictionaries
    (d3, d4) = ({},{})
    
    #iterate over keys in d1
    for k in d1:
        
        #compare to keys in d2
        if k in d2:
            
            #perform function on d1[k] and d2[k], add to inter dict
            d3[k] = f(d1[k],d2[k])
            
            #delete from d2
            del d2[k]
            
        #if k not in both, add d1[k] to diff dict
        else:
            d4[k] = d1[k]
            
    #if any left in d2, add to diff dict
    d4.update(d2)
    
    return (d3, d4)