#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 06:04:35 2018

@author: Jake
"""

# Makes Python package NumPy available using import method
import numpy as np

# Creates matrix t (right side of the augmented matrix).
t = np.array([4, 11])

# Creates matrix vw (left side of the augmented matrix).
vw = np.array([[1, 2], [3, 5]])

# Prints vw and t
print("\nMatrix vw:", vw, "\nVector t:", t, sep="\n")

def check_vector_span(set_of_vectors, vector_to_check):
    
    # Creates an empty vector of correct size
    vector_of_scalars = np.asarray([None]*set_of_vectors.shape[0])
    
    # Solves for the scalars that make the equation true if vector is within the span
    try:
        
        # DONE: Use np.linalg.solve() function here to solve for vector_of_scalars
        vector_of_scalars = np.linalg.solve(set_of_vectors, vector_to_check)
        
        if not (vector_of_scalars is None):
            print("\nVector is within span.\nScalars in s:", vector_of_scalars)
            
    # Handles the cases when the vector is NOT within the span   
    except Exception as exception_type:
        
        if str(exception_type) == "Singular matrix":
            print("\nNo single solution\nVector is NOT within span")
            
        else:
            print("\nUnexpected Exception Error:", exception_type)
            
    return vector_of_scalars

# Call to check_vector_span to check vectors in Equation 1
print("\nEquation 1:\n Matrix vw:", vw, "\nVector t:", t, sep="\n")
s = check_vector_span(vw,t)

# Call to check a new set of vectors vw2 and t2
vw2 = np.array([[1, 2], [2, 4]]) 
t2 = np.array([6, 12])
print("\nNew Vectors:\n Matrix vw2:", vw2, "\nVector t2:", t2, sep="\n")    
# Call to check_vector_span
s2 = check_vector_span(vw2,t2)

# Call to check a new set of vectors vw3 and t3
vw3 = np.array([[1, 2], [1, 2]]) 
t3 = np.array([6, 10])
print("\nNew Vectors:\n Matrix vw3:", vw3, "\nVector t3:", t3, sep="\n")    
# Call to check_vector_span
s3 = check_vector_span(vw3,t3)
