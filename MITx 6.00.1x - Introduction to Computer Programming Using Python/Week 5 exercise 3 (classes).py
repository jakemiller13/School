#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:01:02 2017

@author: Jake
"""

class Weird(object):
    def __init__(self, x, y): 
        self.y = y
        self.x = x
    def getX(self):
        return x 
    def getY(self):
        return y

class Wild(object):
    def __init__(self, x, y): 
        self.y = y
        self.x = x
    def getX(self):
        return self.x 
    def getY(self):
        return self.y

X = 7
Y = 8

print('\nstart')

print('\n1.')
w1 = Weird(X, Y)
#print(w1.getX())

print('\n2.')

#print(w1.getY())

print('\n3.')

w2 = Wild(X, Y)
print(w2.getX())

print('\n4.')

print(w2.getY())

print('\n5.')

w3 = Wild(17, 18)
print(w3.getX())

print('\n6.')

print(w3.getY())

print('\n7.')

w4 = Wild(X, 18)
print(w4.getX())

print('\n8.')

print(w4.getY())

print('\n9.')

X = w4.getX() + w3.getX() + w2.getX()
print(X)

print('\n10.')

print(w4.getX())

print('\n11.')

Y = w4.getY() + w3.getY()
Y = Y + w2.getY()
print(Y)

print('\n12.')

print(w2.getY())