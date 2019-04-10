#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 06:27:13 2018

@author: Jake
"""

import pylab as plt

mySamples = []
myLinear = []
myQuadratic = []
myCubic = []
myExponential = []

for i in range(0,30):
    mySamples.append(i)
    myLinear.append(i)
    myQuadratic.append(i**2)
    myCubic.append(i**3)
    myExponential.append(1.5**i)

#Linear Plot
plt.figure('lin')
plt.clf()
plt.ylim(0,1000)
plt.title('Linear')
plt.plot(mySamples,myLinear)

#Quadratic Plot
plt.figure('quad')
plt.clf()
plt.ylim(0,1000)
plt.title('Quadratic')
plt.plot(mySamples,myQuadratic)

#Cubic Plot
plt.figure('cube')
plt.clf()
plt.plot(mySamples,myCubic)
plt.title('Cubic')

#Exponential Plot
plt.figure('expo')
plt.clf()
plt.plot(mySamples,myExponential)
plt.title('Exponential')

#Linear vs. Quadratic Plot
plt.figure('lin v quad')
plt.clf()
plt.title('Linear v Quadratic')
plt.subplot(211)
plt.ylim(0,900)
plt.plot(mySamples,myLinear,'b-',label = 'Linear',linewidth = 2.0)
plt.subplot(212)
plt.ylim(0,900)
plt.plot(mySamples,myQuadratic,'ro',label = 'Quadratic',linewidth = 3.0)
plt.legend(loc = 'upper left')

#Cubic vs. Exponential Plot
plt.figure('cube v expo')
plt.clf()
plt.title('Cubic v Exponential')
plt.subplot(121)
plt.ylim(0,14000)
plt.plot(mySamples,myCubic,'g^',label = 'Cubic',linewidth = 4.0)
plt.subplot(122)
plt.ylim(0,14000)
plt.plot(mySamples,myExponential,'r--',label = 'Exponential', linewidth = 5.0)
plt.legend(loc = 'upper left')

#Cubic vs. Exponential Using Log-scale
plt.figure('cube v expo')
plt.clf()
plt.title('Cubic v Exponential w/ Log-scale')
plt.plot(mySamples,myCubic,'g^',label = 'Cubic',linewidth = 4.0)
plt.plot(mySamples,myExponential,'r--',label = 'Exponential', linewidth = 5.0)
plt.yscale('log')
plt.legend(loc = 'upper left')