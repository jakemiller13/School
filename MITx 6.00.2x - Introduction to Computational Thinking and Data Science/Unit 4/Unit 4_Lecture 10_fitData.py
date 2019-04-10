#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:03:27 2018

@author: Jake
"""

import pylab, numpy, random

def getData(fileName):
    dataFile = open(fileName, 'r')
    distances = []
    masses = []
    dataFile.readline() #discard header
    for line in dataFile:
        d, m = line.split()
        distances.append(float(d))
        masses.append(float(m))
    dataFile.close()
    return (masses, distances)

def labelPlot():
    pylab.title('Measured Displacement of Spring')
    pylab.xlabel('|Force| (Newtons)')
    pylab.ylabel('Distance (meters)')

def fitData(fileName):
    xVals, yVals = getData(fileName)
    xVals = pylab.array(xVals)
    yVals = pylab.array(yVals)
    xVals = xVals * 9.81 #gets force
    pylab.figure(1)
    pylab.plot(xVals, yVals, 'bo', label = 'Measured Points')
    labelPlot()
    a,b = pylab.polyfit(xVals, yVals, 1)
    estYVals = a * xVals + b
    print('a =', a, 'b =', b)
    pylab.plot(xVals, estYVals, 'r', label = 'Linear fit, k =' + str(round(1/a, 5)))
    pylab.legend(loc = 'best')

fitData('springData.txt')

def fitData1(fileName):
    xVals, yVals = getData(fileName)
    xVals = pylab.array(xVals)
    yVals = pylab.array(yVals)
    xVals = xVals * 9.81 #gets force
    pylab.figure(2)
    pylab.plot(xVals, yVals, 'bo', label = 'Measured Points')
    labelPlot()
    model = pylab.polyfit(xVals, yVals, 1)
    estYVals = pylab.polyval(model, xVals)
    pylab.plot(xVals, estYVals, 'r', label = 'Linear fit, k =' + str(round(1/model[0], 5)))
    pylab.legend(loc = 'best')

fitData1('springData.txt')

def aveMeanSquareError(data, predicted):
    error = 0.0
    for i in range(len(data)):
        error += (data[i] - predicted[i])**2
    return error/len(data)

#Import data
xVals, yVals = getData('mysteryData.txt')
pylab.figure(3)
pylab.plot(xVals, yVals, 'o', label = 'Data Points')
pylab.title('Mystery Data')

#Linear model
model1 = pylab.polyfit(xVals, yVals, 1)
pylab.plot(xVals, pylab.polyval(model1, xVals), label = 'Linear Model')

#Quadratic model
model2 = pylab.polyfit(xVals, yVals, 2)
pylab.plot(xVals, pylab.polyval(model2, xVals), 'r--', label = 'Quadratic Model')
pylab.legend()

estYVals = pylab.polyval(model1, xVals)
print('Ave. mean square error for linear model =', aveMeanSquareError(yVals, estYVals))
estYVals = pylab.polyval(model2, xVals)
print('Ave. mean square error for quadratice model =', aveMeanSquareError(yVals, estYVals))