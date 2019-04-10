#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:21:52 2017

@author: Jake
"""

balance = 3329; annualInterestRate = 0.2

minMonthlyPayment = 0

#    z = 10
while True:
    
    newBalance = balance
    
#        print('z = ' + str(z))
#        z -= 1
    
    for i in range(0,12):

        monthlyInterest = annualInterestRate / 12.0
        monthlyUnpaidBalance = newBalance - minMonthlyPayment
        newBalance = monthlyUnpaidBalance + (monthlyInterest * monthlyUnpaidBalance)
        
#            print('Monthly unpaid balance: ' + str(monthlyUnpaidBalance))
        print('New balance: ' + str(newBalance))
    
    if newBalance > 0:

        minMonthlyPayment += 10
    
    else:
        
        print('Lowest Payment: ' + str(minMonthlyPayment))
        break