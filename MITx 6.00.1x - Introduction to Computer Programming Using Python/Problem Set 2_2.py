#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 06:47:29 2017

@author: Jake
"""

def lowestMonthlyPayment(balance, annualInterestRate):
    '''
    Given an initial balance and annual interest rate,
    calculates the minimum monthly payment needed to pay off balance in one year.
    
    balance: balance each month
    annualInterestRate: annual interest rate
    '''

#    newBalance = balance
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
#            print('New balance: ' + str(newBalance))
        
        if newBalance > 0:

            minMonthlyPayment += 10
        
        else:
            
            return minMonthlyPayment
        
#        print('Montly interest: ' + str(monthlyInterest))
#        print('Monthly unpaid balance: ' + str(monthlyUnpaidBalance))
#        print('New Balance: ' + str(newBalance))
#        print('')


#print('Lowest Payment: ' + str(round(lowestMonthlyPayment(3926, 0.2),2)))

print('Lowest Payment: ' + str(lowestMonthlyPayment(3329, 0.2)))

#print(minMonthlyPayment)

#print("Lowest Payment: " + str(payment))