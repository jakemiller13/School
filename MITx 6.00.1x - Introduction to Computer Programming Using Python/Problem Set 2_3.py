#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:07:26 2017

@author: Jake
"""

#balance = 320000
#annualInterestRate = 0.2

minMonthlyPayment = balance / 2
monthlyInterest = annualInterestRate / 12.0
monthlyPaymentLB = balance / 12
monthlyPaymentUB = (balance * (1 + monthlyInterest)**12) / 12

#z = 15
#while z > 0:
while True:
    
#    print('\nnew iteration')
#    print('minMonthlyPayment: ' + str(minMonthlyPayment))
    newBalance = balance
    
#    print('z = ' + str(z))
#    z -= 1
    
    for i in range(0,12):

#       monthlyInterest = annualInterestRate / 12.0
        monthlyUnpaidBalance = newBalance - minMonthlyPayment
        newBalance = monthlyUnpaidBalance + (monthlyInterest * monthlyUnpaidBalance)
        
#        print('Monthly unpaid balance: ' + str(monthlyUnpaidBalance))
#        print('New balance: ' + str(newBalance))

#    print('balance - newBalance: ' + str(round(balance - newBalance,2)))

    if newBalance <= 0.01 and newBalance >= 0.00:
        
#        print('if')
        print('Lowest Payment: ' + str(round(minMonthlyPayment,2)))
        break
    
    elif newBalance > 0.01:
        
#        print('elif')
        monthlyPaymentLB = minMonthlyPayment
        minMonthlyPayment = (monthlyPaymentUB + monthlyPaymentLB) / 2

    else:
        
#        print('else')
        monthlyPaymentUB = minMonthlyPayment
        minMonthlyPayment = (monthlyPaymentUB + monthlyPaymentLB) / 2

        
#Monthly interest rate = (Annual interest rate) / 12.0
#Monthly payment lower bound = Balance / 12
#Monthly payment upper bound = (Balance x (1 + Monthly interest rate)12) / 12.0