#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 06:23:17 2017

@author: Jake
"""

def yearlyBalance(balance, annualInterestRate, monthlyPaymentRate):
    
    '''
    Given a balance, annual interest rate, and monthly payment rate,
    will calculate credit card balance at end of one year
    
    balance: the outstanding balance on the credit card
    
    annualInterestRate: annual interest rate as a decimal
    
    monthlyPaymentRate: minimum monthly payment rate as a decimal
    '''
    
    for i in range(0,12):
        
        monthlyInterest = annualInterestRate / 12.0
        minMonthlyPayment = monthlyPaymentRate * balance
        monthlyUnpaidBalance = balance - minMonthlyPayment
        balance = monthlyUnpaidBalance + (monthlyInterest * monthlyUnpaidBalance)
        
    return balance

print("Remaining balance: " + str(round(yearlyBalance(484, 0.2, 0.04),2)))

# Test Case 1:
# balance = 42
# annualInterestRate = 0.2
# monthlyPaymentRate = 0.04
# Remaining balance: 31.38

# Test Case 2:
# balance = 484
# annualInterestRate = 0.2
# monthlyPaymentRate = 0.04
# Remaining balance: 361.61


### This is the answer. Above is putting it in function
#for i in range(0,12):
        
#    monthlyInterest = annualInterestRate / 12.0
#    minMonthlyPayment = monthlyPaymentRate * balance
#    monthlyUnpaidBalance = balance - minMonthlyPayment
#    balance = monthlyUnpaidBalance + (monthlyInterest * monthlyUnpaidBalance)

#print("Remaining balance: " + str(round(balance,2)))