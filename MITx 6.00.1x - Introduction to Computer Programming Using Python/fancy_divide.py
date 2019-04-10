#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:56:56 2017

@author: Jake
"""

def fancy_divide(numbers,index):
    try:
        denom = numbers[index]
        for i in range(len(numbers)):
            numbers[i] /= denom
    except IndexError:
        print("-1")
    else:
        print("1")
    finally:
        print("0")

print('Fancy Divide')
fancy_divide([0, 2, 4], 1)
print()
fancy_divide([0, 2, 4], 4)
print()
fancy_divide([0, 2, 4], 0) #index error terminates code
print()



def fancy_divide2(numbers, index):
    try:
        denom = numbers[index]
        for i in range(len(numbers)):
            numbers[i] /= denom
    except IndexError:
        fancy_divide(numbers, len(numbers) - 1)
    except ZeroDivisionError:
        print("-2")
    else:
        print("1")
    finally:
        print("0")

print('Fancy Divide 2')
fancy_divide2([0, 2, 4], 1)
print()
fancy_divide2([0, 2, 4], 4)
print()
fancy_divide2([0, 2, 4], 0)
print()



def fancy_divide3(numbers, index):
    try:
        try:
            denom = numbers[index]
            for i in range(len(numbers)):
                numbers[i] /= denom
        except IndexError:
            fancy_divide(numbers, len(numbers) - 1)
        else:
            print("1")
        finally:
            print("0")
    except ZeroDivisionError:
        print("-2")

print('Fancy Divide 3')
fancy_divide3([0, 2, 4], 1)
print()
fancy_divide3([0, 2, 4], 4)
print()
fancy_divide3([0, 2, 4], 0)
print()



def fancy_divide4(list_of_numbers, index):
    try:
        try:
            raise Exception("0")
        finally:
            denom = list_of_numbers[index]
            for i in range(len(list_of_numbers)):
                list_of_numbers[i] /= denom
    except Exception as ex:
        print(ex)

print('Fancy Divide 4')
fancy_divide4([0, 2, 4], 0)
print()



def fancy_divide5(list_of_numbers, index):
    try:
        try:
            denom = list_of_numbers[index]
            for i in range(len(list_of_numbers)):
                list_of_numbers[i] /= denom
        finally:
            raise Exception("0")
    except Exception as ex:
        print(ex)

print('Fancy Divide 5')
fancy_divide5([0, 2, 4], 0)