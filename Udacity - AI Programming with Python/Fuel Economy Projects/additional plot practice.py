#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 07:35:21 2018

@author: Jake
"""

# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel_econ.csv')
#print(fuel_econ.head(5))

#TODO: Task 1: Plot the distribution of combined fuel mileage (column 'comb', in miles per gallon) by manufacturer (column 'make'), for all manufacturers with at least eighty cars in the dataset. Consider which manufacturer order will convey the most information when constructing your final plot. Hint: Completing this exercise will take multiple steps! Add additional code cells as needed in order to achieve the goal.


#make_counts = fuel_econ.groupby('make')['comb'].value_counts()

#make_counts = [x for x in fuel_econ['make'].value_counts() if x > 80]
#print(make_counts)

#sb.countplot(data = fuel_econ, x = 'VClass', hue = 'fuelType')

#order = (fuel_econ['make'].value_counts() > 80)
#
#index_end = (order).sum()
#
#print(index_end)
#
#sb.countplot(data = fuel_econ[:index_end], x = 'VClass', hue = 'fuelType')
#
#test1 = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline','Regular Gasoline'])]

make_counts = fuel_econ['make'].value_counts()
categories = make_counts[make_counts > 80].index

fuel_econ_sub = fuel_econ.loc[fuel_econ['make'].isin(categories)]
#
#fType = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline', 'Regular Gasoline'])]
#
#sb.countplot(data = fuel_econ_sub, x = 'VClass', hue = fType)

grid = sb.FacetGrid(data = fuel_econ, col = 'make')
grid.map(plt.hist, 'comb')