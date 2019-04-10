#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 06:20:06 2018

@author: Jake
"""

# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# solution script imports
#from solutions_univ import bar_chart_solution_1, bar_chart_solution_2

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.head())

plot = sb.barplot(x = ')