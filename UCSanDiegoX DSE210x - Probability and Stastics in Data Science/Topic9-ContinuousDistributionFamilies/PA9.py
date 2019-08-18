#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:30:49 2019

@author: Jake
"""

# Imports
import numpy as np
import pandas as pd
from math import exp, pi, sqrt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
 
# Load data
df = pd.read_csv('temperature.csv', usecols = ['datetime', 'Detroit'])
df = df.dropna()
 
# Plot histogram, setup second plot
fig, ax1 = plt.subplots()
ax1.hist(df['Detroit'], bins = 100)
ax1.set_ylabel('Detroit')
ax2 = ax1.twinx()
ax2.set_ylabel('distribution')
 
# Get stats
max_temp = df.max()['Detroit']
min_temp = df.min()['Detroit']
plot_rng = np.linspace(min_temp, max_temp, 100)
 
# Distribution 1
mu_1, sigma_1 = 283, 11
Px_1 = [(1 / sqrt(2 * pi * sigma_1**2)) * \
        exp((-(x - mu_1)**2) / (2 * sigma_1**2)) \
        for x in plot_rng]
ax2.plot(plot_rng, Px_1, 'orange', label = 'Px_1')
 
# Distribution 2
mu_21, sigma_21, mu_22, sigma_22 = 276, 6, 293, 6.5
Px_2 = [0.5 * (1 / sqrt(2 * pi * sigma_21**2)) * \
        exp((-(x - mu_21)**2) / (2 * sigma_21**2)) + \
        0.5 * (1 / sqrt(2 * pi * sigma_22**2)) * \
        exp((-(x - mu_22)**2) / (2 * sigma_22**2)) \
        for x in plot_rng]
plt.plot(plot_rng, Px_2, 'black', label = 'Px_2')
 
# Distribution 3
mu_31, sigma_31, mu_32, sigma_32 = 276, 6.5, 293, 6
Px_3 = [0.5 * (1 / sqrt(2 * pi * sigma_31**2)) * \
        exp((-(x - mu_31)**2) / (2 * sigma_31**2)) + \
        0.5 * (1 / sqrt(2 * pi * sigma_32**2)) * \
        exp((-(x - mu_32)**2) / (2 * sigma_32**2)) \
        for x in plot_rng]
 
ax2.plot(plot_rng, Px_3, 'red', label = 'Px_3')
plt.legend()
plt.show()

# Probability between 281 and 291
prob = integrate.quad(lambda x: 0.5 * (1 / sqrt(2 * pi * sigma_31**2)) * \
                      exp((-(x - mu_31)**2) / (2 * sigma_31**2)) + \
                      0.5 * (1 / sqrt(2 * pi * sigma_32**2)) * \
                      exp((-(x - mu_32)**2) / (2 * sigma_32**2)), 281, 291)
print('\nProbability temp between 281K and 291K = ' + str(prob[0]))