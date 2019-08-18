#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:30:49 2019

@author: Jake
"""

# Imports
import numpy as np
import pandas as pd
from scipy import stats
from math import exp, pi, sqrt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
 
# Load data
df = pd.read_csv('../Topic9-ContinuousDistributionFamilies/temperature.csv',
                 usecols = ['datetime', 'San Diego'])
df = df.dropna()
 
# Plot histogram, setup second plot
fig, ax1 = plt.subplots()
ax1.hist(df['San Diego'], bins = 100)
ax1.set_ylabel('Temperature (kelvin)')
ax2 = ax1.twinx()
ax2.set_ylabel('distribution')
ax1.set_title('San Diego')
plt.show()
 
# Get stats
mean = np.mean(df['San Diego'])
var = np.var(df['San Diego'], ddof = 1)
feb13 = np.mean(df.loc[df.datetime.str.contains('2013-02')]).values[0]

# Calculate convidence interval
t = stats.t.ppf(.90, df.shape[0]-1)
conf_upper = feb13 + t * sqrt(var) / sqrt(df.shape[0])
conf_lower = feb13 - t * sqrt(var) / sqrt(df.shape[0])

print('\n--- Statistics ---')
print('Mean: ' + str(mean))
print('Variance: ' + str(var))
print('Sample mean in Feb \'13: ' + str(feb13))

print('\n--- Confidence Intervals ---')
print('Upper: ' + str(conf_upper))
print('Lower: ' + str(conf_lower))

