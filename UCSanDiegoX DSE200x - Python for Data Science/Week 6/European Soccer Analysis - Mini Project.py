#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 06:18:38 2019

@author: Jake
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Player_Attributes as df
cnx = sqlite3.connect('../Week 5/Week-5-Exercises-2/database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

# Count NaN, drop NaN values
print('\nNaN values -before- dropping:')
print(df.isnull().sum().sum())
df.dropna(inplace = True)
print('\nNaN values -after- dropping:')
print(df.isnull().sum().sum())

# Count how many prefer left/right
preferred_foot = df['preferred_foot'].value_counts()
print('\nPreferred foot:\n' + preferred_foot.to_string())

# Display preferred foot as pie chart
fig1, ax1 = plt.subplots()
ax1.pie(preferred_foot,
        labels = ['Right', 'Left'],
        autopct = '%1.1f%%',
        explode = [0.1, 0.1],
        shadow = True)
ax1.axis('equal')
plt.title('Preferred Foot',
          fontdict = {'fontsize' : 18})
plt.show()

# Turn preferred foot into integer for analysis
df.loc[df['preferred_foot'] == 'right', 'preferred_foot'] = 0
df.loc[df['preferred_foot'] == 'left', 'preferred_foot'] = 1

# Pull out foot-related skills
foot_skills = df[['preferred_foot',
                  'crossing',
                  'finishing',
                  'short_passing',
                  'volleys',
                  'dribbling',
                  'curve',
                  'free_kick_accuracy',
                  'long_passing',
                  'ball_control',
                  'shot_power',
                  'long_shots',
                  'penalties']]

# Compute and plot correlations
correlations = foot_skills.corr()
ax2 = sns.heatmap(correlations, square = True)
plt.title('Correlations', fontdict = {'fontsize' : 18})
plt.show()

# Plot correlations with preferred foot
fig3, ax3 = plt.subplots()
ax3.plot(correlations['preferred_foot'], 'bo--')
plt.grid(True)
plt.xlabel('Foot Skills')
plt.ylabel('Correlation with Preferred Foot')
plt.xticks(rotation = 90)
plt.title('Correlation of Preferred Foot vs. Other Foot Skills',
          fontdict = {'fontsize' : 18})
plt.show()