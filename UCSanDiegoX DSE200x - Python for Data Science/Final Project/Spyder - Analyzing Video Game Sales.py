#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 08:02:48 2019
@author: Jake
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset: https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings
# Describe dataset - organized by Global Sales
# Sales are in millions of units
# Metacritic scores

# Load data frame
df = pd.read_csv('Video_Games_Sales_as_of_22_Dec_2016.csv')

# Explore shape
print('--- Shape ---\n' + str(df.shape))
# Includes 16719 games

# Check columns
print('\n--- Columns ---\n' + str(df.columns))

# Check out head
print('\n--- Head ---\n' + str(df.head()))

# Look at one entry
print('\n--- First Entry ---\n' + str(df.iloc[0]))

# Find NaN values
print('\n--- Rows with NaN ---\n' + str(df[df.isnull().any(axis = 1)].shape))
# Can tell there are NaN values in each column
# Too many rows to drop all NaN values
# Lets look at just names

print('\n--- Names that are "NaN" ---\n' + str(df[df['Name'].isnull()]))
# Only 2 names NaN, so let's drop those since we can't get any useful info

df.dropna(axis = 0, subset = ['Name'], inplace = True)

# Check shape afterwards - can tell 2 dropped
print('\n--- Shape after dropping "NaN" names: ---\n' + str(df.shape))

# Check names again just to be sure
print('\n--- Names that are "NaN" ---\n' + str(df[df['Name'].isnull()]))

# Let's look for some Final Fantasy games
print('\n--- Number of Final Fantasy Games included ---\n' + \
      str(df[df['Name'].str.contains('Final Fantasy')].shape[0]))
# That's a lot to display, so let's just look at top 25

print('\n--- Top 25 Final Fantasy Games ---\n' + \
      str(df[df['Name'].str.contains('Final Fantasy')][:25]))

# Let's look at one of my favorite games of all time
print('\n--- Final Fantasy IX ---\n' + \
      str(df.loc[175]))

# Want to cut out reviews that have <5 ratings for either critic or user
# First, let's check one thing
print('\n--- Python types ---')
print('Critic_Score type: ' + str(type(df['Critic_Score'][0])))
print('User_Score type: ' + str(type(df['User_Score'][0])))

# Let's cast User_Score to float64
try:
    df['User_Score'] = df['User_Score'].astype(np.float64)
except ValueError as e:
    print('\n--- Oops! --- \nValueError: ' + str(e))

# Aha! Now we know what's going on
print('\n--- Number of "TBD" in User_Score before dropping ---\n' + \
      str(len(df[df['User_Score'] == 'tbd'])))

# Let's just drop them all
df.drop(df[(df['User_Score'] == 'tbd')].index, inplace = True)
print('\n--- Number of "TBD" in User_Score after dropping ---\n' + \
      str(len(df[df['User_Score'] == 'tbd'])))

# And now try casting to float64 again
df['User_Score'] = df['User_Score'].astype(np.float64)

# Find indices with some threshold for critics, see if overlap with users
print('\n--- Titles with <5 reviews from both critics and users ---\n' +
      str(df[(df['Critic_Count'] < 5) & (df['User_Count'] < 5)]))

# Check shape before dropping games
print('\n--- Shape before dropping games with <5 reviews ---\n' + \
      str(df.shape))

# Drop title with <5 reviews from both critics and users
df.drop(df[(df['Critic_Count'] < 5) & (df['User_Count'] < 5)].index,
           inplace = True)

# Check shape afterwards - can tell 23 dropped
print('\n--- Shape after dropping games with <5 reviews ---\n' + \
      str(df.shape))

# Copy dataframe so we can keep original unaltered
integer_df = df.copy()

# Translate all strings to integers
# There are developers and ratings which are stored numerical
# Need all to be same type (numerical) for correlation
integer_df['Developer'] = integer_df['Developer'].astype(str)
integer_df['Rating'] = integer_df['Rating'].astype(str)

def integer_dictionary(string_column):
    '''
    Returns a dictionary of {string : integer} for a column/series in dataframe
    '''
    return {string : i for i, string in \
            enumerate(np.unique(integer_df[string_column]))}

platform_ints = integer_dictionary('Platform')
genre_ints = integer_dictionary('Genre')
developer_ints = integer_dictionary('Developer')
rating_ints = integer_dictionary('Rating')

def apply_int_trans(string_column, integer_dict):
    '''
    Translates all instances of a string to an integer within one column
    '''
    return integer_df[string_column].apply(lambda x: integer_dict[x])

integer_df['Platform'] = apply_int_trans('Platform', platform_ints)
integer_df['Genre'] = apply_int_trans('Genre', genre_ints)
integer_df['Developer'] = apply_int_trans('Developer', developer_ints)
integer_df['Rating'] = apply_int_trans('Rating', rating_ints)

# Correlation features
correlation_features = ['Platform',
                        'Genre',
                        'NA_Sales',
                        'EU_Sales',
                        'JP_Sales',
                        'Global_Sales',
                        'Critic_Score',
                        'User_Score',
                        'Developer',
                        'Rating']

# Heatmap
sns.heatmap(integer_df[correlation_features].corr(), cmap = 'YlGnBu')
plt.title('Correlation between Platform, Genre, \nSales and Game Scores')
plt.show()

def sales_by_genre(dataframe, region):
    '''
    Returns a dictionary of {genre : sales in millions of units} for "region"
    '''
    return {genre : dataframe[dataframe['Genre'] == genre][region].sum() \
            for genre in genre_ints.keys()}

na_sales_by_genre = sales_by_genre(df, 'NA_Sales')
eu_sales_by_genre = sales_by_genre(df, 'EU_Sales')
jp_sales_by_genre = sales_by_genre(df, 'JP_Sales')

genre_indices = range(len(na_sales_by_genre))
genre_width = np.min(np.diff(genre_indices))/4

fig, ax = plt.subplots(figsize = (10, 8))
bar1 = ax.bar(genre_indices - genre_width, na_sales_by_genre.values(),
              genre_width, color = 'blue')
bar2 = ax.bar(genre_indices, eu_sales_by_genre.values(),
              genre_width, color = 'red')
bar3 = ax.bar(genre_indices + genre_width, jp_sales_by_genre.values(),
              genre_width, color = 'green')

plt.xticks(rotation = 90)
ax.set_xticks(genre_indices - genre_width/5)
ax.set_xticklabels((na_sales_by_genre.keys()))

plt.title('Units Sold by Genre')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['North America', 'Europe', 'Japan'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_genre), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()

def sales_by_platform(dataframe, region):
    '''
    Returns a dictionary of {platform : sales in millions of units}
    for "region"
    '''
    return {platform : dataframe[dataframe['Platform'] == platform][region].\
            sum() for platform in platform_ints.keys()}

na_sales_by_platform = sales_by_platform(df, 'NA_Sales')
eu_sales_by_platform = sales_by_platform(df, 'EU_Sales')
jp_sales_by_platform = sales_by_platform(df, 'JP_Sales')

platform_indices = range(len(na_sales_by_platform))
platform_width = np.min(np.diff(platform_indices))/4

fig, ax = plt.subplots(figsize = (10, 8))

bar1 = ax.bar(platform_indices - platform_width, na_sales_by_platform.values(),
              platform_width, color = 'blue')
bar2 = ax.bar(platform_indices, eu_sales_by_platform.values(),
              platform_width, color = 'red')
bar3 = ax.bar(platform_indices + platform_width, jp_sales_by_platform.values(),
              platform_width, color = 'green')

plt.xticks(rotation = 90)
ax.set_xticks(platform_indices - platform_width/5)
ax.set_xticklabels((na_sales_by_platform.keys()))

plt.title('Units Sold by Platform')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['North America', 'Europe', 'Japan'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_platform), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()

devs, dev_counts = np.unique(integer_df['Developer'], return_counts = True)
dev_counts_dict = {i : [dev_counts, i] for i, dev_counts in \
                   enumerate(dev_counts)}
top_devs_integers = sorted(dev_counts_dict.values(), reverse = True)

# Switch keys and values so we can lookup developers by integers
rev = dict({(v, k) for k, v in developer_ints.items()})

# Top developer is...
print('\n--- Top Developer ---\n', rev[top_devs_integers[0][1]])

# Oops... Let's look at the top 20 not-NaN developers by games developed
top_20_integers = top_devs_integers[1:21]
top_20_devs = [rev[dev[1]] for dev in top_20_integers]
print('\n--- Top 20 Developers ---')
for i, dev in enumerate(top_20_devs):
    print(str(i + 1) + '. ' + dev)

# Huh...never heard of Vicarious Visions
# Oh neat... Developed a lot of Guitar Hero and Crash games
print('\n--- Games Developed by Vicarious Visions ---')
print(df[df['Developer'] == 'Vicarious Visions']\
      [['Name', 'Developer', 'Critic_Score']].head(20))

def sales_by_developer(region):
    '''
    Returns a dictionary of {developer : sales in millions of units}
    for "region"
    '''
    return {developer : df[df['Developer'] == developer][region].sum() for \
            developer in top_20_devs}

na_sales_by_developer = sales_by_developer('NA_Sales')
eu_sales_by_developer = sales_by_developer('EU_Sales')
jp_sales_by_developer = sales_by_developer('JP_Sales')

developer_indices = range(len(na_sales_by_developer))
developer_width = np.min(np.diff(developer_indices))/4

fig, ax = plt.subplots(figsize = (10, 8))

bar1 = ax.bar(developer_indices - developer_width,
              na_sales_by_developer.values(),
              developer_width, color = 'blue')
bar2 = ax.bar(developer_indices,
              eu_sales_by_developer.values(),
              developer_width, color = 'red')
bar3 = ax.bar(developer_indices + developer_width,
              jp_sales_by_developer.values(),
              developer_width, color = 'green')

plt.xticks(rotation = 90)
ax.set_xticks(developer_indices - developer_width/5)
ax.set_xticklabels((na_sales_by_developer.keys()))

plt.title('Units Sold for Top 20 Developers')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['North America', 'Europe', 'Japan'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_developer), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()

# Correlation between user/critic score obvious
# How about correlation between developer and rating
# for some reason, interpreting E10+ as numeric
df['Rating'] = df['Rating'].astype(str)
ratings, ratings_counts = np.unique(df['Rating'], return_counts = True)

# Let's check out ratings and their counts really quickly
print('\n--- Sales by Rating ---')
for i in zip(ratings, ratings_counts):
    print(i)

# Ok, we can ignore A0, EC, K-A, RP, nan
# Only current ratings are E, E10+, T, M, A0 anyways

def sales_by_rating(rating, dev_list):
    '''
    Returns dictionary of {rating : sales in millions of units} for each
    "developer"
    '''
    return {developer : df[(df['Developer'] == developer) & \
                           (df['Rating'] == rating)]['Global_Sales'].sum() \
                            for developer in dev_list}

e_sales_by_developer = sales_by_rating('E', top_20_devs)
e10_sales_by_developer = sales_by_rating('E10+', top_20_devs)
t_sales_by_developer = sales_by_rating('T', top_20_devs)
m_sales_by_developer = sales_by_rating('M', top_20_devs)

rating_indices = range(len(e_sales_by_developer))
rating_width = np.min(np.diff(rating_indices))/5

fig, ax = plt.subplots(figsize = (10, 8))

bar1 = ax.bar(rating_indices - 1.5*rating_width, e_sales_by_developer.values(),
              rating_width, color = 'blue')
bar2 = ax.bar(rating_indices - 0.5*rating_width, e10_sales_by_developer.values(),
              rating_width, color = 'red')
bar3 = ax.bar(rating_indices + 0.5*rating_width, t_sales_by_developer.values(),
              rating_width, color = 'green')
bar4 = ax.bar(rating_indices + 1.5*rating_width, m_sales_by_developer.values(),
              rating_width, color = 'orange')

plt.xticks(rotation = 90)
ax.set_xticks(developer_indices - developer_width/5)
ax.set_xticklabels((na_sales_by_developer.keys()))

plt.title('Units Sold by Rating for Top 20 Developers')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['E', 'E10+', 'T', 'M'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_developer), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()

# Ok, Nintendo is distorting plot, let's redo it without Nintendo
top_19_devs = [dev for dev in top_20_devs if not dev == 'Nintendo']

e_sales_by_developer = sales_by_rating('E', top_19_devs)
e10_sales_by_developer = sales_by_rating('E10+', top_19_devs)
t_sales_by_developer = sales_by_rating('T', top_19_devs)
m_sales_by_developer = sales_by_rating('M', top_19_devs)

rating_indices = range(len(e_sales_by_developer))
rating_width = np.min(np.diff(rating_indices))/5

fig, ax = plt.subplots(figsize = (10, 8))

bar1 = ax.bar(rating_indices - 1.5*rating_width, e_sales_by_developer.values(),
              rating_width, color = 'blue')
bar2 = ax.bar(rating_indices - 0.5*rating_width, e10_sales_by_developer.values(),
              rating_width, color = 'red')
bar3 = ax.bar(rating_indices + 0.5*rating_width, t_sales_by_developer.values(),
              rating_width, color = 'green')
bar4 = ax.bar(rating_indices + 1.5*rating_width, m_sales_by_developer.values(),
              rating_width, color = 'orange')

plt.xticks(rotation = 90)
ax.set_xticks(developer_indices - developer_width/5)
ax.set_xticklabels(top_19_devs)

plt.title('Units Sold by Rating for Top 19 Developers (Dropping Nintendo)')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['E', 'E10+', 'T', 'M'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_developer), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()

# Let's take a look specifically at recent generations of consoles
# Important to note this dataset is from before Switch
modern = ['PS3', 'PS4', 'Wii', 'WiiU', 'X360', 'XOne']
modern_df = pd.DataFrame()
for platform in modern:
    modern_df = pd.concat([modern_df, df[df['Platform'] == platform]])

na_sales_by_genre = sales_by_genre(modern_df, 'NA_Sales')
eu_sales_by_genre = sales_by_genre(modern_df, 'EU_Sales')
jp_sales_by_genre = sales_by_genre(modern_df, 'JP_Sales')

genre_indices = range(len(na_sales_by_genre))
genre_width = np.min(np.diff(genre_indices))/4

fig, ax = plt.subplots(figsize = (10, 8))
bar1 = ax.bar(genre_indices - genre_width, na_sales_by_genre.values(),
              genre_width, color = 'blue')
bar2 = ax.bar(genre_indices, eu_sales_by_genre.values(),
              genre_width, color = 'red')
bar3 = ax.bar(genre_indices + genre_width, jp_sales_by_genre.values(),
              genre_width, color = 'green')

plt.xticks(rotation = 90)
ax.set_xticks(genre_indices - genre_width/5)
ax.set_xticklabels((na_sales_by_genre.keys()))

plt.title('Units Sold by Genre (Modern Consoles)')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['North America', 'Europe', 'Japan'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_genre), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()

def sales_by_platform(region):
    '''
    Returns a dictionary of {platform : sales in millions of units}
    for "region"
    '''
    return {platform : modern_df[modern_df['Platform'] == platform][region].\
            sum() for platform in modern}

na_sales_by_platform = sales_by_platform('NA_Sales')
eu_sales_by_platform = sales_by_platform('EU_Sales')
jp_sales_by_platform = sales_by_platform('JP_Sales')

platform_indices = range(len(modern))
platform_width = np.min(np.diff(platform_indices))/4

fig, ax = plt.subplots(figsize = (10, 8))

bar1 = ax.bar(platform_indices - platform_width, na_sales_by_platform.values(),
              platform_width, color = 'blue')
bar2 = ax.bar(platform_indices, eu_sales_by_platform.values(),
              platform_width, color = 'red')
bar3 = ax.bar(platform_indices + platform_width, jp_sales_by_platform.values(),
              platform_width, color = 'green')

plt.xticks(rotation = 90)
ax.set_xticks(platform_indices - platform_width/5)
ax.set_xticklabels((na_sales_by_platform.keys()))

plt.title('Units Sold by Platform (Modern Consoles)')
plt.ylabel('Units Sold (millions)')
plt.legend(labels = ['North America', 'Europe', 'Japan'], loc = 'best')
plt.grid(which = 'major', axis = 'y')

for i in range(0, len(na_sales_by_platform), 2):
    plt.axvspan(i - 0.5, i + 0.5, facecolor = 'black', alpha = 0.1)

plt.show()