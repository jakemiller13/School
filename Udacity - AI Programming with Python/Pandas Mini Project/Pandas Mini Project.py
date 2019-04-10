#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 06:27:57 2018

@author: Jake
"""

# We import pandas into Python
import pandas as pd

# We read in a stock data data file into a data frame and see what it looks like
df = pd.read_csv('./GOOG.csv')

# We display the first 5 rows of the DataFrame
print(df.head())

# We load the Google stock data into a DataFrame
google_stock = pd.read_csv('./GOOG.csv', index_col = 'Date', usecols = ['Date', 'Adj Close'], parse_dates = True)
print()
print(google_stock.head())

# We load the Apple stock data into a DataFrame
apple_stock = pd.read_csv('./AAPL.csv', index_col = 'Date', usecols = ['Date', 'Adj Close'], parse_dates = True)
print()
print(apple_stock.head())

# We load the Amazon stock data into a DataFrame
amazon_stock = pd.read_csv('./AMZN.csv', index_col = 'Date', usecols = ['Date', 'Adj Close'], parse_dates = True)
print()
print(amazon_stock.head())

# We create calendar dates between '2000-01-01' and  '2016-12-31'
dates = pd.date_range('2000-01-01', '2016-12-31')

# We create an empty DataFrame that uses the above dates as indices
all_stocks = pd.DataFrame(index = dates)

# Change the Adj Close column label to Google
google_stock = google_stock.rename(columns = {'Adj Close':'Google'})
print()
print(google_stock.head())

# Change the Adj Close column label to Apple
apple_stock = apple_stock.rename(columns = {'Adj Close':'Apple'})
print()
print(apple_stock.head())

# Change the Adj Close column label to Amazon
amazon_stock = amazon_stock.rename(columns = {'Adj Close':'Amazon'})
print()
print(amazon_stock.head())

# We join the Google stock to all_stocks
all_stocks = all_stocks.join(google_stock)

# We join the Apple stock to all_stocks
all_stocks = all_stocks.join(apple_stock)

# We join the Amazon stock to all_stocks
all_stocks = all_stocks.join(amazon_stock)

print()
print(all_stocks.head())

# Check if there are any NaN values in the all_stocks dataframe
print('\nNaN values:', all_stocks.isnull().sum().sum())

# Remove any rows that contain NaN values
all_stocks.dropna(axis = 0, inplace = True)
print()
print(all_stocks.head())
print('\nNaN values:', all_stocks.isnull().sum().sum())

# Print the average stock price for each stock
print('\nMean:')
print(all_stocks.mean())

# Print the median stock price for each stock
print('\nMedian:')
print(all_stocks.median())

# Print the standard deviation of the stock price for each stock 
print('\nStandard deviation:')
print(all_stocks.std()) 

# Print the correlation between stocks
print('\nCorrelation:')
print(all_stocks.corr())

# We compute the rolling mean using a 150-Day window for Google stock
rollingMean = all_stocks.rolling(150).mean()
print('\nRolling mean:')
print(rollingMean.tail())

#### This is just copy/paste from lesson:

# We import matplotlib into Python
import matplotlib.pyplot as plt

# We plot the Google stock data
plt.plot(all_stocks['Google'])

# We plot the rolling mean ontop of our Google stock data
plt.plot(rollingMean)
plt.legend(['Google Stock Price', 'Rolling Mean'])
plt.show()