#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 06:55:44 2019

@author: Jake
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv('mystery.dat', header = None)

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

shrinkage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.58, 0.6, 0.7, 0.8, 0.9]

for i in shrinkage:
    lasso_clf = linear_model.Lasso(alpha = i)
    lasso_clf.fit(x, y)
    print('\n' + str(np.sum(lasso_clf.coef_ != 0)) + \
          ' Coefficients for lambda = ' + str(i))
    print(np.flatnonzero(lasso_clf.coef_) + 1)