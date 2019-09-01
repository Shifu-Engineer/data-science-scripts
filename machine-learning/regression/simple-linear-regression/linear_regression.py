#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:29:15 2019

@author: bruce
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
feature = dataset.iloc[:, :1].values
target = dataset.iloc[:, 1].values

# Split data into Training set and Testing set
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size = 1/3, random_state = 0)

# Fitting linear regression to the train data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(feature_train, target_train)

# Predicting the Test set result
target_prediction = regressor.predict(feature_test)

# Visualizing the training set
plt.scatter(feature_train, target_train, color = 'red')
plt.plot(feature_train, regressor.predict(feature_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test data set
plt.scatter(feature_test, target_test, color = 'red')
plt.plot(feature_train, regressor.predict(feature_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')