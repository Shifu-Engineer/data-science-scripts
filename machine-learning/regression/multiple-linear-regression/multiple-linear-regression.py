#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 04:37:29 2019

@author: bruce
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# get the data
dataset = pd.read_csv('50_Startups.csv')
feature = dataset.iloc[:, :-1].values
target = dataset.iloc[:, -1].values

# encoding categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_feature = LabelEncoder()
feature[:,3] = labelencoder_feature.fit_transform(feature[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
feature = onehotencoder.fit_transform(feature).toarray()

# avoiding the dummy variable trap
feature = feature[:, 1:]

# splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size = 0.20, random_state = 0)

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(feature_train, target_train)

# Predicting the target
target_prediction = regressor.predict(feature_test)