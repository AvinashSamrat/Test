# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:42:51 2020

@author: avinash samrat
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
FileName ='IELTS.csv'
dataset = pd.read_csv(FileName)
X = dataset.iloc[:, 2:6].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
prediction=np.array([[6.5,7.5,8.5,7.5]])
print(regressor.predict(prediction))

#We Can Also do This
data={"Reading Bands":[6.5], "Writing Bands":[7.5], "Listening Bands":[8.5],"Speaking Bands":[7.5]}
df= pd.DataFrame(data)
z_pred=regressor.predict(df)
z_pred

#The Answer We Got is 7.55 Band Overall