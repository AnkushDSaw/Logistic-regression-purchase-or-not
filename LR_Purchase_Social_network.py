# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:59:38 2022

@author: ankus
"""

# Logistic regression Step by Step for predicting item purchase or not
##  Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing dataset
dataset=pd.read_csv(r'C:\Users\ankus\OneDrive\Desktop\Naresh IT\April\28th,29th_April\Social_Network_Ads.csv')

#*********************    Variables
#Response
#* Purchased: purchased or not (1/0)

#Explanatory
#* Gender: male female
#* age: age of participant
# Userid is not useful we can drop it
dataset.drop(['User ID'],inplace=True, axis = 1)


# **************    Update some column Vlaues
#   gender: 1= male, 0-female
dataset['Gender'].replace('Female', 0, inplace=True)
dataset['Gender'].replace('Male', 1, inplace=True)


# Taking care of Missing Values and Null values
dataset.isnull().sum()


## *******************   Split the data
# We have to predict the HD column given the features.
X = dataset.drop(['Purchased'], axis = 1) # independent variable ( Remove mpg from X data)
y = dataset[['Purchased']] #dependent variable


#****************    Splitting Dataset- Xtrain and y Train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=0)

#************      Feature Scaling for Improving model Performance
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# ****************     Model buidling with logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

# ************     Predicting test set results
y_pred = classifier.predict(x_test)

# ***********     Evaluating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#****************    Accuracy of Model
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

# ********     Bias and Vriance
bias= classifier.score(x_train,y_train)
variance=classifier.score(x_test,y_test)
print(bias)
print(variance)