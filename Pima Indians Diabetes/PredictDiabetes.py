# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:30:15 2020

@author: Gavi
"""
#%% Import and read data into Spyder

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.impute import SimpleImputer
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

#Options for pandas
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)

# Read in data train.csv and test.csv
data = pd.read_csv('diabetes.csv')

#%%#Part 1: Data Cleaning

# Check for missing data or invalid values.
data.isnull().sum()
# There are no missing values. 

# Summarise the number of unique values in each column
print(data.nunique())

# calculate duplicates
dups = data.duplicated()
# report if there are any duplicates
print(dups.any())

# list all duplicate rows
print(data[dups])
# There are no duplicate rows values. 

# Check for statistics
data.describe()
# However, the following coloumns have an invalid zero value :
# ---> Glucose Conc, Blood Pressure, Skin Thickness, Insulin, BMI 

# mark zero values as missing or NaN
data.iloc[:,1:6] = data.iloc[:,1:6].replace(0, np.NaN)

# Drop row if more than 2 values are missing.
data.dropna(thresh=2, axis=0, inplace=True)

# Replace np.nan with median value. 
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'median')
imputer = SimpleImputer().fit(data.iloc[:,1:6])
data.iloc[:,1:6] = imputer.transform(data.iloc[:,1:6])


#%%#Part 2: Exploratory Data Analysis

corr=data.corr()
sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');

#%% Part 3: Build Model


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 5)
print("Training and testing split by 80/20 was successful")

# Logistic Regression

LR = LogisticRegression()

#fiting the model
LR.fit(X_train, y_train)

#prediction
y_pred = LR.predict(X_test)

#Accuracy
print("Accuracy ", LR.score(X_test, y_test)*100)

#Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()

print ("\nClassification Report : \n")
print(classification_report(y_test, y_pred))


#%%# Logistic Regression

LR = LogisticRegression()

#fiting the model
LR.fit(X_train, y_train)

#prediction
y_pred_LR = LR.predict(X_test)

#Accuracy
print("Accuracy ", LR.score(X_test, y_test)*100)

#Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_LR)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.title('Confusion matrix Logistic Regression')
plt.show()

print ("\nClassification Report Logistic Regression : \n")
print(classification_report(y_test, y_pred_LR))

#%% KNeighborsClassifier
KNN = KNeighborsClassifier()

#fiting the model
KNN.fit(X_train, y_train)

#prediction
y_pred_KNN = KNN.predict(X_test)

#Accuracy
print("Accuracy ", KNN.score(X_test, y_test)*100)

#Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_KNN)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.title('Confusion matrix KNeighborsClassifier')
plt.show()

print ("\nClassification Report KNeighborsClassifier: \n")
print(classification_report(y_test, y_pred_KNN))

#%% RandomForrest
RF = RandomForestClassifier()

#fiting the model
RF.fit(X_train, y_train)

#prediction
y_pred_RF = RF.predict(X_test)

#Accuracy
print("Accuracy ", RF.score(X_test, y_test)*100)

#Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_RF)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title('Confusion matrix RandomForestClassifier')
plt.show()

print ("\nClassification Report RandomForestClassifier, Accuracy: \n",RF.score(X_test, y_test)*100 )
print(classification_report(y_test, y_pred_RF))


