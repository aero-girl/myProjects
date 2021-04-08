# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:05:21 2020
My first Kaggle competition
Titanic: Machine Learning from Disaster
@author: Gavi
"""

import pandas as pd

#load training data from the system
train = pd.read_csv("train.csv") 

train.head()
train.describe()


# First dropping 'Cabin' column because it has a lot of null values.
train = train.drop(['Cabin'], 1, inplace=False) 

#delete the rows with empty values
train = train.dropna() 

#select the column representing survival 
y = train['Survived'] 

# drop the irrelevant columns and keep the rest
X = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True) 

# convert non-numerical variables to dummy variables
X = pd.get_dummies(train) 

from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(X, y)

# load the testing data
test = pd.read_csv("test.csv") 

# create a sub-dataset for submission file and saving it
ids = test[['PassengerId']] 

# drop the irrelevant and keeping the rest
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True) 

# fill (instead of drop) empty rows so that I would get the exact row number required for submission
test.fillna(2, inplace=True) 

# convert non-numerical variables to dummy variables
test = pd.get_dummies(test) 

predictions = dtc.predict(test)

# assign predictions to ids
results = ids.assign(Survived = predictions) 

# write the final dataset to a csv file.
results.to_csv("titanic-results.csv", index=False) 

