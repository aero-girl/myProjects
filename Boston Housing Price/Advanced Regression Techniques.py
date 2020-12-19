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
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
# from pandas_profiling import ProfileReport
# import pandas_profiling


#from functions import *
    
#Options for pandas
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)

# Read in data train.csv and test.csv
column_names = ['capitaCrimeRate', 'residenLandZone', 'nonRetailBusinessAcres', 'trackBoundRiver', 'nitricOxidesConcentra', 'AvgNumRoom', 'buildingAge', 'distanceEmployCenter', 'radialHighWay', 'propertyTaxRate', 'teacherRatioTown', 'blackbyTown', 'workingPoorNeigh', 'medianValue']
df_train = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

df = pd.read_csv('housing.csv')
# df_train_report = ProfileReport(df)
# df_train_report.to_file("df_info_report.html")
# df_train_report

#%%#Part 1: Exploratory Data Analysis
# Add column names

print("Size of taining data is",df_train.shape)
print("Info of training data",df_train.info())



df_train.dtypes
df_train.head()
df_train.describe()

# Check for missing values
df_train.isnull().sum()


#%% Finding variables which are useful for prediction


# Find columns with strong correlation to target variable
corr = df_train.corr()
subjective_corr = df_train.corr()
subjective_corr[np.abs(subjective_corr)<.45] = 0


# Plotting the heatmap of correlation between features
plt.figure(figsize=(12, 12))
sns.set(font_scale=1)
sns.heatmap(subjective_corr, cmap='coolwarm',linewidths=1.5, annot=True, square=True,
                fmt='.2f', annot_kws={'size': 10})
plt.title('Pearson Correlation of Matrix', y=1.05, size=15)
plt.show()


#The target variable : 'medianValue'
#Box Plot and Distribution Plot for Dependent variable medianValue
plt.figure(figsize=(20,3))

plt.subplot(1,2,1)
sns.boxplot(df_train.medianValue)
plt.title('Box Plot of medianValue')

plt.subplot(1,2,2)
sns.distplot(a=df_train.medianValue)
plt.title('Distribution Plot of MEDV')
# Target variable medianValue is normally distributed.

#since some of these features shows quite good and very good correlation with our predictive variable Houese Price(MEDV)
df_train1 = df_train.drop(['capitaCrimeRate','residenLandZone','nonRetailBusinessAcres', 'buildingAge','nitricOxidesConcentra','distanceEmployCenter','radialHighWay', 'trackBoundRiver', 'blackbyTown'], axis = 1)


# Check first few lines
df_train1.head(5)

# Plot pairwise relationships in a dataset.
sns.pairplot(df_train1)
plt.title('Pairwise relationships', y=1.05, size=15)

#description about data
desc = df_train1.describe().round(2)
desc

#%% Build Model
from sklearn.model_selection import train_test_split


features = df_train1.drop(['medianValue'], 1)
prices = df_train1['medianValue']

# Split data to 80% training data and 20% of test to check the accuracy of our model
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=42)
print("Training and testing split by 80/20 was successful")

#%% Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets 
lm.fit(X_train, y_train)

# Model prediction on test data
y_pred_linear = lm.predict(X_test)

r2_lin = lm.score(X_test, y_test)

# Model Evaluation metrics
print('R^2:',metrics.r2_score(y_test, y_pred_linear))
print('MSE:',metrics.mean_squared_error(y_test, y_pred_linear))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred_linear)))

# Visualizing the differences between actual prices and predicted values
plt.scatter(y_test, y_pred_linear,alpha=0.5,  color='orange')
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Predicited Values vs True Values ")
plt.show()

sns.distplot(y_test-y_pred_linear)
plt.xlabel('Residual',size=12)
plt.ylabel('Frquency',size=12)
plt.title('Distribution of Residuals',size=15)
plt.show()

#%% Lasso Regression
from sklearn.linear_model import Lasso
score_calc = 'neg_mean_squared_error'

lasso = Lasso(alpha=0.3)

# Train the model using the training sets 
lasso.fit(X_train, y_train)

# Model prediction on test data
y_pred_lasso = lasso.predict(X_test)

# Return the coefficient of determination R^2 of the prediction.
r2_lasso = lasso.score(X_test, y_test)

param_grid={'alpha':np.arange(1,10,110)} #range from 1-500 with equal interval of 10
lasso = Lasso() 

# Define GridCV
lasso_best_alpha = GridSearchCV(lasso, param_grid, cv = 3, scoring='r2') 

# Train the model using the training sets 
lasso_best_alpha.fit(X_train,y_train)

# Model prediction on test data
y_best_alpha_pred_lasso = lasso_best_alpha.predict(X_test)

# Return the coefficient of determination R^2 of the prediction.
r2_lasso_best_alpha = lasso_best_alpha.score(X_test, y_test)

[y_pred_lasso, y_best_alpha_pred_lasso]
[r2_lasso, r2_lasso_best_alpha]

print("Best alpha for Lasso Regression:",lasso_best_alpha.best_params_)


#%% Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()

# Train the model using the training sets 
reg.fit(X_train, y_train)

# Model prediction on test data
y_pred = reg.predict(X_test)

# Return the coefficient of determination R^2 of the prediction.
r2_reg = reg.score(X_test, y_test)

# Create the parameter grid based on the results of random search 
parameters = {
    'max_depth': [70, 80, 90, 100],
    'n_estimators': [900, 1000, 1100]
}
# # Create the parameter grid based on the results of random search 
# parameters = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
reg_best_alpha = GridSearchCV(reg, parameters, cv = 3, n_jobs = -1, verbose = 1,scoring='r2')


# Train the model using the training sets 
reg_best_alpha.fit(X_train, y_train)

# Model prediction on test data
y_best_alpha_reg = reg_best_alpha.predict(X_test)

# Return the coefficient of determination R^2 of the prediction.
r2_reg_best_alpha = reg_best_alpha.score(X_test, y_test)

[r2_reg, r2_reg_best_alpha]

#%% Create results
Resultsdata = {"Models": ["Linear Regression","Lasso Regression",
                          "Lasso Regression (K-Fold CV)","Random Forest", "Random Forest (K-Fold CV)"],
               "R^2": [r2_lin*100,r2_lasso*100,r2_lasso_best_alpha*100,r2_reg*100,r2_reg_best_alpha*100]}

# Load into dataframe
dataFrame = pd.DataFrame(data=Resultsdata);

results=pd.DataFrame({'Linear Regression':[r2_lin*100],'Lasso Regression' :[r2_lasso*100],'Lasso Regression (K-Fold CV)':[r2_lasso_best_alpha*100],
                      'Random Forest':[r2_reg*100], 'Random Forest (K-Fold CV)':[r2_reg_best_alpha*100]},index=['Model R^2'])
results.plot(kind='bar',alpha=0.7,grid=True,title='Interpreting Results',rot=0,figsize=(10,7),legend=True,colormap='jet')
results

# Visualizing the differences between actual prices and predicted values
plt.scatter(y_test, y_pred_linear,alpha=0.5,  color='orange',label='Linear')
plt.scatter(y_test, y_best_alpha_pred_lasso , alpha=0.5,  color='blue', label='Lasso (tuned)')
plt.scatter(y_test, y_best_alpha_reg,alpha=0.5,  color='red', label='Random Forest (tuned)')
plt.xlabel("True Prices")
plt.ylabel("Predicted prices")
plt.legend()
plt.show()

plt.subplot(3, 2, 1)
plt.scatter(y_test, y_pred_linear,alpha=0.5,  color='orange',label='Linear')
plt.xlabel("True Prices")
plt.ylabel("Predicted prices")
plt.legend(loc="best")
plt.subplot(3, 2, 2)
plt.scatter(y_test, y_pred_lasso , alpha=0.5,  color='green', label='Lasso')
plt.xlabel("True Prices")
plt.ylabel("Predicted prices")
plt.legend(loc="best")
plt.subplot(3, 2, 3)
plt.scatter(y_test, y_best_alpha_pred_lasso , alpha=0.5,  color='blue', label='Lasso (tuned)')
plt.xlabel("True Prices")
plt.ylabel("Predicted prices")
plt.legend(loc="best")
plt.subplot(3, 2, 4)
plt.scatter(y_test, y_pred,alpha=0.5,  color='red', label='Random Forest')
plt.xlabel("True Prices")
plt.ylabel("Predicted prices")
plt.legend(loc="best")
plt.subplot(3, 2,5)
plt.scatter(y_test, y_best_alpha_reg,alpha=0.5,  color='gray', label='Random Forest (tuned)')
plt.xlabel("True Prices")
plt.ylabel("Predicted prices")
plt.legend(loc="best")
plt.show()