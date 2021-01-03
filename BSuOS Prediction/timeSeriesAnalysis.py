# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 06:16:52 2020
# Porterbrook Data Science Challenge
@author: Gavi
"""
# !cls
# %reset -f
#%% Import functions
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from matplotlib import style
style.use('fivethirtyeight')
import pandas as pd
from missing_values import missing_zero_values_table
from ADF_test import adf_test
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools

#Options for pandas
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)

#%% Import data
data = pd.read_csv('bsuos_project_data.csv')

# Check dimensions
print('Size of data', data.shape)
data.head(3)
data.tail(3)
data.info()

#%% # Work out the number and percentage of missing values in each column
missing_zero_values_table(data)

#%% Deal with missing values
# Gather columns of missing values
missing_col = ['wind_mw_value','dadf_mw_value','residual_mw_value']

# Technique: Using mean to impute the missing values
for i in missing_col:
 data.loc[data.loc[:,i].isnull(),i]=data.loc[:,i].mean()
missing_zero_values_table(data)

#%% Down-sample data

# Convert settlement_date to a datetime format
data.loc[:,'settlement_date'] = pd.to_datetime(data.loc[:,'settlement_date'], format='%d/%m/%Y')

# Check data types of each column
data.dtypes

# Set the settlement_date as the index of the dataframe
data = data.set_index('settlement_date')

## Add columns with year, month, and weekday name
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['Weekday Name'] = data.index.day_name()

# Check dimensions
data.head(3)

# Save csv to folder
data.to_csv (r'C:\Users\Gavi.DESKTOP-FVETBND\Dropbox\Documents\Job Applications\Porterbrook\Data science test\export_dataframe1.csv', index = True, header=True)

# Plot bsuos data
cols_plot = ['wind_mw_value', 'dadf_mw_value', 'residual_mw_value', 'bsuos']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

#%% Resample dataset for time-series analysis
# Extract relevant data, bsuos
data_bsuos = data[['bsuos']].copy()

# Downsampling: Where you decrease the frequency of the samples, such as from days to months.
data_bsuos_daily = data_bsuos.resample('D').mean()
print('Size of data_bsuos_daily data:', data_bsuos_daily.shape)
data_bsuos_daily.shape
data_bsuos_daily.plot(y ='bsuos',figsize=(30,20), color="blue",linewidth=5,
                      title="Downsampled BSUoS (Daily)", fontsize=30)
plt.show()

# Seasonal decomposition of daily resampled data
start ='2015-01-01' 
stop ='2018-07-01'
result = seasonal_decompose(data_bsuos_daily[start:stop], model='multiplicative')
result.plot()
plt.show()

# Resample weekly
data_bsuos_weekly = data_bsuos.resample('W').mean()
print('Size of data_bsuos_weekly data:', data_bsuos_weekly.shape)
data_bsuos_weekly.plot(y ='bsuos',figsize=(30,20), color="red",linewidth=5,
                      title="Downsampled BSUoS (Weekly)", fontsize=30)
plt.show()

# Seasonal decomposition of weekly resampled data
result = seasonal_decompose(data_bsuos_weekly[start:stop], model='multiplicative')
result.plot()
plt.show()

#Resample calendar month begin
data_bsuos_monthly = data_bsuos.resample('MS').mean()
print('Size of data_bsuos_monthly data:', data_bsuos_monthly.shape)
data_bsuos_monthly.plot(y ='bsuos',figsize=(30,20), color="green",linewidth=5,
                      title="Downsampled BSUoS (Monthly)", fontsize=30)
plt.show()

# Seasonal decomposition of monthly resampled data
result = seasonal_decompose(data_bsuos_monthly[start:stop], model='multiplicative')
result.plot()
plt.show()

# Start and end of the date range to extract
start, stop = '2015-01-01', '2015-03-01'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(data.loc[start:stop, 'bsuos'], marker='.', linestyle='-', linewidth=0.5, label='half hour data')
ax.plot(data_bsuos_daily.loc[start:stop, 'bsuos'], marker='*', markersize=8, linestyle='-', label='Daily Mean Resample')
ax.plot(data_bsuos_weekly.loc[start:stop, 'bsuos'], marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.plot(data_bsuos_monthly.loc[start:stop, 'bsuos'], marker='.', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('BSUoS')
ax.legend();

#%% Build model
# We are interested in predicting bsuos
adf_test(data_bsuos_daily)
print('If p< 0.05 ; Data is stationary, if p>0.05; Data is not stationary')

# Auto-correlation and partial correlation
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_bsuos_daily,lags=20,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_bsuos_daily,lags=20,ax=ax2)

#%% # Uncomment to use grid-cv
# p = d = q = range(0, 2)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# #use a “grid search” to find the optimal set of parameters that yields the best performance for our model.
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(data_bsuos_daily,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)

#             results = mod.fit()

#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue

#%% Fitting the SARIMA model
mod = sm.tsa.statespace.SARIMAX(data_bsuos_daily,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
# Fit model
results = mod.fit()

# Plot residulas and check if acceptable
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Validating forecasts
pred = results.get_prediction(start=pd.to_datetime('2018-04-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = data_bsuos_daily.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Prediction', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Settlement Date')
ax.set_ylabel('BSUoS (£)')
plt.legend()
plt.show()


#%% Extract forecasted and truth
y_forecasted = pred.predicted_mean
y_truth = data_bsuos_daily ['2018-04-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth['bsuos']) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
print('A mean squared error of zero indicates perfect skill, or no error.')
print('As with the mean squared error, an RMSE of zero indicates no error.')

# Predict steps ahead
pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()

ax = data_bsuos_daily.plot(label='observed', figsize=(20,10))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Settlement Date')
ax.set_ylabel('BSUoS (£)')
plt.legend()
plt.show()