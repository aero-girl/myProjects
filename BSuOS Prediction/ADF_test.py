#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 06:26:32 2020
calculating the Augmented Dickey-Fuller test on the Daily Female Births dataset.
@author: Gavi
"""
from statsmodels.tsa.stattools import adfuller

# Test for stationary
def adf_test(dataset):
     data_test = adfuller(dataset)
     print("1. ADF : ",data_test[0])
     print("2. P-Value : %0.10f" % data_test[1])
     print("3. Num Of Lags : ", data_test[2])
     print("4. Num Of Observations Used For ADF Regression:", data_test[3])
     print("5. Critical Values :")
     for key, val in data_test[4].items():
         print("\t", key, ": %0.3f" % val)
   
     