###################################################################
### XGBoost regression model to simulate LAI using VIs data
### Input: Rice VIs and LAI data "Rice_LAI_n_VIs.csv"
### Output: Pickle model "pickle_extra_trees_Rice.pkl"
### Last modified: May 23, 2024
###################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
import pickle

df = pd.read_csv('Rice_LAI_n_VIs.csv')  # Read rice LAI and VIs data
log10_log_DOY = np.log10(np.log(df['DOY'].to_numpy()))
data = df[['MTVI1', 'NDVI', 'OSAVI', 'RDVI']].to_numpy()
data = np.c_[log10_log_DOY, data]
#log_LAI = np.log10(df['LAI'].to_numpy())
LAI = df['LAI'].to_numpy()

#train_input, test_input, train_target, test_target = train_test_split(data, log_LAI, test_size=0.2, random_state=42)
train_input, test_input, train_target, test_target = train_test_split(data, LAI, test_size=0.2, random_state=42)

# XGBoost regression
xgb = XGBRegressor(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target,
                       return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

xgb.fit(train_input, train_target)

# Save to file in the current working directory
pkl_filename = "pickle_XGBoost_Rice.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(xgb, file)
