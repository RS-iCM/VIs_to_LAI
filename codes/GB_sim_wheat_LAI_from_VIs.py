#############################################################################
### Grediant Boost regression model to simulate LAI using VIs data for Wheat
### Input: Barley VIs and LAI data "Barley_LAI_n_VIs.csv"
### Output: Pickle model "pickle_gradient_boost_Wheat.pkl"
### Last modified: June 19, 2025
###########################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor
import pickle

df = pd.read_csv('data/Wheat_LAI_n_VIs.csv')  # Read rice LAI and VIs data
log10_log_DOY = np.log10(np.log(df['DOY'].to_numpy()))
data = df[['MTVI1', 'NDVI', 'OSAVI', 'RDVI']].to_numpy()
data = np.c_[log10_log_DOY, data]
#log_LAI = np.log10(df['LAI'].to_numpy())
LAI = df['LAI'].to_numpy()

#print('log_LAI shape:', log_LAI.shape)
#print('data shape:', data.shape)

#train_input, test_input, train_target, test_target = train_test_split(data, log_LAI, test_size=0.2, random_state=42)
train_input, test_input, train_target, test_target = train_test_split(data, LAI, test_size=0.2, random_state=42)

# Gradient Boost regression model
gb = GradientBoostingRegressor(random_state=42)
scores = cross_validate(gb, train_input, train_target,
                       return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

gb.fit(train_input, train_target)

# Save to file in the current working directory
pkl_filename = "models/pickle_gradient_boost_Wheat.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(gb, file)
