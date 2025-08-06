########################################################################
### Extra Trees regression model to simulate LAI using VIs data for Maize
### Input: Rice VIs and LAI data "Rice_LAI_n_VIs.csv"
### Output: Pickle model "pickle_extra_trees_Maize.pkl"
### Last modified: June 19, 2025
########################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesRegressor
import pickle

df = pd.read_csv('data/Maize_LAI_n_VIs.csv')  # Read maize LAI and VIs data
log10_log_DOY = np.log10(np.log(df['DOY'].to_numpy()))
data = df[['MTVI1', 'NDVI', 'OSAVI', 'RDVI']].to_numpy()
data = np.c_[log10_log_DOY, data]
LAI = df['LAI'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, LAI, test_size=0.2, random_state=42)

# Extra Trees regression
et = ExtraTreesRegressor(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target,
                       return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

et.fit(train_input, train_target)

# Save to file in the current working directory
pkl_filename = "models/pickle_extra_trees_Maize.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(et, file)
