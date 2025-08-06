##########################################################################
### Extra Trees regression model to simulate LAI using VIs data for Wheat
### Input: Wheat VIs and LAI data "Wheat_LAI_n_VIs.csv"
### Output: Pickle model "pickle_gradient_boost_Wheat_seq.pkl"
### Last modified: July 01, 2025
##########################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load data
csv_path = 'data/Wheat_LAI_n_VIs.csv'
df = pd.read_csv(csv_path)

# Create a sequential variable replacing DOY (1, 2, ..., n)
df = df.reset_index(drop=True)
seq = np.arange(1, len(df) + 1)

# Transform the sequential variable: log10(log(seq))
with np.errstate(divide='ignore'):
    log_seq = np.log(seq)
# Avoid -inf for seq=1 by shifting index or starting from 2 if needed
# Here, seq starts at 1 => log(1)=0 => log10(0) is -inf. To handle, add a small epsilon:
eps = 1e-6
log_seq_safe = np.log(seq + eps)
log10_log_seq = np.log10(log_seq_safe)

# Shrink factor α: smaller α → sequence matters less
alpha = 1
log10_log_seq_shrunk = log10_log_seq * alpha

# Feature matrix: shrunk sequential variable + four vegetation indices
X = np.column_stack([
    log10_log_seq_shrunk,
    df[['MTVI1', 'NDVI', 'OSAVI', 'RDVI']].values
])

# Target variable
y = df['LAI'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validation & training
gb = GradientBoostingRegressor(random_state=42)
scores = cross_validate(gb, X_train, y_train,
                        return_train_score=True, n_jobs=-1)
print(f"Train R²: {scores['train_score'].mean():.3f}, "
      f"Test R²: {scores['test_score'].mean():.3f}")

# Fit the model
gb.fit(X_train, y_train)

# Inspect feature importances
feat_names = ['seq', 'MTVI1', 'NDVI', 'OSAVI', 'RDVI']
for name, imp in zip(feat_names, gb.feature_importances_):
    print(f"{name:>6}: {imp:.3f}")

# Save the trained model
model_path = 'models/pickle_gradient_boost_Wheat_seq.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(gb, f)

print(f"Model saved to {model_path}")

