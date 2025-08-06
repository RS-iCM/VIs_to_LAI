########################################################################
### Lasso regression model to simulate LAI using VIs data for Maize
### Input: Rice VIs and LAI data "Rice_LAI_n_VIs.csv"
### Output: Pickle model "pickle_extra_trees_Maize.pkl"
### Last modified: June 19, 2025
########################################################################
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model    import LassoCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics         import r2_score, mean_squared_error

# Load data
df = pd.read_csv('data/Maize_LAI_n_VIs.csv')  
# compute your DOY feature
log10_log_DOY = np.log10(np.log(df['DOY'].to_numpy()))

X = np.c_[log10_log_DOY,
           df[['MTVI1','NDVI','OSAVI','RDVI']].to_numpy() ]
y = df['LAI'].to_numpy()

# Split train/test
train_input, test_input, train_target, test_target = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LassoCV pipeline
def build_lasso_pipeline(cv_folds=5, random_state=42):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lasso',  LassoCV(cv=cv_folds, random_state=random_state, n_jobs=-1))
    ])

pipeline = build_lasso_pipeline(cv_folds=5, random_state=42)

# Cross-validate on the training set
scoring = {'r2':'r2', 'rmse':'neg_root_mean_squared_error'}
cv_results = cross_validate(
    pipeline,
    train_input,
    train_target,
    scoring=scoring,
    return_train_score=True,
    cv=5,
    n_jobs=-1
)

def summarize_cv(scores):
    print(f"▶ CV Train R²: {scores['train_r2'].mean():.3f} | CV Test R²: {scores['test_r2'].mean():.3f}")
    print(f"▶ CV Train RMSE: {(-scores['train_rmse']).mean():.3f} | CV Test RMSE: {(-scores['test_rmse']).mean():.3f}")

summarize_cv(cv_results)

# Fit on full training set
pipeline.fit(train_input, train_target)

# held-out test evaluation
pred_test = pipeline.predict(test_input)
test_r2   = r2_score(test_target, pred_test)
test_rmse = np.sqrt(mean_squared_error(test_target, pred_test))
print(f"\n▶ Held-out Test R²: {test_r2:.3f} | Test RMSE: {test_rmse:.3f}")

# best alpha and coefficients
best_alpha   = pipeline.named_steps['lasso'].alpha_
coefficients = pipeline.named_steps['lasso'].coef_
print(f"\n▶ Optimal α: {best_alpha:.4f}")
print("▶ Lasso coefficients:", coefficients)

# Save the trained pipeline
model_dir = os.path.dirname("models/pickle_lasso_Maize.pkl")
os.makedirs(model_dir, exist_ok=True)
with open("models/pickle_lasso_Maize.pkl", 'wb') as f:
    pickle.dump(pipeline, f)
print("\nSaved model to models/pickle_lasso_Maize.pkl")
