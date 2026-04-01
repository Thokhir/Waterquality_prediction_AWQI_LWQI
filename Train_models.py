"""
Combined Water Quality Training Script
Trains models for BOTH Aquaculture (AWQI) and Livestock (LWQI)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("COMBINED WATER QUALITY TRAINING - AQUACULTURE + LIVESTOCK")
print("=" * 80)

os.makedirs("models/aquaculture", exist_ok=True)
os.makedirs("models/livestock", exist_ok=True)

# ========================== AQUACULTURE ==========================
print("\n" + "="*80)
print("TRAINING AQUACULTURE (AWQI) MODELS")
print("="*80)

df_aqua = pd.read_csv('Aquaculture.csv')
print(f"✓ Loaded Aquaculture dataset: {df_aqua.shape[0]} samples")

# Feature Engineering
df_aqua['Time_sin'] = np.sin(2 * np.pi * df_aqua['Time'] / 12)
df_aqua['Time_cos'] = np.cos(2 * np.pi * df_aqua['Time'] / 12)

X_aqua = df_aqua.drop(['AWQI', 'Code', 'Time', 'Seasons'], axis=1)
y_aqua = df_aqua['AWQI']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_aqua, y_aqua, test_size=0.3, random_state=42)

scaler_aqua = StandardScaler()
X_train_a_scaled = scaler_aqua.fit_transform(X_train_a)
X_test_a_scaled = scaler_aqua.transform(X_test_a)

# Train best regression models (Simplified - focusing on top performers)
reg_models_aqua = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1.0, kernel='rbf'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

print("\nTraining Aquaculture Regression Models...")
aqua_best_models = {}
for name, model in reg_models_aqua.items():
    model.fit(X_train_a_scaled, y_train_a)
    y_pred = model.predict(X_test_a_scaled)
    r2 = r2_score(y_test_a, y_pred)
    mse = mean_squared_error(y_test_a, y_pred)
    aqua_best_models[name] = model
    print(f"  ✓ {name:20s} - R²: {r2:.4f} | MSE: {mse:.4f}")

# Save Aquaculture models
for name, model in aqua_best_models.items():
    joblib.dump(model, f"models/aquaculture/{name.replace(' ', '_').lower()}_reg.pkl")

joblib.dump(scaler_aqua, "models/aquaculture/scaler_regression.pkl")
joblib.dump(list(X_aqua.columns), "models/aquaculture/feature_names.pkl")
print("✅ Aquaculture models saved")

# ========================== LIVESTOCK ==========================
print("\n" + "="*80)
print("TRAINING LIVESTOCK (LWQI) MODELS")
print("="*80)

df_live = pd.read_csv('Live_stock.csv')
print(f"✓ Loaded Livestock dataset: {df_live.shape[0]} samples")

df_live['Time_sin'] = np.sin(2 * np.pi * df_live['Time'] / 12)
df_live['Time_cos'] = np.cos(2 * np.pi * df_live['Time'] / 12)

X_live = df_live.drop(['LWQI', 'Code', 'Time', 'Seasons'], axis=1)
y_live = df_live['LWQI']

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_live, y_live, test_size=0.3, random_state=42)

scaler_live = StandardScaler()
X_train_l_scaled = scaler_live.fit_transform(X_train_l)
X_test_l_scaled = scaler_live.transform(X_test_l)

reg_models_live = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1.0, kernel='rbf'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

print("\nTraining Livestock Regression Models...")
live_best_models = {}
for name, model in reg_models_live.items():
    model.fit(X_train_l_scaled, y_train_l)
    y_pred = model.predict(X_test_l_scaled)
    r2 = r2_score(y_test_l, y_pred)
    mse = mean_squared_error(y_test_l, y_pred)
    live_best_models[name] = model
    print(f"  ✓ {name:20s} - R²: {r2:.4f} | MSE: {mse:.4f}")

# Save Livestock models
for name, model in live_best_models.items():
    joblib.dump(model, f"models/livestock/{name.replace(' ', '_').lower()}_reg.pkl")

joblib.dump(scaler_live, "models/livestock/scaler_regression.pkl")
joblib.dump(list(X_live.columns), "models/livestock/feature_names.pkl")
print("✅ Livestock models saved")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE! Models saved in 'models/' folder")
print("="*80)