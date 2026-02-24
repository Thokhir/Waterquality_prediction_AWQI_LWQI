"""
Combined Water Quality Training Script
Trains models for BOTH Aquaculture (AWQI) and Livestock (LWQI)
Generates models for unified Streamlit deployment
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMBINED WATER QUALITY TRAINING - AQUACULTURE + LIVESTOCK")
print("=" * 80)

# Create models directory
os.makedirs("models/aquaculture", exist_ok=True)
os.makedirs("models/livestock", exist_ok=True)

# ============================================================================
# AQUACULTURE (AWQI) TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING AQUACULTURE (AWQI) MODELS")
print("="*80)

print("\n[1/6] Loading Aquaculture dataset...")
df_aqua = pd.read_csv('Aquaculture.csv')
print(f"✓ Loaded: {df_aqua.shape[0]} samples, {df_aqua.shape[1]} features")

print("\n[2/6] Engineering Aquaculture features...")
df_aqua['Time_sin'] = np.sin(2 * np.pi * df_aqua['Time'] / 12)
df_aqua['Time_cos'] = np.cos(2 * np.pi * df_aqua['Time'] / 12)

X_aqua = df_aqua.drop(['AWQI', 'Code', 'Time', 'Seasons'], axis=1)
y_aqua_reg = df_aqua['AWQI']

X_train_aqua, X_test_aqua, y_train_aqua, y_test_aqua = train_test_split(
    X_aqua, y_aqua_reg, test_size=0.3, random_state=42
)

scaler_aqua_reg = StandardScaler()
X_train_aqua_scaled = scaler_aqua_reg.fit_transform(X_train_aqua)
X_test_aqua_scaled = scaler_aqua_reg.transform(X_test_aqua)

print(f"✓ Training set: {X_train_aqua_scaled.shape}")
print(f"✓ Test set: {X_test_aqua_scaled.shape}")

print("\n[3/6] Training Aquaculture Regression Models...")
aqua_reg_models = {
    'Linear Regression': {'model': LinearRegression(), 'param_grid': {}},
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'param_grid': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    },
    'SVR': {
        'model': SVR(),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    },
    'ANN': {
        'model': MLPRegressor(random_state=42, max_iter=1000),
        'param_grid': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

aqua_best_reg_models = {}
for name, info in aqua_reg_models.items():
    model = info['model']
    param_grid = info['param_grid']
    
    if param_grid:
        gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        gs.fit(X_train_aqua_scaled, y_train_aqua)
        best_model = gs.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train_aqua_scaled, y_train_aqua)
    
    aqua_best_reg_models[name] = best_model
    y_pred = best_model.predict(X_test_aqua_scaled)
    mse = mean_squared_error(y_test_aqua, y_pred)
    r2 = r2_score(y_test_aqua, y_pred)
    print(f"  ✓ {name:20s} - R²: {r2:.4f}, MSE: {mse:.4f}")

print("\n[4/6] Training Aquaculture Classification Models...")
bins_aqua = [0, 25, 50, float('inf')]
labels_aqua = ['Excellent', 'Good', 'Moderate']
df_aqua['AWQI_class'] = pd.cut(df_aqua['AWQI'], bins=bins_aqua, labels=labels_aqua, right=False)

X_aqua_clf = X_aqua
y_aqua_clf = df_aqua['AWQI_class']
le_aqua = LabelEncoder()
y_aqua_clf_encoded = le_aqua.fit_transform(y_aqua_clf)

X_train_aqua_clf, X_test_aqua_clf, y_train_aqua_clf, y_test_aqua_clf = train_test_split(
    X_aqua_clf, y_aqua_clf_encoded, test_size=0.3, random_state=42
)

scaler_aqua_clf = StandardScaler()
X_train_aqua_clf_scaled = scaler_aqua_clf.fit_transform(X_train_aqua_clf)
X_test_aqua_clf_scaled = scaler_aqua_clf.transform(X_test_aqua_clf)

aqua_clf_models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'param_grid': {'C': [0.1, 1, 10]}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {'max_depth': [None, 10, 20]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    },
    'SVC': {
        'model': SVC(random_state=42, class_weight='balanced', probability=True),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    },
    'ANN': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'param_grid': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

aqua_best_clf_models = {}
for name, info in aqua_clf_models.items():
    model = info['model']
    param_grid = info['param_grid']
    
    gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train_aqua_clf_scaled, y_train_aqua_clf)
    best_model = gs.best_estimator_
    
    aqua_best_clf_models[name] = best_model
    y_pred = best_model.predict(X_test_aqua_clf_scaled)
    acc = accuracy_score(y_test_aqua_clf, y_pred)
    print(f"  ✓ {name:20s} - Accuracy: {acc:.4f}")

# ============================================================================
# LIVESTOCK (LWQI) TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING LIVESTOCK (LWQI) MODELS")
print("="*80)

print("\n[1/6] Loading Livestock dataset...")
df_live = pd.read_csv('Live_stock.csv')
print(f"✓ Loaded: {df_live.shape[0]} samples, {df_live.shape[1]} features")

print("\n[2/6] Engineering Livestock features...")
df_live['Time_sin'] = np.sin(2 * np.pi * df_live['Time'] / 12)
df_live['Time_cos'] = np.cos(2 * np.pi * df_live['Time'] / 12)

X_live = df_live.drop(['LWQI', 'Code', 'Time', 'Seasons'], axis=1)
y_live_reg = df_live['LWQI']

X_train_live, X_test_live, y_train_live, y_test_live = train_test_split(
    X_live, y_live_reg, test_size=0.3, random_state=42
)

scaler_live_reg = StandardScaler()
X_train_live_scaled = scaler_live_reg.fit_transform(X_train_live)
X_test_live_scaled = scaler_live_reg.transform(X_test_live)

print(f"✓ Training set: {X_train_live_scaled.shape}")
print(f"✓ Test set: {X_test_live_scaled.shape}")

print("\n[3/6] Training Livestock Regression Models...")
live_reg_models = {
    'Linear Regression': {'model': LinearRegression(), 'param_grid': {}},
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'param_grid': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    },
    'SVR': {
        'model': SVR(),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    },
    'ANN': {
        'model': MLPRegressor(random_state=42, max_iter=1000),
        'param_grid': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

live_best_reg_models = {}
for name, info in live_reg_models.items():
    model = info['model']
    param_grid = info['param_grid']
    
    if param_grid:
        gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        gs.fit(X_train_live_scaled, y_train_live)
        best_model = gs.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train_live_scaled, y_train_live)
    
    live_best_reg_models[name] = best_model
    y_pred = best_model.predict(X_test_live_scaled)
    mse = mean_squared_error(y_test_live, y_pred)
    r2 = r2_score(y_test_live, y_pred)
    print(f"  ✓ {name:20s} - R²: {r2:.4f}, MSE: {mse:.4f}")

print("\n[4/6] Training Livestock Classification Models...")
bins_live = [0, 40, 80, float('inf')]
labels_live = ['Good', 'Fair', 'Poor']
df_live['LWQI_class'] = pd.cut(df_live['LWQI'], bins=bins_live, labels=labels_live, right=False)

X_live_clf = X_live
y_live_clf = df_live['LWQI_class']
le_live = LabelEncoder()
y_live_clf_encoded = le_live.fit_transform(y_live_clf)

X_train_live_clf, X_test_live_clf, y_train_live_clf, y_test_live_clf = train_test_split(
    X_live_clf, y_live_clf_encoded, test_size=0.3, random_state=42
)

scaler_live_clf = StandardScaler()
X_train_live_clf_scaled = scaler_live_clf.fit_transform(X_train_live_clf)
X_test_live_clf_scaled = scaler_live_clf.transform(X_test_live_clf)

live_clf_models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'param_grid': {'C': [0.1, 1, 10]}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {'max_depth': [None, 10, 20]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    },
    'SVC': {
        'model': SVC(random_state=42, class_weight='balanced', probability=True),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    },
    'ANN': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'param_grid': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

live_best_clf_models = {}
for name, info in live_clf_models.items():
    model = info['model']
    param_grid = info['param_grid']
    
    gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train_live_clf_scaled, y_train_live_clf)
    best_model = gs.best_estimator_
    
    live_best_clf_models[name] = best_model
    y_pred = best_model.predict(X_test_live_clf_scaled)
    acc = accuracy_score(y_test_live_clf, y_pred)
    print(f"  ✓ {name:20s} - Accuracy: {acc:.4f}")

# ============================================================================
# SAVE ALL MODELS
# ============================================================================
print("\n" + "="*80)
print("SAVING ALL MODELS")
print("="*80)

print("\n[5/6] Saving Aquaculture models...")
for name, model in aqua_best_reg_models.items():
    filename = f"models/aquaculture/{name.replace(' ', '_').lower()}_reg.pkl"
    joblib.dump(model, filename)
    print(f"  ✓ {filename}")

for name, model in aqua_best_clf_models.items():
    filename = f"models/aquaculture/{name.replace(' ', '_').lower()}_clf.pkl"
    joblib.dump(model, filename)
    print(f"  ✓ {filename}")

joblib.dump(scaler_aqua_reg, "models/aquaculture/scaler_regression.pkl")
joblib.dump(scaler_aqua_clf, "models/aquaculture/scaler_classification.pkl")
joblib.dump(le_aqua, "models/aquaculture/label_encoder.pkl")
joblib.dump(list(X_aqua.columns), "models/aquaculture/feature_names.pkl")
joblib.dump(le_aqua.classes_, "models/aquaculture/class_names.pkl")
print(f"  ✓ Scalers and encoders saved")

print("\n[6/6] Saving Livestock models...")
for name, model in live_best_reg_models.items():
    filename = f"models/livestock/{name.replace(' ', '_').lower()}_reg.pkl"
    joblib.dump(model, filename)
    print(f"  ✓ {filename}")

for name, model in live_best_clf_models.items():
    filename = f"models/livestock/{name.replace(' ', '_').lower()}_clf.pkl"
    joblib.dump(model, filename)
    print(f"  ✓ {filename}")

joblib.dump(scaler_live_reg, "models/livestock/scaler_regression.pkl")
joblib.dump(scaler_live_clf, "models/livestock/scaler_classification.pkl")
joblib.dump(le_live, "models/livestock/label_encoder.pkl")
joblib.dump(list(X_live.columns), "models/livestock/feature_names.pkl")
joblib.dump(le_live.classes_, "models/livestock/class_names.pkl")
print(f"  ✓ Scalers and encoders saved")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nModels saved in:")
print("  • models/aquaculture/")
print("  • models/livestock/")
print("\nReady for Streamlit deployment!")
