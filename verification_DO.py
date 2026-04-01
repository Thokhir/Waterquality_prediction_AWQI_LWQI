import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load Livestock model
model = joblib.load('models/livestock/linear_regression_reg.pkl')
scaler = joblib.load('models/livestock/scaler_regression.pkl')
features = joblib.load('models/livestock/feature_names.pkl')

# Test: Change ONLY DO, keep everything else constant
baseline = {
    'DO': 3.58, 'Nitrate': 0.99, 'CaH': 11.55, 'pH': 7.76,
    'Sulphates': 10.87, 'Sodium': 29.75, 'EC': 328.12,
    'Iron': 2.27, 'Time_sin': 0, 'Time_cos': 1
}

print("Testing DO effect (other params constant):")
for do_val in [2.0, 4.0, 6.0, 8.0]:
    test = baseline.copy()
    test['DO'] = do_val
    df = pd.DataFrame([test])
    pred = model.predict(scaler.transform(df[features]))[0]
    print(f"DO={do_val:.1f} → LWQI={pred:.2f}")