import joblib

# Load and display AWQI (Aquaculture) metrics
print('=' * 80)
print('AQUACULTURE (AWQI) METRICS')
print('=' * 80)

reg_metrics = joblib.load('models/aquaculture/regression_metrics.pkl')
print('\nRegression Models (R² and MSE):')
for model_name, metrics in reg_metrics.items():
    print(f'  {model_name:20s} - R²: {metrics["R2"]:.4f}, MSE: {metrics["MSE"]:.4f}')

clf_metrics = joblib.load('models/aquaculture/classification_metrics.pkl')
print('\nClassification Models (Accuracy):')
for model_name, metrics in clf_metrics.items():
    print(f'  {model_name:20s} - Accuracy: {metrics["Accuracy"]:.4f}')

# Load and display LWQI (Livestock) metrics
print('\n' + '=' * 80)
print('LIVESTOCK (LWQI) METRICS')
print('=' * 80)

reg_metrics = joblib.load('models/livestock/regression_metrics.pkl')
print('\nRegression Models (R² and MSE):')
for model_name, metrics in reg_metrics.items():
    print(f'  {model_name:20s} - R²: {metrics["R2"]:.4f}, MSE: {metrics["MSE"]:.4f}')

clf_metrics = joblib.load('models/livestock/classification_metrics.pkl')
print('\nClassification Models (Accuracy):')
for model_name, metrics in clf_metrics.items():
    print(f'  {model_name:20s} - Accuracy: {metrics["Accuracy"]:.4f}')
