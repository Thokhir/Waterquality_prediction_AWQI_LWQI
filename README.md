# 💧 Dual Water Quality Prediction System (AWQI + LWQI)

A Streamlit-based machine learning application for predicting and assessing water quality for **Aquaculture** and **Livestock** use cases using multiple ML models — Version 4.2.

---

## 🌊 Overview

This system provides a unified interface to evaluate water quality through two specialized indices:

- **AWQI** — Aquaculture Water Quality Index (for fish farming / pond water)
- **LWQI** — Livestock Water Quality Index (for animal drinking water)

Users input water parameter readings and receive instant quality scores, classifications, severity assessments, and actionable treatment recommendations — powered by an ensemble of 24 trained machine learning models (12 per system: 6 regression + 6 classification).

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📊 Prediction Dashboard | Enter parameters → get AWQI/LWQI score, class, and confidence |
| 🤖 Multi-Model Ensemble | Predictions from Linear Regression, SVR, Random Forest, Decision Tree, XGBoost, ANN |
| 💡 Smart Recommendations | Parameter-level actionable advice with severity tiers (Excellent → Critical) |
| 📚 Parameter Guide | Reference ranges and descriptions for all water quality parameters |
| 📈 Model Performance | R² and MSE scores for all 6 regression models per system |
| ⏱️ Time-Aware Input | Cyclical time encoding (sin/cos) for diurnal variation |

---

## 🗂️ Repository Structure

```
Waterquality_prediction_AWQI_LWQI/
│
├── app_combined_v4.2.py        # Main Streamlit application
├── train_combined_models.py    # Model training script
│
├── Aquaculture.csv             # Training data for AWQI
├── Live_stock.csv              # Training data for LWQI
│
├── models/
│   ├── aquaculture/            # Trained models for AWQI
│   │   ├── *_reg.pkl           # Regression models
│   │   ├── *_clf.pkl           # Classification models
│   │   ├── scaler_regression.pkl
│   │   ├── scaler_classification.pkl
│   │   ├── label_encoder.pkl
│   │   ├── feature_names.pkl
│   │   └── class_names.pkl
│   └── livestock/              # Trained models for LWQI
│       └── (same structure)
│
├── .stremlit/                  # Streamlit configuration
├── requirements.txt
└── .gitignore
```

---

## 🧪 Water Quality Parameters

### 🐟 Aquaculture (AWQI)

| Parameter | Unit | Optimal Range | Role |
|---|---|---|---|
| TDS | mg/L | < 250 | Total dissolved solids / mineral content |
| DO | mg/L | > 7 | Dissolved oxygen — critical for aquatic life |
| Nitrate | mg/L | < 10 | Nutrient pollution indicator |
| Total Hardness (TH) | mg/L | 50–150 | Calcium & magnesium concentration |
| pH | — | 6.5–8.5 | Acidity / alkalinity |
| Chlorides | mg/L | < 250 | Salt concentration |
| Alkalinity | mg/L | 50–200 | Buffering capacity |
| EC | µS/cm | 500–1500 | Electrical conductivity / dissolved ions |
| Ammonia | mg/L | < 0.5 | Organic pollution — **dominant factor** |

### 🐄 Livestock (LWQI)

| Parameter | Unit | Optimal Range | Role |
|---|---|---|---|
| DO | mg/L | > 5 | Dissolved oxygen |
| Nitrate | mg/L | < 50 | Nutrient pollution |
| Calcium Hardness (CaH) | mg/L | < 300 | Calcium ion level |
| pH | — | 6.5–8.5 | Acidity / alkalinity |
| Sulphates | mg/L | < 500 | Sulphate content |
| Sodium | mg/L | < 200 | Sodium level |
| EC | µS/cm | < 1500 | Electrical conductivity — **dominant factor** |
| Iron | mg/L | < 2 | Iron content — **dominant factor** |

---

## 📈 Model Performance

### Aquaculture (AWQI) — Regression R² Scores

| Model | R² Score | MSE |
|---|---|---|
| Linear Regression | 0.9999 | 0.0004 |
| SVR | 0.9999 | 0.0058 |
| Random Forest | 0.9482 | 6.0648 |
| XGBoost | 0.8940 | 12.4190 |
| ANN | 0.9062 | 10.9935 |
| Decision Tree | 0.8717 | 15.0384 |

### Livestock (LWQI) — Regression R² Scores

| Model | R² Score | MSE |
|---|---|---|
| Linear Regression | 0.9999 | 0.00001 |
| SVR | 0.9990 | 0.004 |
| XGBoost | 0.9330 | 40.68 |
| Random Forest | 0.9150 | 51.49 |
| Decision Tree | 0.8940 | 64.69 |
| ANN | -0.6020 | 980.72 |

> **Primary model used for scoring:** Linear Regression (or SVR as fallback) due to highest R² score.

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Thokhir/Waterquality_prediction_AWQI_LWQI.git
cd Waterquality_prediction_AWQI_LWQI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models (if not already present)

```bash
python train_combined_models.py
```

This will generate all `.pkl` model files inside the `models/aquaculture/` and `models/livestock/` directories.

### 4. Run the Streamlit app

```bash
streamlit run app_combined_v4.2.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scipy>=1.10.0
```

---

## 🔍 How to Use

1. Launch the app and select either **🐟 Aquaculture (AWQI)** or **🐄 Livestock (LWQI)**.
2. Navigate to the **Prediction Dashboard** from the sidebar.
3. Enter water parameter values and optionally set the **time of day** (used for diurnal encoding).
4. Click **🔍 Predict Water Quality**.
5. Review:
   - **Quality Score** and **Class** (Excellent / Good / Moderate / Poor)
   - **Regression predictions** from all models
   - **Classification result** with confidence
   - **Detailed recommendations** with severity-coded action steps
6. Use the **Parameter Guide** for reference ranges and the **Model Performance** page to compare ML models.

---

## ⚠️ Important Notes

- The models are trained on domain-specific datasets. **Dominant parameters** (Ammonia & DO for Aquaculture; Iron, DO & EC for Livestock) have the highest statistical influence on predicted scores.
- Use results as a **decision-support tool**, not a substitute for laboratory analysis.
- A high score on one or two minor parameters may not significantly affect the overall index if dominant parameters are within range — this is by design, reflecting the training data patterns.

---

## 🧠 Technology Stack

- **Frontend:** Streamlit
- **ML Models:** Scikit-learn (Linear Regression, SVR, Random Forest, Decision Tree), XGBoost, ANN (MLPRegressor)
- **Data:** Custom aquaculture and livestock water quality datasets
- **Serialization:** Joblib (`.pkl` files)
- **Encoding:** Cyclical time features (sin/cos), Label Encoding for classification

---

## 📄 License

This project is open source. Contributions and improvements are welcome via pull requests.

---

*Version 4.2 — Merged Unified Quality Assessment Tool | Status: ✅ Production Ready*
