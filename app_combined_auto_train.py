"""
Combined Water Quality Prediction System - Version 3.3 (AUTO-TRAIN)
Aquaculture (AWQI) + Livestock (LWQI) in one unified application
AUTO-TRAIN: Automatically trains models on first Streamlit Cloud run if missing
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Water Quality Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 2rem 1rem; }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .critical-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .note-box {
        background-color: #f5f5f5;
        border-left: 5px solid #6c757d;
        padding: 12px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 12px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AUTO-TRAIN MODELS IF MISSING
# ============================================================================

def train_models_auto():
    """Auto-train models if they don't exist"""
    st.info("🔄 First run detected - Training models automatically... (This happens once)")
    
    try:
        os.makedirs("models/aquaculture", exist_ok=True)
        os.makedirs("models/livestock", exist_ok=True)
        
        # Load datasets
        st.write("📊 Loading datasets...")
        df_aqua = pd.read_csv('Aquaculture.csv')
        df_live = pd.read_csv('Live_stock.csv')
        
        # AQUACULTURE TRAINING
        st.write("🐟 Training Aquaculture models...")
        df_aqua['Time_sin'] = np.sin(2 * np.pi * df_aqua['Time'] / 12)
        df_aqua['Time_cos'] = np.cos(2 * np.pi * df_aqua['Time'] / 12)
        X_aqua = df_aqua.drop(['AWQI', 'Code', 'Time', 'Seasons'], axis=1)
        y_aqua = df_aqua['AWQI']
        
        X_train_aqua, X_test_aqua, y_train_aqua, y_test_aqua = train_test_split(
            X_aqua, y_aqua, test_size=0.3, random_state=42
        )
        
        scaler_aqua_reg = StandardScaler()
        X_train_aqua_scaled = scaler_aqua_reg.fit_transform(X_train_aqua)
        X_test_aqua_scaled = scaler_aqua_reg.transform(X_test_aqua)
        
        # Train regression models
        aqua_models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'svr': SVR(kernel='rbf'),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'ann': MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
        }
        
        for name, model in aqua_models.items():
            model.fit(X_train_aqua_scaled, y_train_aqua)
            joblib.dump(model, f"models/aquaculture/{name}_reg.pkl")
        
        # Classification
        bins_aqua = [0, 25, 50, float('inf')]
        labels_aqua = ['Excellent', 'Good', 'Moderate']
        df_aqua['class'] = pd.cut(df_aqua['AWQI'], bins=bins_aqua, labels=labels_aqua, right=False)
        y_aqua_clf = df_aqua['class']
        le_aqua = LabelEncoder()
        y_aqua_clf_encoded = le_aqua.fit_transform(y_aqua_clf)
        
        X_train_aqua_clf, X_test_aqua_clf, y_train_aqua_clf, y_test_aqua_clf = train_test_split(
            X_aqua, y_aqua_clf_encoded, test_size=0.3, random_state=42
        )
        
        scaler_aqua_clf = StandardScaler()
        X_train_aqua_clf_scaled = scaler_aqua_clf.fit_transform(X_train_aqua_clf)
        
        aqua_clf_models = {
            'linear_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svc': SVC(kernel='rbf', probability=True, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'ann': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
        }
        
        for name, model in aqua_clf_models.items():
            model.fit(X_train_aqua_clf_scaled, y_train_aqua_clf)
            joblib.dump(model, f"models/aquaculture/{name}_clf.pkl")
        
        joblib.dump(scaler_aqua_reg, "models/aquaculture/scaler_regression.pkl")
        joblib.dump(scaler_aqua_clf, "models/aquaculture/scaler_classification.pkl")
        joblib.dump(le_aqua, "models/aquaculture/label_encoder.pkl")
        joblib.dump(list(X_aqua.columns), "models/aquaculture/feature_names.pkl")
        joblib.dump(le_aqua.classes_, "models/aquaculture/class_names.pkl")
        
        # LIVESTOCK TRAINING
        st.write("🐄 Training Livestock models...")
        df_live['Time_sin'] = np.sin(2 * np.pi * df_live['Time'] / 12)
        df_live['Time_cos'] = np.cos(2 * np.pi * df_live['Time'] / 12)
        X_live = df_live.drop(['LWQI', 'Code', 'Time', 'Seasons'], axis=1)
        y_live = df_live['LWQI']
        
        X_train_live, X_test_live, y_train_live, y_test_live = train_test_split(
            X_live, y_live, test_size=0.3, random_state=42
        )
        
        scaler_live_reg = StandardScaler()
        X_train_live_scaled = scaler_live_reg.fit_transform(X_train_live)
        X_test_live_scaled = scaler_live_reg.transform(X_test_live)
        
        # Train regression models
        live_models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'svr': SVR(kernel='rbf'),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'ann': MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
        }
        
        for name, model in live_models.items():
            model.fit(X_train_live_scaled, y_train_live)
            joblib.dump(model, f"models/livestock/{name}_reg.pkl")
        
        # Classification
        bins_live = [0, 40, 80, float('inf')]
        labels_live = ['Good', 'Fair', 'Poor']
        df_live['class'] = pd.cut(df_live['LWQI'], bins=bins_live, labels=labels_live, right=False)
        y_live_clf = df_live['class']
        le_live = LabelEncoder()
        y_live_clf_encoded = le_live.fit_transform(y_live_clf)
        
        X_train_live_clf, X_test_live_clf, y_train_live_clf, y_test_live_clf = train_test_split(
            X_live, y_live_clf_encoded, test_size=0.3, random_state=42
        )
        
        scaler_live_clf = StandardScaler()
        X_train_live_clf_scaled = scaler_live_clf.fit_transform(X_train_live_clf)
        
        live_clf_models = {
            'linear_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svc': SVC(kernel='rbf', probability=True, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'ann': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
        }
        
        for name, model in live_clf_models.items():
            model.fit(X_train_live_clf_scaled, y_train_live_clf)
            joblib.dump(model, f"models/livestock/{name}_clf.pkl")
        
        joblib.dump(scaler_live_reg, "models/livestock/scaler_regression.pkl")
        joblib.dump(scaler_live_clf, "models/livestock/scaler_classification.pkl")
        joblib.dump(le_live, "models/livestock/label_encoder.pkl")
        joblib.dump(list(X_live.columns), "models/livestock/feature_names.pkl")
        joblib.dump(le_live.classes_, "models/livestock/class_names.pkl")
        
        st.success("✅ Models trained successfully!")
        return True
    
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return False

# ============================================================================
# CHECK MODELS
# ============================================================================

def models_exist():
    """Check if models are trained"""
    aqua_path = "models/aquaculture"
    live_path = "models/livestock"
    return os.path.exists(aqua_path) and os.path.exists(live_path) and \
           len(os.listdir(aqua_path)) > 0 and len(os.listdir(live_path)) > 0

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================

@st.cache_resource
def load_system(system_type):
    """Load all models for selected system"""
    try:
        reg_models = {}
        clf_models = {}
        model_dir = f"models/{system_type}"
        
        # Regression models
        for fname in os.listdir(model_dir):
            if '_reg.pkl' in fname and fname != 'scaler_regression.pkl':
                model_name = fname.replace('_reg.pkl', '').replace('_', ' ').title()
                reg_models[model_name] = joblib.load(f"{model_dir}/{fname}")
        
        # Classification models
        for fname in os.listdir(model_dir):
            if '_clf.pkl' in fname and fname != 'scaler_classification.pkl':
                model_name = fname.replace('_clf.pkl', '').replace('_', ' ').title()
                clf_models[model_name] = joblib.load(f"{model_dir}/{fname}")
        
        # Scalers and encoders
        scalers = {
            'regression': joblib.load(f"{model_dir}/scaler_regression.pkl"),
            'classification': joblib.load(f"{model_dir}/scaler_classification.pkl")
        }
        
        encoders = {
            'label_encoder': joblib.load(f"{model_dir}/label_encoder.pkl"),
            'feature_names': joblib.load(f"{model_dir}/feature_names.pkl"),
            'class_names': joblib.load(f"{model_dir}/class_names.pkl")
        }
        
        return reg_models, clf_models, scalers, encoders
    except Exception as e:
        return {}, {}, {}, {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_quality_interpretation(value, system_type):
    """Interpret quality score based on system type"""
    if system_type == "Aquaculture":
        if value < 25:
            return {
                'class': 'Excellent',
                'description': '✅ Excellent water quality',
                'color': '#28a745',
                'emoji': '✅'
            }
        elif value < 50:
            return {
                'class': 'Good',
                'description': '👍 Good water quality',
                'color': '#17a2b8',
                'emoji': '👍'
            }
        elif value < 75:
            return {
                'class': 'Moderate',
                'description': '⚠️ Moderate water quality',
                'color': '#ffc107',
                'emoji': '⚠️'
            }
        else:
            return {
                'class': 'Poor',
                'description': '🚫 Poor water quality',
                'color': '#dc3545',
                'emoji': '🚫'
            }
    else:  # Livestock
        if value < 40:
            return {'class': 'Good', 'description': '✅ Good', 'color': '#28a745', 'emoji': '✅'}
        elif value < 80:
            return {'class': 'Fair', 'description': '⚠️ Fair', 'color': '#ffc107', 'emoji': '⚠️'}
        else:
            return {'class': 'Poor', 'description': '🚫 Poor', 'color': '#dc3545', 'emoji': '🚫'}

def prepare_features(input_dict, feature_names):
    """Prepare features for prediction"""
    features = pd.DataFrame([input_dict])
    return features[feature_names]

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>💧 Combined Water Quality Prediction System 💧</h1>
        <p><i>Aquaculture (AWQI) + Livestock (LWQI) Analysis</i></p>
        <p style="color: #666; font-size: 14px;">Version 3.3 - Auto-Training</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models exist, if not train them
    if not models_exist():
        st.warning("⏳ Models not found. Training on first run...")
        if not train_models_auto():
            st.stop()
        st.rerun()
    
    # System Selection
    st.markdown("""
    <div class="info-box">
    <b>Select Water Quality System:</b> Choose Aquaculture or Livestock
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🐟 Aquaculture (AWQI)", use_container_width=True):
            st.session_state.system = "Aquaculture"
    
    with col2:
        if st.button("🐄 Livestock (LWQI)", use_container_width=True):
            st.session_state.system = "Livestock"
    
    if "system" not in st.session_state:
        st.session_state.system = "Aquaculture"
    
    # Load models
    reg_models, clf_models, scalers, encoders = load_system(st.session_state.system)
    
    if not reg_models or not scalers:
        st.error(f"❌ Failed to load {st.session_state.system} models.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select page:", ["📊 Prediction Dashboard", "📚 Parameter Guide", "ℹ️ About"])
    
    # ========================================================================
    # PAGE: PREDICTION DASHBOARD
    # ========================================================================
    if page == "📊 Prediction Dashboard":
        st.header(f"{st.session_state.system} - Water Quality Prediction")
        
        feature_names = encoders['feature_names']
        
        if st.session_state.system == "Aquaculture":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tds = st.number_input("TDS (mg/L)", 0.0, 5000.0, 170.0)
                ph = st.number_input("pH", 0.0, 14.0, 7.8)
                alkalinity = st.number_input("Alkalinity (mg/L)", 0.0, 2000.0, 70.0)
            
            with col2:
                do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0)
                chlorides = st.number_input("Chlorides (mg/L)", 0.0, 3000.0, 25.0)
                ec = st.number_input("EC (µS/cm)", 0.0, 10000.0, 280.0)
            
            with col3:
                nitrate = st.number_input("Nitrate (mg/L)", 0.0, 2000.0, 0.4)
                th = st.number_input("Total Hardness (mg/L)", 0.0, 2000.0, 140.0)
                ammonia = st.number_input("Ammonia (mg/L)", 0.0, 100.0, 0.01)
            
            time_value = st.slider("Time (hours)", 0, 23, 12)
            time_sin = np.sin(2 * np.pi * time_value / 12)
            time_cos = np.cos(2 * np.pi * time_value / 12)
            
            input_dict = {
                'TDS': tds, 'DO': do, 'Nitrate': nitrate, 'TH': th, 'pH': ph,
                'Chlorides': chlorides, 'Alkalinity': alkalinity, 'EC': ec,
                'Ammonia': ammonia, 'Time_sin': time_sin, 'Time_cos': time_cos
            }
        
        else:  # Livestock
            col1, col2, col3 = st.columns(3)
            
            with col1:
                do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0)
                ph = st.number_input("pH", 0.0, 14.0, 7.8)
                na = st.number_input("Sodium (mg/L)", 0.0, 500.0, 20.0)
            
            with col2:
                nitrate = st.number_input("Nitrate (mg/L)", 0.0, 500.0, 0.5)
                cah = st.number_input("Calcium Hardness (mg/L)", 0.0, 500.0, 8.0)
                sulphates = st.number_input("Sulphates (mg/L)", 0.0, 500.0, 6.0)
            
            with col3:
                ec = st.number_input("EC (µS/cm)", 0.0, 2000.0, 300.0)
                iron = st.number_input("Iron (mg/L)", 0.0, 100.0, 2.0)
            
            time_value = st.slider("Time (hours)", 0, 23, 12)
            time_sin = np.sin(2 * np.pi * time_value / 12)
            time_cos = np.cos(2 * np.pi * time_value / 12)
            
            input_dict = {
                'DO': do, 'Nitrate': nitrate, 'CaH': cah, 'pH': ph,
                'Sulphates': sulphates, 'Sodium': na, 'EC': ec, 'Iron': iron,
                'Time_sin': time_sin, 'Time_cos': time_cos
            }
        
        if st.button("🔍 Predict Water Quality", use_container_width=True, type="primary"):
            try:
                features = prepare_features(input_dict, feature_names)
                scaled_features = scalers['regression'].transform(features)
                
                predictions = {}
                for name, model in reg_models.items():
                    try:
                        pred = model.predict(scaled_features)[0]
                        predictions[name] = pred
                    except:
                        pass
                
                if predictions:
                    best_name = list(predictions.keys())[0]
                    quality_score = predictions[best_name]
                    interpretation = get_quality_interpretation(quality_score, st.session_state.system)
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h2>{quality_score:.2f}</h2>
                            <p>{interpretation['class'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div style="background-color: {interpretation['color']}; 
                                    color: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3>{interpretation['emoji']} {interpretation['class']}</h3>
                            <p>{interpretation['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="note-box">
                    <b>ℹ️ Important Note:</b> Models trained on historical data where 2-3 dominant parameters 
                    strongly influence water quality. Always review individual parameters alongside the overall score.
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # ========================================================================
    # PAGE: PARAMETER GUIDE
    # ========================================================================
    elif page == "📚 Parameter Guide":
        st.header("Water Quality Parameters")
        st.info("Reference guide for optimal parameter ranges")
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.header("About")
        st.write("Combined Water Quality Prediction System v3.3")
        st.write("Automatically trains models on first run.")

if __name__ == "__main__":
    main()
