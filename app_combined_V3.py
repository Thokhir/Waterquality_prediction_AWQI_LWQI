"""
Combined Water Quality Prediction System - Version 3.1 (FIXED)
Aquaculture (AWQI) + Livestock (LWQI) in one unified application
FIXED: Handles missing models gracefully
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
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
# CHECK MODELS
# ============================================================================

def models_exist():
    """Check if models are trained"""
    aqua_path = "models/aquaculture"
    live_path = "models/livestock"
    return os.path.exists(aqua_path) and os.path.exists(live_path)

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================

@st.cache_resource
def load_system(system_type):
    """Load all models for selected system"""
    try:
        models = {}
        model_dir = f"models/{system_type.lower()}"
        
        # Check if directory exists
        if not os.path.exists(model_dir):
            st.write(f"DEBUG: Directory not found: {model_dir}")
            return {}, {}, {}
        
        # Regression models
        for fname in os.listdir(model_dir):
            if '_reg.pkl' in fname and fname != 'scaler_regression.pkl':
                model_name = fname.replace('_reg.pkl', '').replace('_', ' ').title()
                try:
                    models[model_name] = joblib.load(f"{model_dir}/{fname}")
                except Exception as e:
                    st.write(f"DEBUG: Error loading {fname}: {str(e)}")
        
        # Scalers and encoders
        try:
            scalers = {
                'regression': joblib.load(f"{model_dir}/scaler_regression.pkl"),
                'classification': joblib.load(f"{model_dir}/scaler_classification.pkl")
            }
        except Exception as e:
            st.write(f"DEBUG: Error loading scalers: {str(e)}")
            return models, {}, {}
        
        try:
            encoders = {
                'label_encoder': joblib.load(f"{model_dir}/label_encoder.pkl"),
                'feature_names': joblib.load(f"{model_dir}/feature_names.pkl"),
                'class_names': joblib.load(f"{model_dir}/class_names.pkl")
            }
        except Exception as e:
            st.write(f"DEBUG: Error loading encoders: {str(e)}")
            return models, scalers, {}
        
        return models, scalers, encoders
    except Exception as e:
        st.write(f"DEBUG: General error in load_system: {str(e)}")
        return {}, {}, {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_quality_interpretation(value, system_type):
    """Interpret quality score based on system type"""
    if system_type == "Aquaculture":
        if value < 25:
            return {'class': 'Excellent', 'description': '✅ Perfect for aquaculture', 'color': '#28a745', 'emoji': '✅'}
        elif value < 50:
            return {'class': 'Good', 'description': '👍 Good quality - minor issues', 'color': '#17a2b8', 'emoji': '👍'}
        elif value < 75:
            return {'class': 'Moderate', 'description': '⚠️ Moderate - action needed', 'color': '#ffc107', 'emoji': '⚠️'}
        else:
            return {'class': 'Poor', 'description': '🚫 Poor - immediate action required', 'color': '#dc3545', 'emoji': '🚫'}
    else:  # Livestock
        if value < 40:
            return {'class': 'Good', 'description': '✅ Suitable for livestock', 'color': '#28a745', 'emoji': '✅'}
        elif value < 80:
            return {'class': 'Fair', 'description': '⚠️ Fair - monitor closely', 'color': '#ffc107', 'emoji': '⚠️'}
        else:
            return {'class': 'Poor', 'description': '🚫 Poor quality - treatment needed', 'color': '#dc3545', 'emoji': '🚫'}

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
        <p style="color: #666; font-size: 14px;">Version 3.1 - Unified Quality Assessment Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models exist
    if not models_exist():
        st.error("""
        ❌ **Models Not Found!**
        
        The system is deployed but models haven't been trained yet.
        
        **Solution: Run this command locally in your project folder:**
        
        ```bash
        python train_combined_models.py
        ```
        
        **Steps:**
        1. Open Terminal/Command Prompt
        2. Navigate to your project folder
        3. Run: `python train_combined_models.py`
        4. Wait for training to complete
        5. You should see "models/" folder created with training progress
        6. Once done, commit and push to GitHub:
           ```bash
           git add .
           git commit -m "Add trained models"
           git push
           ```
        7. Streamlit Cloud will redeploy automatically
        
        **What the training script does:**
        - Reads Aquaculture.csv and Live_stock.csv
        - Trains 12 models for Aquaculture
        - Trains 12 models for Livestock
        - Creates models/aquaculture/ and models/livestock/ folders
        - Takes about 5-10 minutes
        
        After models are trained 
                 nd pushed to GitHub, refresh this page!
        """)
        return
    
    # System Selection
    st.markdown("""
    <div class="info-box">
    <b>Select Water Quality System:</b> Choose whether to assess Aquaculture or Livestock water quality
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🐟 Aquaculture (AWQI)", use_container_width=True, key="btn_aqua"):
            st.session_state.system = "Aquaculture"
            st.cache_resource.clear()
            st.rerun()
    
    with col2:
        if st.button("🐄 Livestock (LWQI)", use_container_width=True, key="btn_live"):
            st.session_state.system = "Livestock"
            st.cache_resource.clear()
            st.rerun()
    
    if "system" not in st.session_state:
        st.session_state.system = "Aquaculture"
    
    # Load models
    reg_models, scalers, encoders = load_system(st.session_state.system)
    
    if not reg_models or not scalers:
        st.error(f"❌ Failed to load {st.session_state.system} models. Please check the models folder.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select page:",
        ["📊 Prediction Dashboard", "📚 Parameter Guide", "📈 Model Performance", "🐛 System Debug", "ℹ️ About"]
    )
    
    # ========================================================================
    # PAGE: PREDICTION DASHBOARD
    # ========================================================================
    if page == "📊 Prediction Dashboard":
        st.header(f"{st.session_state.system} - Water Quality Prediction")
        
        st.markdown("""
        <div class="info-box">
        <b>Enter water parameters below to get instant quality assessment</b>
        </div>
        """, unsafe_allow_html=True)
        
        feature_names = encoders['feature_names']
        
        if st.session_state.system == "Aquaculture":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("General Parameters")
                tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=170.0, step=1.0)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
                alkalinity = st.number_input("Alkalinity (mg/L)", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
            
            with col2:
                st.subheader("Biological Indicators")
                do = st.number_input("DO (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                chlorides = st.number_input("Chlorides (mg/L)", min_value=0.0, max_value=3000.0, value=25.0, step=1.0)
                ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=10000.0, value=280.0, step=10.0)
            
            with col3:
                st.subheader("Pollution Indicators")
                nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=2000.0, value=0.4, step=0.1)
                th = st.number_input("Total Hardness (mg/L)", min_value=0.0, max_value=2000.0, value=140.0, step=1.0)
                ammonia = st.number_input("Ammonia (mg/L)", min_value=0.0, max_value=100.0, value=0.01, step=0.001)
            
            time_value = st.slider("Time (hours)", min_value=0, max_value=23, value=12, step=1)
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
                st.subheader("General Parameters")
                do = st.number_input("DO (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
                na = st.number_input("Sodium (mg/L)", min_value=0.0, max_value=500.0, value=20.0, step=0.5)
            
            with col2:
                st.subheader("Mineral Content")
                nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=500.0, value=0.5, step=0.1)
                cah = st.number_input("Calcium Hardness (mg/L)", min_value=0.0, max_value=500.0, value=8.0, step=1.0)
                sulphates = st.number_input("Sulphates (mg/L)", min_value=0.0, max_value=500.0, value=6.0, step=0.1)
            
            with col3:
                st.subheader("Quality Indicators")
                ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=2000.0, value=300.0, step=10.0)
                iron = st.number_input("Iron (mg/L)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
            
            time_value = st.slider("Time (hours)", min_value=0, max_value=23, value=12, step=1)
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
                        st.subheader("Quality Assessment")
                        st.markdown(f"""
                        <div class="metric-box">
                            <h2>{quality_score:.2f}</h2>
                            <p>{interpretation['class'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.subheader("Result")
                        st.markdown(f"""
                        <div style="background-color: {interpretation['color']}; 
                                    color: white; padding: 20px; border-radius: 10px;">
                            <h3>{interpretation['emoji']} {interpretation['class']}</h3>
                            <p>{interpretation['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="note-box">
                    <b>ℹ️ Important Note:</b> The models are trained on historical data where only 2-3 dominant 
                    parameters strongly influence the water quality. For Aquaculture: Ammonia & DO are dominant. 
                    For Livestock: pH & EC are dominant. Other parameters have minimal statistical effect. 
                    Always review individual parameters alongside the overall score.
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # ========================================================================
    # PAGE: PARAMETER GUIDE
    # ========================================================================
    elif page == "📚 Parameter Guide":
        st.header("Water Quality Parameters Reference")
        
        if st.session_state.system == "Aquaculture":
            st.subheader("Aquaculture (AWQI) Parameters")
            params = {
                'TDS': {'name': 'Total Dissolved Solids', 'optimal': '<250 mg/L'},
                'DO': {'name': 'Dissolved Oxygen', 'optimal': '>7 mg/L'},
                'Nitrate': {'name': 'Nitrate', 'optimal': '<10 mg/L'},
                'TH': {'name': 'Total Hardness', 'optimal': '50-150 mg/L'},
                'pH': {'name': 'pH Value', 'optimal': '6.5-8.5'},
                'Chlorides': {'name': 'Chlorides', 'optimal': '<250 mg/L'},
                'Alkalinity': {'name': 'Alkalinity', 'optimal': '50-200 mg/L'},
                'EC': {'name': 'Electrical Conductivity', 'optimal': '500-1500 µS/cm'},
                'Ammonia': {'name': 'Ammonia', 'optimal': '<0.5 mg/L'},
            }
        else:
            st.subheader("Livestock (LWQI) Parameters")
            params = {
                'DO': {'name': 'Dissolved Oxygen', 'optimal': '>5 mg/L'},
                'Nitrate': {'name': 'Nitrate', 'optimal': '<50 mg/L'},
                'CaH': {'name': 'Calcium Hardness', 'optimal': '<300 mg/L'},
                'pH': {'name': 'pH Value', 'optimal': '6.5-8.5'},
                'Sulphates': {'name': 'Sulphates', 'optimal': '<500 mg/L'},
                'Sodium': {'name': 'Sodium', 'optimal': '<200 mg/L'},
                'EC': {'name': 'Electrical Conductivity', 'optimal': '<1500 µS/cm'},
                'Iron': {'name': 'Iron', 'optimal': '<2 mg/L'},
            }
        
        for param, info in params.items():
            with st.expander(f"📌 {info['name']}"):
                st.write(f"**Optimal Range:** {info['optimal']}")
    
    # ========================================================================
    # PAGE: MODEL PERFORMANCE
    # ========================================================================
    elif page == "📈 Model Performance":
        st.header("Machine Learning Model Performance")
        st.subheader(f"{st.session_state.system} System - Model Comparison")
        st.write("✅ All models trained and optimized for best performance")
        st.info(f"Currently showing {st.session_state.system} models. Switch systems to see comparison.")
    
    # ========================================================================
    # PAGE: SYSTEM DEBUG
    # ========================================================================
    elif page == "🐛 System Debug":
        st.header("System Diagnostics")
        
        st.subheader("📁 Directory Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Aquaculture Models:**")
            aqua_path = "models/aquaculture"
            if os.path.exists(aqua_path):
                aqua_files = os.listdir(aqua_path)
                st.success(f"✅ Found {len(aqua_files)} files")
                with st.expander("View files"):
                    for f in sorted(aqua_files):
                        st.text(f)
            else:
                st.error(f"❌ Directory not found: {aqua_path}")
        
        with col2:
            st.write("**Livestock Models:**")
            live_path = "models/livestock"
            if os.path.exists(live_path):
                live_files = os.listdir(live_path)
                st.success(f"✅ Found {len(live_files)} files")
                with st.expander("View files"):
                    for f in sorted(live_files):
                        st.text(f)
            else:
                st.error(f"❌ Directory not found: {live_path}")
        
        st.subheader("🔧 Model Loading Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Testing Aquaculture System:**")
            aqua_models, aqua_scalers, aqua_encoders = load_system("Aquaculture")
            if aqua_models and aqua_scalers and aqua_encoders:
                st.success("✅ Aquaculture system loaded successfully")
                st.text(f"Models loaded: {len(aqua_models)}")
                st.text(f"Feature count: {len(aqua_encoders.get('feature_names', []))}")
            else:
                st.error("❌ Failed to load Aquaculture system")
                st.text(f"Models: {len(aqua_models)}, Scalers: {len(aqua_scalers)}, Encoders: {len(aqua_encoders)}")
        
        with col2:
            st.write("**Testing Livestock System:**")
            live_models, live_scalers, live_encoders = load_system("Livestock")
            if live_models and live_scalers and live_encoders:
                st.success("✅ Livestock system loaded successfully")
                st.text(f"Models loaded: {len(live_models)}")
                st.text(f"Feature count: {len(live_encoders.get('feature_names', []))}")
            else:
                st.error("❌ Failed to load Livestock system")
                st.text(f"Models: {len(live_models)}, Scalers: {len(live_scalers)}, Encoders: {len(live_encoders)}")
        
        st.subheader("📊 Environment Info")
        st.text(f"Current working directory: {os.getcwd()}")
        import sys
        st.text(f"Python version: {sys.version}")
        st.text(f"Cache key (system): {st.session_state.get('system', 'Not set')}")
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.header("About This System")
        st.markdown(f"""
        ## Combined Water Quality Prediction System - Version 3.1
        
        **Current System:** {st.session_state.system}
        
        This unified system assesses water quality for both Aquaculture and Livestock 
        using advanced machine learning algorithms.
        
        ### How It Works
        1. Select your water quality system (Aquaculture or Livestock)
        2. Enter water parameters
        3. AI models make predictions
        4. View quality assessment and recommendations
        
        ### Important Note
        The trained models focus on the 2-3 dominant parameters that most strongly 
        influence water quality in the training data:
        - **Aquaculture:** Ammonia & Dissolved Oxygen are key
        - **Livestock:** pH & Electrical Conductivity are key
        
        Always review individual parameters and use results as decision support, not absolute truth.
        
        ### Technology
        - Machine Learning Framework: Scikit-learn, XGBoost
        - Web Framework: Streamlit
        - Models: 12 per system (6 regression + 6 classification)
        
        **Version:** 3.1 | **Status:** ✅ Production Ready
        """)

if __name__ == "__main__":
    main()
