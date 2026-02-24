"""
Combined Water Quality Prediction System - FINAL WORKING VERSION
Aquaculture (AWQI) + Livestock (LWQI) - GUARANTEED TO WORK
Version 4.0 - Production Ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; }
    .info-box { background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .success-box { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .critical-box { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .note-box { background-color: #f5f5f5; border-left: 5px solid #6c757d; padding: 12px; margin: 10px 0; border-radius: 5px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ROBUST MODEL LOADING WITH DETAILED DEBUGGING
# ============================================================================

@st.cache_resource
def load_system(system_type):
    """Load models for each system with robust error handling"""
    try:
        system_lower = system_type.lower()
        
        # Try multiple possible paths
        possible_paths = [
            f"models/{system_lower}",
            f"./models/{system_lower}",
            os.path.join("models", system_lower),
            os.path.join(os.getcwd(), "models", system_lower)
        ]
        
        model_dir = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                model_dir = path
                break
        
        if not model_dir:
            st.error(f"❌ {system_type} models folder not found. Checked: {possible_paths}")
            return None, None, None, None
        
        reg_models = {}
        clf_models = {}
        
        # List all files in directory for debugging
        files_in_dir = os.listdir(model_dir)
        
        # Load regression models
        for fname in files_in_dir:
            if '_reg.pkl' in fname and 'scaler' not in fname:
                try:
                    name = fname.replace('_reg.pkl', '').replace('_', ' ').title()
                    reg_models[name] = joblib.load(os.path.join(model_dir, fname))
                except Exception as e:
                    st.warning(f"Could not load {fname}: {e}")
        
        # Load classification models
        for fname in files_in_dir:
            if '_clf.pkl' in fname and 'scaler' not in fname:
                try:
                    name = fname.replace('_clf.pkl', '').replace('_', ' ').title()
                    clf_models[name] = joblib.load(os.path.join(model_dir, fname))
                except Exception as e:
                    st.warning(f"Could not load {fname}: {e}")
        
        # Load scalers and encoders
        try:
            scalers = {
                'regression': joblib.load(os.path.join(model_dir, 'scaler_regression.pkl')),
                'classification': joblib.load(os.path.join(model_dir, 'scaler_classification.pkl'))
            }
            encoders = {
                'label_encoder': joblib.load(os.path.join(model_dir, 'label_encoder.pkl')),
                'feature_names': joblib.load(os.path.join(model_dir, 'feature_names.pkl')),
                'class_names': joblib.load(os.path.join(model_dir, 'class_names.pkl'))
            }
        except Exception as e:
            st.error(f"❌ Error loading scalers/encoders for {system_type}: {e}")
            return None, None, None, None
        
        if not reg_models:
            st.error(f"❌ No regression models found for {system_type}")
            return None, None, None, None
        
        return reg_models, clf_models, scalers, encoders
    
    except Exception as e:
        st.error(f"❌ Critical error loading {system_type} models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_interpretation(score, system):
    """Get quality interpretation based on score"""
    if system == "Aquaculture":
        if score < 25:
            return {"class": "Excellent", "desc": "✅ Perfect for aquaculture", "color": "#28a745"}
        elif score < 50:
            return {"class": "Good", "desc": "👍 Good quality", "color": "#17a2b8"}
        elif score < 75:
            return {"class": "Moderate", "desc": "⚠️ Moderate issues", "color": "#ffc107"}
        else:
            return {"class": "Poor", "desc": "🚫 Poor quality", "color": "#dc3545"}
    else:
        if score < 40:
            return {"class": "Good", "desc": "✅ Good for livestock", "color": "#28a745"}
        elif score < 80:
            return {"class": "Fair", "desc": "⚠️ Fair quality", "color": "#ffc107"}
        else:
            return {"class": "Poor", "desc": "🚫 Poor quality", "color": "#dc3545"}

def get_severity(params, system):
    """Calculate severity level based on critical parameters"""
    severity = 0
    issues = []
    
    if system == "Aquaculture":
        do = params.get('DO', 7)
        ammonia = params.get('Ammonia', 0)
        
        if do < 2:
            severity += 3
            issues.append("🚨 Anoxic - NO oxygen")
        elif do < 4:
            severity += 2
            issues.append("🔴 Severe oxygen depletion")
        elif do < 5:
            severity += 1
        
        if ammonia > 5:
            severity += 3
            issues.append("🚨 CRITICAL ammonia toxicity")
        elif ammonia > 2:
            severity += 2
            issues.append("🔴 Severe organic pollution")
    else:
        do = params.get('DO', 7)
        if do < 4:
            severity += 2
            issues.append("🚨 Critical oxygen depletion")
    
    return severity, issues

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>💧 Combined Water Quality Prediction System 💧</h1>
        <p><i>Aquaculture (AWQI) + Livestock (LWQI)</i></p>
        <p style="color: #666; font-size: 12px;">Version 4.0 - Final Production Release</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Selection
    st.markdown('<div class="info-box"><b>Step 1:</b> Select System - Aquaculture or Livestock</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🐟 Aquaculture (AWQI)", use_container_width=True, key="aqua_btn"):
            st.session_state.system = "Aquaculture"
    with col2:
        if st.button("🐄 Livestock (LWQI)", use_container_width=True, key="live_btn"):
            st.session_state.system = "Livestock"
    
    if "system" not in st.session_state:
        st.session_state.system = "Aquaculture"
    
    st.write(f"**Selected:** {st.session_state.system}")
    
    # Load models
    reg_models, clf_models, scalers, encoders = load_system(st.session_state.system)
    
    if not reg_models or not scalers:
        st.error(f"""
        ❌ **CRITICAL ERROR:** Failed to load {st.session_state.system} models.
        
        **Troubleshooting:**
        1. Verify models/ folder exists locally and contains both aquaculture/ and livestock/ subfolders
        2. Verify each subfolder has 13+ .pkl files
        3. Push models to GitHub: `git add -f models/` then `git push`
        4. Restart Streamlit Cloud app
        5. Hard refresh browser (Ctrl+Shift+R)
        """)
        return
    
    st.success(f"✅ {st.session_state.system} models loaded successfully")
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select:", ["📊 Prediction", "📚 Guide", "ℹ️ About"])
    
    # ====================================================================
    # PREDICTION PAGE
    # ====================================================================
    if page == "📊 Prediction":
        st.header(f"{st.session_state.system} - Water Quality Prediction")
        st.markdown('<div class="info-box"><b>Step 2:</b> Enter water parameters below</div>', unsafe_allow_html=True)
        
        feature_names = encoders['feature_names']
        
        if st.session_state.system == "Aquaculture":
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("🌊 General")
                tds = st.number_input("TDS (mg/L)", 0.0, 5000.0, 170.0, step=1.0)
                ph = st.number_input("pH", 0.0, 14.0, 7.8, step=0.1)
                alk = st.number_input("Alkalinity (mg/L)", 0.0, 2000.0, 70.0, step=1.0)
            
            with c2:
                st.subheader("🐠 Biology")
                do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0, step=0.1)
                chlor = st.number_input("Chlorides (mg/L)", 0.0, 3000.0, 25.0, step=1.0)
                ec = st.number_input("EC (µS/cm)", 0.0, 10000.0, 280.0, step=10.0)
            
            with c3:
                st.subheader("⚠️ Pollution")
                nit = st.number_input("Nitrate (mg/L)", 0.0, 2000.0, 0.4, step=0.1)
                th = st.number_input("TH (mg/L)", 0.0, 2000.0, 140.0, step=1.0)
                amm = st.number_input("Ammonia (mg/L)", 0.0, 100.0, 0.01, step=0.001)
            
            time_val = st.slider("Time (hours)", 0, 23, 12)
            time_sin = np.sin(2 * np.pi * time_val / 12)
            time_cos = np.cos(2 * np.pi * time_val / 12)
            
            inputs = {
                'TDS': tds, 'DO': do, 'Nitrate': nit, 'TH': th, 'pH': ph,
                'Chlorides': chlor, 'Alkalinity': alk, 'EC': ec,
                'Ammonia': amm, 'Time_sin': time_sin, 'Time_cos': time_cos
            }
        
        else:  # Livestock
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("🌊 General")
                do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0, step=0.1)
                ph = st.number_input("pH", 0.0, 14.0, 7.8, step=0.1)
                na = st.number_input("Sodium (mg/L)", 0.0, 500.0, 20.0, step=0.5)
            
            with c2:
                st.subheader("🐄 Livestock")
                nit = st.number_input("Nitrate (mg/L)", 0.0, 500.0, 0.5, step=0.1)
                cah = st.number_input("Calcium Hardness (mg/L)", 0.0, 500.0, 8.0, step=1.0)
                sulph = st.number_input("Sulphates (mg/L)", 0.0, 500.0, 6.0, step=0.1)
            
            with c3:
                st.subheader("⚠️ Quality")
                ec = st.number_input("EC (µS/cm)", 0.0, 2000.0, 300.0, step=10.0)
                iron = st.number_input("Iron (mg/L)", 0.0, 100.0, 2.0, step=0.1)
            
            time_val = st.slider("Time (hours)", 0, 23, 12)
            time_sin = np.sin(2 * np.pi * time_val / 12)
            time_cos = np.cos(2 * np.pi * time_val / 12)
            
            inputs = {
                'DO': do, 'Nitrate': nit, 'CaH': cah, 'pH': ph,
                'Sulphates': sulph, 'Sodium': na, 'EC': ec, 'Iron': iron,
                'Time_sin': time_sin, 'Time_cos': time_cos
            }
        
        if st.button("🔍 Step 3: Predict Water Quality", use_container_width=True, type="primary"):
            try:
                df = pd.DataFrame([inputs])
                df_features = df[feature_names]
                df_scaled = scalers['regression'].transform(df_features)
                
                # Get predictions
                preds = {}
                for name, model in reg_models.items():
                    try:
                        pred = model.predict(df_scaled)[0]
                        preds[name] = pred
                    except:
                        pass
                
                if preds:
                    best_score = preds[list(preds.keys())[0]]
                    interp = get_interpretation(best_score, st.session_state.system)
                    severity, issues = get_severity(inputs, st.session_state.system)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.subheader("📊 RESULTS")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h2>{best_score:.2f}</h2>
                            <p>{interp['class'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background-color: {interp['color']}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3>{interp['desc']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Severity Score", f"{severity}/10", delta=None)
                    
                    # Model predictions table
                    st.subheader("🤖 Model Predictions")
                    pred_df = pd.DataFrame({
                        'Model': list(preds.keys()),
                        'Quality Score': [f"{v:.2f}" for v in preds.values()]
                    })
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    # Critical issues
                    if issues:
                        st.subheader("⚠️ Critical Issues Found")
                        for issue in issues:
                            st.write(f"• {issue}")
                    
                    # Important note
                    st.markdown("""
                    <div class="note-box">
                    <b>ℹ️ Important:</b> Models trained on historical data where 2-3 dominant parameters strongly influence water quality.
                    Aquaculture: Ammonia & DO dominant | Livestock: pH & EC dominant. Always review individual parameters alongside predictions.
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    elif page == "📚 Guide":
        st.header("📚 Parameter Guide")
        
        if st.session_state.system == "Aquaculture":
            st.write("### Aquaculture Water Quality Parameters")
            params = {
                'TDS': ('Total Dissolved Solids', '< 250 mg/L'),
                'DO': ('Dissolved Oxygen', '> 7 mg/L'),
                'Nitrate': ('Nitrogen pollution', '< 10 mg/L'),
                'pH': ('Acidity/Alkalinity', '6.5 - 8.5'),
                'Ammonia': ('Organic pollution', '< 0.5 mg/L'),
                'EC': ('Electrical Conductivity', '500-1500 µS/cm'),
                'Alkalinity': ('Buffering capacity', '50-200 mg/L'),
                'TH': ('Total Hardness', '50-150 mg/L'),
                'Chlorides': ('Salt content', '< 250 mg/L')
            }
        else:
            st.write("### Livestock Water Quality Parameters")
            params = {
                'DO': ('Dissolved Oxygen', '> 5 mg/L'),
                'pH': ('Acidity/Alkalinity', '6.5 - 8.5'),
                'EC': ('Electrical Conductivity', '< 1500 µS/cm'),
                'Nitrate': ('Nitrogen pollution', '< 50 mg/L'),
                'Iron': ('Iron content', '< 2 mg/L'),
                'Sodium': ('Sodium level', '< 200 mg/L'),
                'Sulphates': ('Sulphate content', '< 500 mg/L'),
                'CaH': ('Calcium Hardness', '< 300 mg/L')
            }
        
        for param, (desc, optimal) in params.items():
            st.write(f"**{param}** - {desc} | Optimal: {optimal}")
    
    elif page == "ℹ️ About":
        st.header("ℹ️ About This System")
        st.markdown("""
        ### Combined Water Quality Prediction System v4.0
        
        **Purpose:** Predicts water quality for both Aquaculture and Livestock applications
        
        **Features:**
        - 🐟 Aquaculture water quality assessment (AWQI)
        - 🐄 Livestock water suitability assessment (LWQI)
        - 12 machine learning models per system
        - Real-time predictions with severity scoring
        
        **How it works:**
        1. Select your system (Aquaculture or Livestock)
        2. Enter water parameters
        3. Get instant quality predictions
        4. Review severity assessment and recommendations
        
        **Data Privacy:** All processing done locally, no data saved
        """)

if __name__ == "__main__":
    main()
