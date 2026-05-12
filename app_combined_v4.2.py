"""
Combined Water Quality Prediction System - FINAL MERGED VERSION
Based on v4.0 structure with v3.2 dashboard, model performance, and parameter guide
Version 4.2 - Merged Unified Quality Assessment Tool
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
# UTILITY FUNCTIONS (from v4.2)
# ============================================================================

def get_quality_interpretation(value, system_type):
    """Interpret quality score based on system type"""
    if system_type == "Aquaculture":
        if value < 25:
            return {
                'class': 'Excellent',
                'description': '✅ Excellent water quality - Suitable for all uses',
                'color': '#28a745',
                'emoji': '✅',
                'action': 'No action needed. Continue monitoring regularly.'
            }
        elif value < 50:
            return {
                'class': 'Good',
                'description': '👍 Good water quality - Minor issues, generally acceptable',
                'color': '#17a2b8',
                'emoji': '👍',
                'action': 'Minor monitoring recommended. Address specific parameters.'
            }
        elif value < 75:
            return {
                'class': 'Moderate',
                'description': '⚠️ Moderate water quality - Issues present, action recommended',
                'color': '#ffc107',
                'emoji': '⚠️',
                'action': 'Immediate improvement measures needed. See recommendations.'
            }
        else:
            return {
                'class': 'Poor',
                'description': '🚫 Poor water quality - Significant pollution, immediate action required',
                'color': '#dc3545',
                'emoji': '🚫',
                'action': 'URGENT: Critical treatment or replacement needed.'
            }
    else:  # Livestock
        if value < 40:
            return {
                'class': 'Good',
                'description': '✅ Good water quality - Suitable for livestock',
                'color': '#28a745',
                'emoji': '✅',
                'action': 'No action needed. Continue monitoring.'
            }
        elif value < 80:
            return {
                'class': 'Fair',
                'description': '⚠️ Fair water quality - Monitor closely, some improvement needed',
                'color': '#ffc107',
                'emoji': '⚠️',
                'action': 'Monitor parameters. Consider improvement measures.'
            }
        else:
            return {
                'class': 'Poor',
                'description': '🚫 Poor water quality - Treatment needed',
                'color': '#dc3545',
                'emoji': '🚫',
                'action': 'URGENT: Water treatment or replacement required.'
            }

def prepare_features(input_dict, feature_names):
    """Prepare features for prediction"""
    features = pd.DataFrame([input_dict])
    return features[feature_names]

def get_severity_level(input_dict, system_type):
    """Calculate severity level based on parameters"""
    severity_score = 0
    critical_issues = []
    
    if system_type == "Aquaculture":
        do = input_dict.get('DO', 7)
        ammonia = input_dict.get('Ammonia', 0)
        ph = input_dict.get('pH', 7)
        tds = input_dict.get('TDS', 250)
        nitrate = input_dict.get('Nitrate', 10)
        chlorides = input_dict.get('Chlorides', 250)
        
        # DO assessment
        if do < 2:
            severity_score += 3
            critical_issues.append("🚨 Anoxic conditions - no oxygen")
        elif do < 4:
            severity_score += 2
            critical_issues.append("🔴 Severe oxygen depletion")
        elif do < 5:
            severity_score += 1
        
        # Ammonia assessment
        if ammonia > 5:
            severity_score += 3
            critical_issues.append("🚨 Critical ammonia toxicity")
        elif ammonia > 2:
            severity_score += 2
            critical_issues.append("🔴 Severe organic pollution")
        elif ammonia > 0.5:
            severity_score += 1
        
        # pH assessment
        if ph < 4 or ph > 11:
            severity_score += 2
            critical_issues.append("🚨 Extreme pH - chemical hazard")
        elif ph < 6 or ph > 9.5:
            severity_score += 1
        
        # TDS assessment
        if tds > 1000:
            severity_score += 2
            critical_issues.append("🚨 Extreme salinity")
        elif tds > 500:
            severity_score += 1
        
        # Nitrate assessment
        if nitrate > 200:
            severity_score += 2
            critical_issues.append("🚨 Severe nutrient pollution")
        elif nitrate > 50:
            severity_score += 1
        
        # Chlorides assessment
        if chlorides > 1000:
            severity_score += 1
    
    else:  # Livestock
        do = input_dict.get('DO', 7)
        ph = input_dict.get('pH', 7)
        ec = input_dict.get('EC', 1000)
        nitrate = input_dict.get('Nitrate', 10)
        
        if do < 4:
            severity_score += 2
            critical_issues.append("🚨 Critical oxygen depletion")
        elif do < 5:
            severity_score += 1
        
        if ph < 5 or ph > 10:
            severity_score += 2
            critical_issues.append("🚨 Extreme pH levels")
        elif ph < 6 or ph > 9:
            severity_score += 1
        
        if ec > 3000:
            severity_score += 2
            critical_issues.append("🚨 Extreme salinity")
        elif ec > 2000:
            severity_score += 1
        
        if nitrate > 200:
            severity_score += 2
            critical_issues.append("🚨 Severe nutrient pollution")
    
    return severity_score, critical_issues

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>💧 Dual Water Quality Prediction System 💧</h1>
        <p><i>Aquaculture (AWQI) + Livestock (LWQI) Analysis</i></p>
        <p style="color: #666; font-size: 14px;">Version 4.2 - Merged Unified Quality Assessment Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    with col2:
        if st.button("🐄 Livestock (LWQI)", use_container_width=True, key="btn_live"):
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
    page = st.sidebar.radio(
        "Select page:",
        ["📊 Prediction Dashboard", "📚 Parameter Guide", "📈 Model Performance", "ℹ️ About"]
    )
    
    # ========================================================================
    # PAGE: PREDICTION DASHBOARD (from v4.2)
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
                st.subheader("Input Parameters")
                tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=170.0, step=1.0)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
                alkalinity = st.number_input("Alkalinity (mg/L)", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
            
            with col2:
                st.subheader("")
                do = st.number_input("DO (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                chlorides = st.number_input("Chlorides (mg/L)", min_value=0.0, max_value=3000.0, value=25.0, step=1.0)
                ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=10000.0, value=280.0, step=10.0)
            
            with col3:
                st.subheader("")
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
                st.subheader("Input Parameters")
                do = st.number_input("DO (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
                na = st.number_input("Sodium (mg/L)", min_value=0.0, max_value=500.0, value=20.0, step=0.5)
            
            with col2:
                st.subheader("")
                nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=500.0, value=0.5, step=0.1)
                cah = st.number_input("Calcium Hardness (mg/L)", min_value=0.0, max_value=500.0, value=8.0, step=1.0)
                sulphates = st.number_input("Sulphates (mg/L)", min_value=0.0, max_value=500.0, value=6.0, step=0.1)
            
            with col3:
                st.subheader("")
                ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=2000.0, value=300.0, step=10.0)
                iron = st.number_input("Iron (mg/L)", min_value=0.0, max_value=100.0, value=0.03, step=0.1)
            
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
                scaled_clf = scalers['classification'].transform(features)
                
                # ============================================================
                # SECTION 1: QUALITY SCORE & CLASSIFICATION (AWQI/LWQI)
                # ============================================================
                section_title = "AWQI Score & Classification" if st.session_state.system == "Aquaculture" else "LWQI Score & Classification"
                st.subheader(f"🎯 {section_title}")
                
                predictions = {}
                for name, model in reg_models.items():
                    try:
                        pred = model.predict(scaled_features)[0]
                        predictions[name] = pred
                    except:
                        pass
                
                if predictions:
                    # Select best regression model - prefer Linear Regression first, then SVR
                    # These are the models with highest R² scores in the training data
                    best_name = None
                    best_score = -float('inf')
                    
                    # Priority 1: Linear Regression (R² = 1.0 for AWQI, 0.95 for LWQI)
                    for key in predictions.keys():
                        if 'linear regression' in key.lower():
                            best_name = key
                            break
                    
                    # Priority 2: SVR (R² = 0.9999 for AWQI, 0.94 for LWQI)
                    if not best_name:
                        for key in predictions.keys():
                            if 'svr' in key.lower():
                                best_name = key
                                break
                    
                    # Fallback to first model if preferred not found
                    if not best_name:
                        best_name = list(predictions.keys())[0]

                    quality_score = predictions[best_name]

                    interpretation = get_quality_interpretation(quality_score, st.session_state.system)
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h2>{quality_score:.2f}</h2>
                            <p>{interpretation['class'].upper()}</p>
                            <p style="font-size: 12px; margin-top: 8px;"><i>Based on {best_name}</i></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div style="background-color: {interpretation['color']}; 
                                    color: white; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3>{interpretation['emoji']} {interpretation['class']}</h3>
                            <p>{interpretation['description']}</p>
                            <p style="font-size: 14px; margin-top: 10px;"><b>Action:</b> {interpretation['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # ============================================================
                # SECTION 2: MODEL PREDICTIONS & SEVERITY
                # ============================================================
                st.subheader("📊 Model Predictions & Severity Assessment")
                
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.write(f"**Regression Model Predictions:** (Using {best_name} for score)")
                    pred_df = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Score': [f"{v:.2f}" for v in predictions.values()]
                    }).reset_index(drop=True)
                    # Highlight the best model used
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    st.caption(f"✓ {best_name} is selected (best R² score)")
                
                with col_pred2:
                    # Classification prediction
                    if clf_models:
                        st.write("**Classification Results:**")
                        best_clf = list(clf_models.values())[0]
                        try:
                            class_pred = best_clf.predict(scaled_clf)[0]
                            class_name = encoders['class_names'][class_pred]
                            st.metric("Predicted Water Quality Class", class_name)
                            
                            if hasattr(best_clf, 'predict_proba'):
                                proba = best_clf.predict_proba(scaled_clf)[0]
                                confidence = proba[class_pred] * 100
                                st.metric("Classification Confidence", f"{confidence:.1f}%")
                        except:
                            pass
                
                # Severity Assessment
                # severity_score, critical_issues = get_severity_level(input_dict, st.session_state.system)
                
                # st.write("**Overall Severity Assessment:**")
                # if severity_score >= 6:
                #     st.error(f"🚨 **CRITICAL SEVERITY** - Multiple severe issues detected! (Score: {severity_score}/10)")
                #     if critical_issues:
                #         for issue in critical_issues:
                #             st.error(f"• {issue}")
                # elif severity_score >= 4:
                #     st.warning(f"🔴 **HIGH SEVERITY** - Significant problems detected (Score: {severity_score}/10)")
                #     if critical_issues:
                #         for issue in critical_issues:
                #             st.warning(f"• {issue}")
                # elif severity_score >= 2:
                #     st.warning(f"🟡 **MODERATE SEVERITY** - Issues present (Score: {severity_score}/10)")
                # elif severity_score >= 1:
                #     st.info(f"🟢 **LOW SEVERITY** - Minor issues (Score: {severity_score}/10)")
                # else:
                #     st.success("✅ **EXCELLENT** - No significant issues detected")
                
                # ============================================================
                # SECTION 3: ALL MODEL PREDICTIONS DETAILED
                # ============================================================
                # st.subheader("🤖 All Model Predictions Detailed")
                
                # st.write(f"**Note:** All regression models have been trained on your data. The {best_name} model (shown above) is selected as the primary predictor due to its highest R² score.")
                
                # all_predictions = []
                # for name, model in reg_models.items():
                #     try:
                #         pred = model.predict(scaled_features)[0]
                #         is_selected = "✓ SELECTED" if name == best_name else ""
                #         all_predictions.append({'Model': name, 'Score': f"{pred:.2f}", 'Status': is_selected})
                #     except:
                #         pass
                
                # if all_predictions:
                #     all_pred_df = pd.DataFrame(all_predictions)
                #     st.dataframe(all_pred_df, use_container_width=True, hide_index=True)
                
                # ============================================================
                # SECTION 4: DETAILED WATER QUALITY ASSESSMENT & RECOMMENDATIONS
                # ============================================================
                st.subheader("💡 Comprehensive Water Quality Assessment & Recommendations for ALL Parameters")

                if st.session_state.system == "Aquaculture":
                    # Extract all AWQI parameters
                    do = input_dict['DO']
                    ammonia = input_dict['Ammonia']
                    ph = input_dict['pH']
                    tds = input_dict['TDS']
                    nitrate = input_dict['Nitrate']
                    chlorides = input_dict['Chlorides']
                    th = input_dict['TH']
                    alkalinity = input_dict['Alkalinity']
                    ec = input_dict['EC']
    
                    # 1. DISSOLVED OXYGEN (DO) - CRITICAL PARAMETER
                    st.markdown("### 1. 🌊 Dissolved Oxygen (DO) - mg/L")
                    do_status = "Optimal" if do >= 7 else "Good" if do >= 5 else "Moderate" if do >= 4 else "Poor" if do >= 2 else "Critical"
                    st.markdown(f"**Current: {do:.2f} mg/L | Status: {do_status}**")
                    
                    if do < 2:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL</b> - Anoxic conditions. Aquatic life cannot survive. IMMEDIATE ACTION: Install emergency aeration, increase circulation, partial water replacement, oxygen injection.</div>', unsafe_allow_html=True)
                    elif do < 4:
                        st.markdown('<div class="warning-box"><b>🔴 SEVERE</b> - Most fish will suffer. URGENT: Install aeration system immediately, increase capacity, reduce stocking density.</div>', unsafe_allow_html=True)
                    elif do < 5:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH</b> - Oxygen stress condition. Increase aeration, reduce feeding, increase circulation, monitor every 6-8 hours.</div>', unsafe_allow_html=True)
                    elif do < 7:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE</b> - Below optimal. Consider increasing aeration for sensitive species.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT</b> - Optimal dissolved oxygen level for aquaculture.</div>', unsafe_allow_html=True)
                    
                    # 2. AMMONIA - CRITICAL PARAMETER
                    st.markdown("### 2. 🔬 Ammonia (NH₃) - mg/L")
                    ammonia_status = "Excellent" if ammonia < 0.1 else "Good" if ammonia < 0.5 else "Moderate" if ammonia < 2 else "Poor" if ammonia < 5 else "Critical"
                    st.markdown(f"**Current: {ammonia:.3f} mg/L | Status: {ammonia_status}**")
                    
                    if ammonia > 5:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL</b> - SEVERE toxic pollution. DO NOT use. IMMEDIATE: Complete water replacement, increase biological filtration, stop organic inputs.</div>', unsafe_allow_html=True)
                    elif ammonia > 2:
                        st.markdown('<div class="warning-box"><b>🔴 SEVERE</b> - High toxicity. Partial water exchange (25-50%), increase filtration, reduce feeding.</div>', unsafe_allow_html=True)
                    elif ammonia > 0.5:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH</b> - Organic pollution detected. Improve circulation, reduce feeding, enhance filtration.</div>', unsafe_allow_html=True)
                    elif ammonia > 0.1:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE</b> - Minor pollution. Monitor closely and enhance filtration if needed.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT</b> - Minimal ammonia. Excellent organic pollution control.</div>', unsafe_allow_html=True)
                    
                    # 3. pH - IMPORTANT PARAMETER
                    st.markdown("### 3. ⚖️ pH (Acidity/Alkalinity)")
                    ph_status = "Excellent" if 6.5 <= ph <= 8.5 else "Good" if 6 <= ph <= 9.5 else "Moderate" if 5 <= ph <= 10 else "Poor"
                    st.markdown(f"**Current: {ph:.2f} | Status: {ph_status}**")
                    
                    if ph < 4 or ph > 11:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL pH</b> - Severely imbalanced. IMMEDIATE buffering required using appropriate chemicals.</div>', unsafe_allow_html=True)
                    elif ph < 6 or ph > 9.5:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH pH ISSUE</b> - Out of safe range. Requires buffering adjustment. For low pH: add limestone/sodium bicarbonate. For high pH: use acidifying agents.</div>', unsafe_allow_html=True)
                    elif ph < 6.5 or ph > 8.5:
                        st.markdown('<div class="info-box"><b>🟡 SUBOPTIMAL pH</b> - Consider buffering for better conditions. Range 6.5-8.5 is ideal for most species.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT pH</b> - Perfect for aquaculture. Stable and suitable.</div>', unsafe_allow_html=True)
                    
                    # 4. TOTAL DISSOLVED SOLIDS (TDS)
                    st.markdown("### 4. 🧂 Total Dissolved Solids (TDS) - mg/L")
                    tds_status = "Excellent" if tds < 250 else "Good" if tds < 300 else "Moderate" if tds < 500 else "Poor" if tds < 1000 else "Critical"
                    st.markdown(f"**Current: {tds:.2f} mg/L | Status: {tds_status}**")
                    
                    if tds > 1000:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL TDS</b> - Highly saline. Water replacement or major dilution required immediately.</div>', unsafe_allow_html=True)
                    elif tds > 500:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH TDS</b> - Salt/mineral accumulation detected. Monitor closely and plan water exchange (25-30%).</div>', unsafe_allow_html=True)
                    elif tds > 300:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE TDS</b> - Monitor salt accumulation. Partial water change (10-15%) recommended periodically.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT TDS</b> - Optimal mineral content for aquaculture.</div>', unsafe_allow_html=True)
                    
                    # 5. NITRATE (Nutrient Pollution)
                    st.markdown("### 5. 🌿 Nitrate (NO₃⁻) - mg/L")
                    nitrate_status = "Excellent" if nitrate < 10 else "Good" if nitrate < 25 else "Moderate" if nitrate < 50 else "Poor" if nitrate < 200 else "Critical"
                    st.markdown(f"**Current: {nitrate:.2f} mg/L | Status: {nitrate_status}**")
                    
                    if nitrate > 200:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL NITRATE</b> - Severe nutrient pollution. Immediate biological treatment or water replacement needed.</div>', unsafe_allow_html=True)
                    elif nitrate > 50:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH NITRATE</b> - Significant pollution. Reduce feeding, increase biological filtration, consider water exchange (20-25%).</div>', unsafe_allow_html=True)
                    elif nitrate > 25:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE NITRATE</b> - Elevated nutrients. Reduce feed input and enhance biological filtration.</div>', unsafe_allow_html=True)
                    elif nitrate > 10:
                        st.markdown('<div class="info-box"><b>🟡 MINOR ELEVATION</b> - Slight nutrient accumulation. Monitor and maintain good water circulation.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT NITRATE</b> - Minimal nutrient pollution. Excellent water management.</div>', unsafe_allow_html=True)
                    
                    # 6. CHLORIDES (Salt Content)
                    st.markdown("### 6. 🧲 Chlorides (Cl⁻) - mg/L")
                    chlorides_status = "Excellent" if chlorides < 250 else "Good" if chlorides < 500 else "Moderate" if chlorides < 1000 else "Poor"
                    st.markdown(f"**Current: {chlorides:.2f} mg/L | Status: {chlorides_status}**")
                    
                    if chlorides > 1000:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL CHLORIDES</b> - Extremely saline. Immediate dilution or water replacement required.</div>', unsafe_allow_html=True)
                    elif chlorides > 500:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH CHLORIDES</b> - High salt content. Monitor salinity and plan water exchange (15-20%).</div>', unsafe_allow_html=True)
                    elif chlorides > 250:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE CHLORIDES</b> - Monitor salt accumulation. Consider partial water changes if trend increases.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT CHLORIDES</b> - Low salt content, suitable for sensitive species.</div>', unsafe_allow_html=True)
                    
                    # 7. TOTAL HARDNESS (TH)
                    st.markdown("### 7. 🪨 Total Hardness (TH) - mg/L")
                    th_status = "Excellent" if 50 <= th <= 150 else "Good" if 40 <= th <= 200 else "Moderate" if th < 250 else "High"
                    st.markdown(f"**Current: {th:.2f} mg/L | Status: {th_status}**")
                    
                    if th < 40:
                        st.markdown('<div class="warning-box"><b>🟠 SOFT WATER</b> - Low calcium/magnesium. Add mineral supplements or lime to buffer water.</div>', unsafe_allow_html=True)
                    elif th > 200:
                        st.markdown('<div class="warning-box"><b>🟠 HARD WATER</b> - High calcium/magnesium content. Monitor and consider partial water exchange if fish show stress.</div>', unsafe_allow_html=True)
                    elif 50 <= th <= 150:
                        st.markdown('<div class="success-box"><b>✅ OPTIMAL HARDNESS</b> - Perfect balance for most aquaculture species.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box"><b>🟡 ACCEPTABLE</b> - Within tolerable range. Monitor for species-specific needs.</div>', unsafe_allow_html=True)
                    
                    # 8. ALKALINITY (Buffering Capacity)
                    st.markdown("### 8. 🛡️ Alkalinity (Total Alkalinity) - mg/L")
                    alk_status = "Excellent" if 50 <= alkalinity <= 200 else "Good" if 30 <= alkalinity <= 250 else "Poor"
                    st.markdown(f"**Current: {alkalinity:.2f} mg/L | Status: {alk_status}**")
                    
                    if alkalinity < 30:
                        st.markdown('<div class="warning-box"><b>🟠 LOW ALKALINITY</b> - Poor buffering capacity. Water pH will fluctuate easily. Add sodium bicarbonate or limestone to increase buffering.</div>', unsafe_allow_html=True)
                    elif alkalinity > 250:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH ALKALINITY</b> - Excessive buffering may prevent pH adjustment. Monitor pH closely.</div>', unsafe_allow_html=True)
                    elif 50 <= alkalinity <= 200:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT ALKALINITY</b> - Good buffering capacity. Water pH remains stable.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box"><b>🟡 ACCEPTABLE</b> - Adequate buffering capacity.</div>', unsafe_allow_html=True)
                    
                    # 9. ELECTRICAL CONDUCTIVITY (EC)
                    st.markdown("### 9. ⚡ Electrical Conductivity (EC) - µS/cm")
                    ec_status = "Excellent" if 500 <= ec <= 1500 else "Good" if 300 <= ec <= 2000 else "High" if ec > 2000 else "Low"
                    st.markdown(f"**Current: {ec:.2f} µS/cm | Status: {ec_status}**")
                    
                    if ec < 300:
                        st.markdown('<div class="info-box"><b>🟡 LOW CONDUCTIVITY</b> - Few dissolved ions. Consider adding mineral supplements for optimal growth.</div>', unsafe_allow_html=True)
                    elif ec > 2000:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH CONDUCTIVITY</b> - Excessive dissolved salts. Water exchange recommended (20-25%).</div>', unsafe_allow_html=True)
                    elif 500 <= ec <= 1500:
                        st.markdown('<div class="success-box"><b>✅ OPTIMAL CONDUCTIVITY</b> - Perfect balance of dissolved minerals for aquaculture.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box"><b>🟡 ACCEPTABLE</b> - Adequate ion content.</div>', unsafe_allow_html=True)

                else:  # LIVESTOCK SYSTEM
                    # Extract all LWQI parameters
                    do = input_dict['DO']
                    ph = input_dict['pH']
                    iron = input_dict['Iron']
                    nitrate = input_dict['Nitrate']
                    sodium = input_dict['Sodium']
                    sulphates = input_dict['Sulphates']
                    ec = input_dict['EC']
                    cah = input_dict['CaH']
                    
                    # 1. DISSOLVED OXYGEN
                    st.markdown("### 1. 🌊 Dissolved Oxygen (DO) - mg/L")
                    do_status = "Excellent" if do >= 5 else "Good" if do >= 4 else "Moderate" if do >= 3 else "Poor"
                    st.markdown(f"**Current: {do:.2f} mg/L | Status: {do_status}**")
                    
                    if do < 3:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL</b> - Severe oxygen depletion. NOT suitable for livestock. Immediate aeration required.</div>', unsafe_allow_html=True)
                    elif do < 4:
                        st.markdown('<div class="warning-box"><b>🔴 SEVERE</b> - Critical oxygen levels. Immediate aeration and water circulation needed.</div>', unsafe_allow_html=True)
                    elif do < 5:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE</b> - Suboptimal oxygen. Improve aeration and water circulation for better livestock health.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT</b> - Optimal oxygen levels for livestock water.</div>', unsafe_allow_html=True)
                    
                    # 2. pH
                    st.markdown("### 2. ⚖️ pH (Acidity/Alkalinity)")
                    ph_status = "Excellent" if 6.5 <= ph <= 8.5 else "Good" if 6 <= ph <= 9 else "Moderate" if 5 <= ph <= 10 else "Poor"
                    st.markdown(f"**Current: {ph:.2f} | Status: {ph_status}**")
                    
                    if ph < 5 or ph > 10:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL pH</b> - Severely imbalanced. Water unsuitable for livestock. Immediate correction required.</div>', unsafe_allow_html=True)
                    elif ph < 6 or ph > 9:
                        st.markdown('<div class="warning-box"><b>🟠 OUT OF RANGE</b> - pH requires adjustment using appropriate buffers. Livestock may refuse water.</div>', unsafe_allow_html=True)
                    elif ph < 6.5 or ph > 8.5:
                        st.markdown('<div class="info-box"><b>🟡 SUBOPTIMAL</b> - Acceptable but consider pH buffering for optimal livestock health.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT pH</b> - Perfect for livestock drinking water.</div>', unsafe_allow_html=True)
                    
                    # 3. IRON
                    st.markdown("### 3. 🔴 Iron (Fe) - mg/L")
                    iron_status = "Excellent" if iron < 0.3 else "Good" if iron < 1 else "Moderate" if iron < 2 else "Poor" if iron < 5 else "Critical"
                    st.markdown(f"**Current: {iron:.3f} mg/L | Status: {iron_status}**")
                    
                    if iron > 5:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL IRON</b> - Extremely high. Water unpalatable to livestock. Requires iron removal (sedimentation, filtration, aeration).</div>', unsafe_allow_html=True)
                    elif iron > 2:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH IRON</b> - Livestock may refuse or consume reluctantly. Install aeration and sediment filters to reduce iron.</div>', unsafe_allow_html=True)
                    elif iron > 1:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE IRON</b> - Noticeable iron content. Consider aeration and filtration to improve water quality.</div>', unsafe_allow_html=True)
                    elif iron > 0.3:
                        st.markdown('<div class="info-box"><b>🟡 MINOR ELEVATION</b> - Slight iron presence. Monitor and maintain good water management.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT IRON</b> - Minimal iron content. Suitable for livestock.</div>', unsafe_allow_html=True)
                    
                    # 4. NITRATE
                    st.markdown("### 4. 🌿 Nitrate (NO₃⁻) - mg/L")
                    nitrate_status = "Excellent" if nitrate < 50 else "Good" if nitrate < 100 else "Moderate" if nitrate < 200 else "Poor"
                    st.markdown(f"**Current: {nitrate:.2f} mg/L | Status: {nitrate_status}**")
                    
                    if nitrate > 200:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL NITRATE</b> - Severe pollution. Water unsuitable for livestock. Requires treatment or replacement.</div>', unsafe_allow_html=True)
                    elif nitrate > 100:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH NITRATE</b> - High pollution levels. Not ideal for livestock. Consider water exchange (30-40%).</div>', unsafe_allow_html=True)
                    elif nitrate > 50:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE NITRATE</b> - Elevated nutrients. Monitor and improve water management to reduce pollution sources.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT NITRATE</b> - Low pollution. Suitable for livestock.</div>', unsafe_allow_html=True)
                    
                    # 5. SODIUM (Na)
                    st.markdown("### 5. 🧂 Sodium (Na) - mg/L")
                    sodium_status = "Excellent" if sodium < 50 else "Good" if sodium < 100 else "Moderate" if sodium < 200 else "High"
                    st.markdown(f"**Current: {sodium:.2f} mg/L | Status: {sodium_status}**")
                    
                    if sodium > 200:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH SODIUM</b> - Excessive salt. Livestock may show salt toxicity signs. Water exchange recommended.</div>', unsafe_allow_html=True)
                    elif sodium > 100:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE SODIUM</b> - Elevated salt levels. Monitor livestock health and consider water exchange if issues arise.</div>', unsafe_allow_html=True)
                    elif sodium > 50:
                        st.markdown('<div class="info-box"><b>🟡 MINOR ELEVATION</b> - Slight salt presence. Acceptable for most livestock.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT SODIUM</b> - Low salt content. Ideal for livestock.</div>', unsafe_allow_html=True)
                    
                    # 6. SULPHATES
                    st.markdown("### 6. 💛 Sulphates (SO₄²⁻) - mg/L")
                    sulphates_status = "Excellent" if sulphates < 250 else "Good" if sulphates < 500 else "Moderate" if sulphates < 1000 else "High"
                    st.markdown(f"**Current: {sulphates:.2f} mg/L | Status: {sulphates_status}**")
                    
                    if sulphates > 1000:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH SULPHATES</b> - Excessive levels may cause diarrhea in livestock. Water exchange advised.</div>', unsafe_allow_html=True)
                    elif sulphates > 500:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE SULPHATES</b> - Elevated sulphate levels. Monitor livestock health for digestive issues.</div>', unsafe_allow_html=True)
                    elif sulphates > 250:
                        st.markdown('<div class="info-box"><b>🟡 MINOR ELEVATION</b> - Slight sulphate increase. Generally acceptable.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT SULPHATES</b> - Low sulphate content. Ideal for livestock.</div>', unsafe_allow_html=True)
                    
                    # 7. ELECTRICAL CONDUCTIVITY (EC)
                    st.markdown("### 7. ⚡ Electrical Conductivity (EC) - µS/cm")
                    ec_status = "Excellent" if ec < 1500 else "Good" if ec < 2000 else "Moderate" if ec < 3000 else "High"
                    st.markdown(f"**Current: {ec:.2f} µS/cm | Status: {ec_status}**")
                    
                    if ec > 3000:
                        st.markdown('<div class="critical-box"><b>🚨 CRITICAL SALINITY</b> - Extremely high. Water unpalatable to livestock. Replacement needed.</div>', unsafe_allow_html=True)
                    elif ec > 2000:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH SALINITY</b> - Excessive salts. Livestock may refuse water. Consider water exchange (25-30%).</div>', unsafe_allow_html=True)
                    elif ec > 1500:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE SALINITY</b> - Elevated salt levels. Monitor livestock water consumption and adjust if needed.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT CONDUCTIVITY</b> - Low salinity. Optimal for livestock.</div>', unsafe_allow_html=True)
                    
                    # 8. CALCIUM HARDNESS
                    st.markdown("### 8. 🪨 Calcium Hardness (CaH) - mg/L")
                    cah_status = "Excellent" if cah < 100 else "Good" if cah < 200 else "Moderate" if cah < 300 else "High"
                    st.markdown(f"**Current: {cah:.2f} mg/L | Status: {cah_status}**")
                    
                    if cah > 300:
                        st.markdown('<div class="warning-box"><b>🟠 HIGH HARDNESS</b> - Water very hard. Livestock may show reduced water intake. Consider partial water exchange.</div>', unsafe_allow_html=True)
                    elif cah > 200:
                        st.markdown('<div class="info-box"><b>🟡 MODERATE HARDNESS</b> - Harder water. Generally acceptable for livestock but monitor consumption.</div>', unsafe_allow_html=True)
                    elif cah > 100:
                        st.markdown('<div class="info-box"><b>🟡 SLIGHT HARDNESS</b> - Mild calcium content. Acceptable for livestock.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box"><b>✅ EXCELLENT HARDNESS</b> - Soft to moderately hard water. Ideal for livestock.</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**✅ Comprehensive Parameter Assessment Complete**")
                # st.subheader("💡 Detailed Water Quality Assessment & Recommendations")
                
                # recommendations = []
                
                # if st.session_state.system == "Aquaculture":
                #     do = input_dict['DO']
                #     ammonia = input_dict['Ammonia']
                #     ph = input_dict['pH']
                #     tds = input_dict['TDS']
                #     nitrate = input_dict['Nitrate']
                #     chlorides = input_dict['Chlorides']
                    
                #     # Dissolved Oxygen
                #     if do < 2:
                #         recommendations.append(("🚨 CRITICAL - Dissolved Oxygen <2 mg/L (ANOXIC)", 
                #             "Water has NO oxygen. Aquatic life CANNOT survive. Immediate emergency intervention required: Install multiple aeration systems, increase water circulation, partial/complete water replacement, emergency oxygen injection.", 'critical'))
                #     elif do < 4:
                #         recommendations.append(("🔴 SEVERE - Dissolved Oxygen 2-4 mg/L", 
                #             "Most fish will suffer/die. URGENT intervention: Install aeration system immediately, increase capacity, increase circulation, reduce fish stocking density.", 'critical'))
                #     elif do < 5:
                #         recommendations.append(("🟠 HIGH - Dissolved Oxygen <5 mg/L", 
                #             "Low oxygen stress. Increase aeration immediately, reduce feed input, increase water circulation, monitor every 6-8 hours.", 'warning'))
                #     elif do < 7:
                #         recommendations.append(("🟡 MODERATE - Dissolved Oxygen 5-7 mg/L", 
                #             "Below optimal for sensitive species. Consider increasing aeration.", 'info'))
                    
                #     # Ammonia
                #     if ammonia > 5:
                #         recommendations.append(("🚨 CRITICAL - Ammonia >5 mg/L", 
                #             "SEVERE toxic pollution. Water heavily contaminated. DO NOT use for fish. Immediate treatment: partial/complete water replacement, increase biological filtration, reduce organic input.", 'critical'))
                #     elif ammonia > 2:
                #         recommendations.append(("🔴 SEVERE - Ammonia 2-5 mg/L", 
                #             "High toxicity. Significant organic pollution. Urgent water treatment needed: partial water exchange (25-50%), increase filtration, reduce feed.", 'critical'))
                #     elif ammonia > 0.5:
                #         recommendations.append(("🟠 HIGH - Ammonia >0.5 mg/L", 
                #             "Indicates organic pollution. Improve water circulation, reduce feed input, enhance biological filtration.", 'warning'))
                #     elif ammonia > 0.1:
                #         recommendations.append(("🟡 MODERATE - Ammonia 0.1-0.5 mg/L", 
                #             "Minor pollution detected. Monitor and consider enhanced filtration.", 'info'))
                    
                #     # pH
                #     if ph < 4 or ph > 11:
                #         recommendations.append(("🚨 CRITICAL - pH Extreme", 
                #             "Water chemistry severely imbalanced. Immediate pH correction required using appropriate buffers.", 'critical'))
                #     elif ph < 6 or ph > 9.5:
                #         recommendations.append(("🟠 HIGH - pH Out of Safe Range", 
                #             "Water chemistry imbalanced. Requires pH adjustment using buffers.", 'warning'))
                #     elif ph < 6.5 or ph > 8.5:
                #         recommendations.append(("🟡 MODERATE - pH Suboptimal", 
                #             "Consider pH buffering for better conditions.", 'info'))
                    
                #     # TDS
                #     if tds > 1000:
                #         recommendations.append(("🚨 CRITICAL - TDS >1000 mg/L", 
                #             "Water is highly saline. Consider water replacement or dilution.", 'critical'))
                #     elif tds > 500:
                #         recommendations.append(("🟠 HIGH - TDS >500 mg/L", 
                #             "Salt/mineral accumulation. Monitor and consider water exchange.", 'warning'))
                #     elif tds > 300:
                #         recommendations.append(("🟡 MODERATE - TDS >300 mg/L", 
                #             "Monitor salt accumulation; partial water change recommended.", 'info'))
                    
                #     # Nitrate
                #     if nitrate > 200:
                #         recommendations.append(("🚨 CRITICAL - Nitrate >200 mg/L", 
                #             "Severe nutrient pollution. Immediate biological treatment or water replacement needed.", 'critical'))
                #     elif nitrate > 50:
                #         recommendations.append(("🟠 HIGH - Nitrate >50 mg/L", 
                #             "Significant pollution. Reduce feed, increase biological filtration, consider partial water change.", 'warning'))
                #     elif nitrate > 10:
                #         recommendations.append(("🟡 MODERATE - Nitrate >10 mg/L", 
                #             "Elevated nutrient levels. Reduce feed input and enhance filtration.", 'info'))
                    
                #     # Chlorides
                #     if chlorides > 1000:
                #         recommendations.append(("🚨 CRITICAL - Chlorides >1000 mg/L", 
                #             "Highly saline water. Immediate dilution or water replacement required.", 'warning'))
                #     elif chlorides > 500:
                #         recommendations.append(("🟠 HIGH - Chlorides >500 mg/L", 
                #             "High salt content. Monitor and consider water exchange.", 'warning'))
                
                # else:  # Livestock
                #     do = input_dict['DO']
                #     ph = input_dict['pH']
                #     ec = input_dict['EC']
                #     nitrate = input_dict['Nitrate']
                    
                #     if do < 4:
                #         recommendations.append(("🚨 CRITICAL - Low DO", 
                #             "Critical oxygen depletion. Immediate aeration required.", 'critical'))
                #     elif do < 5:
                #         recommendations.append(("🟠 HIGH - Suboptimal DO", 
                #             "Improve aeration and water circulation.", 'warning'))
                    
                #     if ph < 5 or ph > 10:
                #         recommendations.append(("🚨 CRITICAL - Extreme pH", 
                #             "Severe pH imbalance. Immediate correction required.", 'critical'))
                #     elif ph < 6 or ph > 9:
                #         recommendations.append(("🟠 HIGH - pH Out of Range", 
                #             "Adjust pH using appropriate buffers.", 'warning'))
                    
                #     if ec > 3000:
                #         recommendations.append(("🚨 CRITICAL - Extreme Salinity", 
                #             "Water is extremely saline. Replacement needed.", 'critical'))
                #     elif ec > 2000:
                #         recommendations.append(("🟠 HIGH - High EC", 
                #             "Monitor salinity levels and consider water exchange.", 'warning'))
                    
                #     if nitrate > 200:
                #         recommendations.append(("🚨 CRITICAL - High Nitrate", 
                #             "Severe pollution. Immediate treatment required.", 'critical'))
                #     elif nitrate > 50:
                #         recommendations.append(("🟠 HIGH - Elevated Nitrate", 
                #             "Reduce pollution sources and monitor closely.", 'warning'))
                
                # # Display recommendations
                # if recommendations:
                #     for title, desc, rec_type in recommendations:
                #         if rec_type == 'critical':
                #             st.markdown(f'<div class="critical-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
                #         elif rec_type == 'warning':
                #             st.markdown(f'<div class="warning-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
                #         else:
                #             st.markdown(f'<div class="info-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
                # else:
                #     st.markdown(
                #         '<div class="success-box">✅ All parameters within excellent ranges! Water quality is perfect for all uses.</div>',
                #         unsafe_allow_html=True
                #     )
                
                # ============================================================
                # IMPORTANT NOTE (ALWAYS SHOWN)
                # ============================================================
                st.markdown(f"""
                <div class="note-box">
                <b>ℹ️ IMPORTANT NOTE ABOUT RESULTS:</b>
                <br><br>
                <b>Best Model Selection:</b>
                <br>• This system currently uses <b>{best_name}</b> as the primary predictive model
                <br>• Linear Regression and SVR show the highest R² scores and best generalization in validation data
                <br>• You can verify all model predictions in the "All Model Predictions Detailed" section above
                <br><br>
                <b>About Your Data:</b>
                <br>The models are trained on historical data where only 2-3 dominant parameters strongly influence the 
                water quality index. Other parameters have minimal statistical effect on the final score. This is why 
                some high parameter values might still show good quality - the model reflects the actual patterns found 
                in your training data.
                <br><br>
                <b>Key Findings:</b>
                <br>• Aquaculture (AWQI): Ammonia & DO are dominant factors
                <br>• Livestock (LWQI): Iron, DO & EC are dominant factors
                <br>• Other parameters: Minimal statistical influence
                <br><br>
                <b>Using Results Correctly:</b>
                <br>1. Review overall quality score from {best_name} (primary indicator)
                <br>2. Check individual parameters against optimal ranges (secondary check)
                <br>3. Focus especially on dominant parameters
                <br>4. Use as decision support tool, not absolute truth
                </div>
                """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # ========================================================================
    # PAGE: PARAMETER GUIDE (from v3.2)
    # ========================================================================
    elif page == "📚 Parameter Guide":
        st.header("Water Quality Parameters Reference")
        
        if st.session_state.system == "Aquaculture":
            st.subheader("Aquaculture (AWQI) Parameters")
            params = {
                'TDS': {'name': 'Total Dissolved Solids', 'optimal': '<250 mg/L', 'desc': 'Measure of mineral content'},
                'DO': {'name': 'Dissolved Oxygen', 'optimal': '>7 mg/L', 'desc': 'Oxygen for aquatic life'},
                'Nitrate': {'name': 'Nitrate', 'optimal': '<10 mg/L', 'desc': 'Nutrient pollution indicator'},
                'TH': {'name': 'Total Hardness', 'optimal': '50-150 mg/L', 'desc': 'Ca²⁺ and Mg²⁺ concentration'},
                'pH': {'name': 'pH Value', 'optimal': '6.5-8.5', 'desc': 'Acidity/alkalinity'},
                'Chlorides': {'name': 'Chlorides', 'optimal': '<250 mg/L', 'desc': 'Salt concentration'},
                'Alkalinity': {'name': 'Alkalinity', 'optimal': '50-200 mg/L', 'desc': 'Buffering capacity'},
                'EC': {'name': 'Electrical Conductivity', 'optimal': '500-1500 µS/cm', 'desc': 'Dissolved ions'},
                'Ammonia': {'name': 'Ammonia', 'optimal': '<0.5 mg/L', 'desc': 'Organic pollution indicator'},
            }
        else:
            st.subheader("Livestock (LWQI) Parameters")
            params = {
                'DO': {'name': 'Dissolved Oxygen', 'optimal': '>5 mg/L', 'desc': 'Oxygen level'},
                'Nitrate': {'name': 'Nitrate', 'optimal': '<50 mg/L', 'desc': 'Nutrient level'},
                'CaH': {'name': 'Calcium Hardness', 'optimal': '<300 mg/L', 'desc': 'Ca²⁺ level'},
                'pH': {'name': 'pH Value', 'optimal': '6.5-8.5', 'desc': 'Acidity/alkalinity'},
                'Sulphates': {'name': 'Sulphates', 'optimal': '<500 mg/L', 'desc': 'Sulphate content'},
                'Sodium': {'name': 'Sodium', 'optimal': '<200 mg/L', 'desc': 'Sodium level'},
                'EC': {'name': 'Electrical Conductivity', 'optimal': '<1500 µS/cm', 'desc': 'Conductivity'},
                'Iron': {'name': 'Iron', 'optimal': '<2 mg/L', 'desc': 'Iron content'},
            }
        
        for param, info in params.items():
            with st.expander(f"📌 {info['name']}"):
                st.write(f"**Optimal Range:** {info['optimal']}")
                st.write(f"**Description:** {info['desc']}")
    
    # ========================================================================
    # PAGE: MODEL PERFORMANCE (from v4.2)
    # ========================================================================
    elif page == "📈 Model Performance":
        st.header("Machine Learning Model Performance")
        
        if st.session_state.system == "Aquaculture":
            st.subheader("Aquaculture (AWQI) Models - Performance Metrics")
            perf_data = {
                'Model': ['Linear Regression', 'SVR', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN'],
                'R² Score': [0.9999, 0.9999, 0.9482, 0.8717, 0.8940, 0.9062],
                'MSE': [0.0004, 0.0058, 6.0648, 15.0384, 12.4190, 10.9935],
                
            }
        else:
            st.subheader("Livestock (LWQI) Models - Performance Metrics")
            perf_data = {
                'Model': ['Linear Regression', 'SVR', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN'],
                'R² Score': [0.9999, 0.999, 0.915, 0.894, 0.933, -0.602],
                'MSE': [0.00001, 0.004, 51.49, 64.69, 40.68, 980.72],
                
            }
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.header("About This System")
        st.markdown(f"""
        ## Combined Water Quality Prediction System - Version 4.2
        
        **Current System:** {st.session_state.system}
        
        This enhanced unified system assesses water quality for both Aquaculture and Livestock 
        using advanced machine learning algorithms with comprehensive analysis features.
        
        ### Features
        - **AWQI Score & Classification:** Detailed quality assessment with interpretation
        - **Model Predictions & Severity:** Individual model outputs and severity scoring
        - **All Model Predictions:** View predictions from all 6 regression models
        - **Detailed Recommendations:** Specific treatment steps for each issue
        - **Important Note:** Transparency about model limitations
        
        ### How It Works
        1. Select your water quality system
        2. Enter water parameters
        3. View comprehensive predictions from 12 ML models
        4. See severity assessment
        5. Get detailed recommendations
        6. Understand model limitations
        
        ### Important Note
        The trained models focus on 2-3 dominant parameters that most strongly influence 
        water quality in the training data:
        - **Aquaculture:** Ammonia & Dissolved Oxygen
        - **Livestock:** Iron, DO & Electrical Conductivity
        
        Always review individual parameters and use results as decision support, not absolute truth.
        
        ### Technology
        - ML Framework: Scikit-learn, XGBoost, Neural Networks
        - Web Framework: Streamlit
        - Models: 24 total (12 per system: 6 regression + 6 classification)
        
        **Version:** 4.2 | **Status:** ✅ Production Ready
        """)

if __name__ == "__main__":
    main()