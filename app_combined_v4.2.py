"""
Combined Water Quality Prediction System - FINAL MERGED VERSION
Based on v4.0 structure with v3.2 dashboard, model performance, and parameter guide
Version 4.1 - Merged Production Ready
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
# UTILITY FUNCTIONS (from v3.2)
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
        <h1>💧 Combined Water Quality Prediction System 💧</h1>
        <p><i>Aquaculture (AWQI) + Livestock (LWQI) Analysis</i></p>
        <p style="color: #666; font-size: 14px;">Version 4.1 - Merged Unified Quality Assessment Tool</p>
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
    # PAGE: PREDICTION DASHBOARD (from v3.2)
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
                scaled_clf = scalers['classification'].transform(features)
                
                # ============================================================
                # SECTION 1: AWQI SCORE & CLASSIFICATION
                # ============================================================
                st.subheader("🎯 AWQI Score & Classification")
                
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
                            <p style="font-size: 14px; margin-top: 10px;"><b>Action:</b> {interpretation['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # ============================================================
                # SECTION 2: MODEL PREDICTIONS & SEVERITY
                # ============================================================
                st.subheader("📊 Model Predictions & Severity Assessment")
                
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    st.write("**Regression Model Predictions:**")
                    pred_df = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'AWQI Score': list(predictions.values())
                    }).sort_values('AWQI Score', ascending=False)
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
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
                severity_score, critical_issues = get_severity_level(input_dict, st.session_state.system)
                
                st.write("**Overall Severity Assessment:**")
                if severity_score >= 6:
                    st.error(f"🚨 **CRITICAL SEVERITY** - Multiple severe issues detected! (Score: {severity_score}/10)")
                    if critical_issues:
                        for issue in critical_issues:
                            st.error(f"• {issue}")
                elif severity_score >= 4:
                    st.warning(f"🔴 **HIGH SEVERITY** - Significant problems detected (Score: {severity_score}/10)")
                    if critical_issues:
                        for issue in critical_issues:
                            st.warning(f"• {issue}")
                elif severity_score >= 2:
                    st.warning(f"🟡 **MODERATE SEVERITY** - Issues present (Score: {severity_score}/10)")
                elif severity_score >= 1:
                    st.info(f"🟢 **LOW SEVERITY** - Minor issues (Score: {severity_score}/10)")
                else:
                    st.success("✅ **EXCELLENT** - No significant issues detected")
                
                # ============================================================
                # SECTION 3: ALL MODEL PREDICTIONS
                # ============================================================
                st.subheader("🤖 All Model Predictions Detailed")
                
                all_predictions = []
                for name, model in reg_models.items():
                    try:
                        pred = model.predict(scaled_features)[0]
                        all_predictions.append({'Model': name, 'Score': f"{pred:.2f}"})
                    except:
                        pass
                
                if all_predictions:
                    all_pred_df = pd.DataFrame(all_predictions)
                    st.dataframe(all_pred_df, use_container_width=True, hide_index=True)
                
                # ============================================================
                # SECTION 4: DETAILED WATER QUALITY ASSESSMENT & RECOMMENDATIONS
                # ============================================================
                st.subheader("💡 Detailed Water Quality Assessment & Recommendations")
                
                recommendations = []
                
                if st.session_state.system == "Aquaculture":
                    do = input_dict['DO']
                    ammonia = input_dict['Ammonia']
                    ph = input_dict['pH']
                    tds = input_dict['TDS']
                    nitrate = input_dict['Nitrate']
                    chlorides = input_dict['Chlorides']
                    
                    # Dissolved Oxygen
                    if do < 2:
                        recommendations.append(("🚨 CRITICAL - Dissolved Oxygen <2 mg/L (ANOXIC)", 
                            "Water has NO oxygen. Aquatic life CANNOT survive. Immediate emergency intervention required: Install multiple aeration systems, increase water circulation, partial/complete water replacement, emergency oxygen injection.", 'critical'))
                    elif do < 4:
                        recommendations.append(("🔴 SEVERE - Dissolved Oxygen 2-4 mg/L", 
                            "Most fish will suffer/die. URGENT intervention: Install aeration system immediately, increase capacity, increase circulation, reduce fish stocking density.", 'critical'))
                    elif do < 5:
                        recommendations.append(("🟠 HIGH - Dissolved Oxygen <5 mg/L", 
                            "Low oxygen stress. Increase aeration immediately, reduce feed input, increase water circulation, monitor every 6-8 hours.", 'warning'))
                    elif do < 7:
                        recommendations.append(("🟡 MODERATE - Dissolved Oxygen 5-7 mg/L", 
                            "Below optimal for sensitive species. Consider increasing aeration.", 'info'))
                    
                    # Ammonia
                    if ammonia > 5:
                        recommendations.append(("🚨 CRITICAL - Ammonia >5 mg/L", 
                            "SEVERE toxic pollution. Water heavily contaminated. DO NOT use for fish. Immediate treatment: partial/complete water replacement, increase biological filtration, reduce organic input.", 'critical'))
                    elif ammonia > 2:
                        recommendations.append(("🔴 SEVERE - Ammonia 2-5 mg/L", 
                            "High toxicity. Significant organic pollution. Urgent water treatment needed: partial water exchange (25-50%), increase filtration, reduce feed.", 'critical'))
                    elif ammonia > 0.5:
                        recommendations.append(("🟠 HIGH - Ammonia >0.5 mg/L", 
                            "Indicates organic pollution. Improve water circulation, reduce feed input, enhance biological filtration.", 'warning'))
                    elif ammonia > 0.1:
                        recommendations.append(("🟡 MODERATE - Ammonia 0.1-0.5 mg/L", 
                            "Minor pollution detected. Monitor and consider enhanced filtration.", 'info'))
                    
                    # pH
                    if ph < 4 or ph > 11:
                        recommendations.append(("🚨 CRITICAL - pH Extreme", 
                            "Water chemistry severely imbalanced. Immediate pH correction required using appropriate buffers.", 'critical'))
                    elif ph < 6 or ph > 9.5:
                        recommendations.append(("🟠 HIGH - pH Out of Safe Range", 
                            "Water chemistry imbalanced. Requires pH adjustment using buffers.", 'warning'))
                    elif ph < 6.5 or ph > 8.5:
                        recommendations.append(("🟡 MODERATE - pH Suboptimal", 
                            "Consider pH buffering for better conditions.", 'info'))
                    
                    # TDS
                    if tds > 1000:
                        recommendations.append(("🚨 CRITICAL - TDS >1000 mg/L", 
                            "Water is highly saline. Consider water replacement or dilution.", 'critical'))
                    elif tds > 500:
                        recommendations.append(("🟠 HIGH - TDS >500 mg/L", 
                            "Salt/mineral accumulation. Monitor and consider water exchange.", 'warning'))
                    elif tds > 300:
                        recommendations.append(("🟡 MODERATE - TDS >300 mg/L", 
                            "Monitor salt accumulation; partial water change recommended.", 'info'))
                    
                    # Nitrate
                    if nitrate > 200:
                        recommendations.append(("🚨 CRITICAL - Nitrate >200 mg/L", 
                            "Severe nutrient pollution. Immediate biological treatment or water replacement needed.", 'critical'))
                    elif nitrate > 50:
                        recommendations.append(("🟠 HIGH - Nitrate >50 mg/L", 
                            "Significant pollution. Reduce feed, increase biological filtration, consider partial water change.", 'warning'))
                    elif nitrate > 10:
                        recommendations.append(("🟡 MODERATE - Nitrate >10 mg/L", 
                            "Elevated nutrient levels. Reduce feed input and enhance filtration.", 'info'))
                    
                    # Chlorides
                    if chlorides > 1000:
                        recommendations.append(("🚨 CRITICAL - Chlorides >1000 mg/L", 
                            "Highly saline water. Immediate dilution or water replacement required.", 'warning'))
                    elif chlorides > 500:
                        recommendations.append(("🟠 HIGH - Chlorides >500 mg/L", 
                            "High salt content. Monitor and consider water exchange.", 'warning'))
                
                else:  # Livestock
                    do = input_dict['DO']
                    ph = input_dict['pH']
                    ec = input_dict['EC']
                    nitrate = input_dict['Nitrate']
                    
                    if do < 4:
                        recommendations.append(("🚨 CRITICAL - Low DO", 
                            "Critical oxygen depletion. Immediate aeration required.", 'critical'))
                    elif do < 5:
                        recommendations.append(("🟠 HIGH - Suboptimal DO", 
                            "Improve aeration and water circulation.", 'warning'))
                    
                    if ph < 5 or ph > 10:
                        recommendations.append(("🚨 CRITICAL - Extreme pH", 
                            "Severe pH imbalance. Immediate correction required.", 'critical'))
                    elif ph < 6 or ph > 9:
                        recommendations.append(("🟠 HIGH - pH Out of Range", 
                            "Adjust pH using appropriate buffers.", 'warning'))
                    
                    if ec > 3000:
                        recommendations.append(("🚨 CRITICAL - Extreme Salinity", 
                            "Water is extremely saline. Replacement needed.", 'critical'))
                    elif ec > 2000:
                        recommendations.append(("🟠 HIGH - High EC", 
                            "Monitor salinity levels and consider water exchange.", 'warning'))
                    
                    if nitrate > 200:
                        recommendations.append(("🚨 CRITICAL - High Nitrate", 
                            "Severe pollution. Immediate treatment required.", 'critical'))
                    elif nitrate > 50:
                        recommendations.append(("🟠 HIGH - Elevated Nitrate", 
                            "Reduce pollution sources and monitor closely.", 'warning'))
                
                # Display recommendations
                if recommendations:
                    for title, desc, rec_type in recommendations:
                        if rec_type == 'critical':
                            st.markdown(f'<div class="critical-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
                        elif rec_type == 'warning':
                            st.markdown(f'<div class="warning-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="info-box"><b>{title}</b><br>{desc}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="success-box">✅ All parameters within excellent ranges! Water quality is perfect for all uses.</div>',
                        unsafe_allow_html=True
                    )
                
                # ============================================================
                # IMPORTANT NOTE (ALWAYS SHOWN)
                # ============================================================
                st.markdown("""
                <div class="note-box">
                <b>ℹ️ IMPORTANT NOTE ABOUT RESULTS:</b>
                <br><br>
                The models are trained on historical data where only 2-3 dominant parameters strongly influence the 
                water quality index. Other parameters have minimal statistical effect on the final score. This is why 
                some high parameter values might still show good quality - the model reflects the actual patterns found 
                in your training data.
                <br><br>
                <b>Key Findings:</b>
                <br>• Aquaculture: Ammonia & DO are dominant factors
                <br>• Livestock: pH & EC are dominant factors
                <br>• Other parameters: Minimal statistical influence
                <br><br>
                <b>Using Results Correctly:</b>
                <br>1. Review overall quality score (primary indicator)
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
    # PAGE: MODEL PERFORMANCE (from v3.2)
    # ========================================================================
    elif page == "📈 Model Performance":
        st.header("Machine Learning Model Performance")
        
        if st.session_state.system == "Aquaculture":
            st.subheader("Aquaculture (AWQI) Models - Performance Metrics")
            perf_data = {
                'Model': ['Linear Regression', 'SVR', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN'],
                'R² Score': [1.0000, 0.9999, 0.9482, 0.8717, 0.8940, 0.9734],
                'MSE': [0.0000, 0.0058, 6.0648, 15.0384, 12.4190, 3.1206],
                'Status': ['⭐ Perfect', '⭐ Best', 'Excellent', 'Good', 'Good', 'Excellent']
            }
        else:
            st.subheader("Livestock (LWQI) Models - Performance Metrics")
            perf_data = {
                'Model': ['Linear Regression', 'SVR', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN'],
                'R² Score': [0.95, 0.94, 0.92, 0.88, 0.90, 0.91],
                'MSE': [2.5, 3.1, 4.2, 5.8, 4.5, 4.8],
                'Status': ['⭐ Excellent', 'Excellent', 'Excellent', 'Good', 'Good', 'Excellent']
            }
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.header("About This System")
        st.markdown(f"""
        ## Combined Water Quality Prediction System - Version 4.1
        
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
        
        **Version:** 4.1 | **Status:** ✅ Production Ready
        """)

if __name__ == "__main__":
    main()