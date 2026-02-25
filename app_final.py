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
# ROBUST MODEL LOADING
# ============================================================================

@st.cache_resource
def load_system(system_type):
    try:
        system_lower = system_type.lower()
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
            return None, None, None, None

        reg_models = {}
        clf_models = {}

        files_in_dir = os.listdir(model_dir)

        for fname in files_in_dir:
            if '_reg.pkl' in fname and 'scaler' not in fname:
                name = fname.replace('_reg.pkl', '').replace('_', ' ').title()
                reg_models[name] = joblib.load(os.path.join(model_dir, fname))

        for fname in files_in_dir:
            if '_clf.pkl' in fname and 'scaler' not in fname:
                name = fname.replace('_clf.pkl', '').replace('_', ' ').title()
                clf_models[name] = joblib.load(os.path.join(model_dir, fname))

        scalers = {
            'regression': joblib.load(os.path.join(model_dir, 'scaler_regression.pkl')),
            'classification': joblib.load(os.path.join(model_dir, 'scaler_classification.pkl'))
        }

        encoders = {
            'label_encoder': joblib.load(os.path.join(model_dir, 'label_encoder.pkl')),
            'feature_names': joblib.load(os.path.join(model_dir, 'feature_names.pkl')),
            'class_names': joblib.load(os.path.join(model_dir, 'class_names.pkl'))
        }

        return reg_models, clf_models, scalers, encoders

    except:
        return None, None, None, None

# ============================================================================
# INTERPRETATION FUNCTION (UNCHANGED)
# ============================================================================

def get_quality_interpretation(value, system_type):

    if system_type == "Aquaculture":
        if value < 25:
            return {'class': 'Excellent','description': 'Excellent water quality','color': '#28a745','emoji': '✅','action': 'No action needed.'}
        elif value < 50:
            return {'class': 'Good','description': 'Good water quality','color': '#17a2b8','emoji': '👍','action': 'Monitor regularly.'}
        elif value < 75:
            return {'class': 'Moderate','description': 'Moderate water quality','color': '#ffc107','emoji': '⚠️','action': 'Improvement recommended.'}
        else:
            return {'class': 'Poor','description': 'Poor water quality','color': '#dc3545','emoji': '🚫','action': 'Immediate action required.'}
    else:
        if value < 40:
            return {'class': 'Good','description': 'Good water quality','color': '#28a745','emoji': '✅','action': 'No action needed.'}
        elif value < 80:
            return {'class': 'Fair','description': 'Fair water quality','color': '#ffc107','emoji': '⚠️','action': 'Monitor closely.'}
        else:
            return {'class': 'Poor','description': 'Poor water quality','color': '#dc3545','emoji': '🚫','action': 'Treatment required.'}

def prepare_features(input_dict, feature_names):
    features = pd.DataFrame([input_dict])
    return features[feature_names]

# ============================================================================
# MAIN APP
# ============================================================================

def main():

    st.title("💧 Combined Water Quality Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🐟 Aquaculture (AWQI)"):
            st.session_state.system = "Aquaculture"

    with col2:
        if st.button("🐄 Livestock (LWQI)"):
            st.session_state.system = "Livestock"

    if "system" not in st.session_state:
        st.session_state.system = "Aquaculture"

    reg_models, clf_models, scalers, encoders = load_system(st.session_state.system)

    if not reg_models:
        st.error("Models not found")
        return

    feature_names = encoders['feature_names']

    input_dict = {}
    for f in feature_names:
        input_dict[f] = st.number_input(f, value=0.0)

    if st.button("🔍 Predict Water Quality"):

        features = prepare_features(input_dict, feature_names)
        scaled_features = scalers['regression'].transform(features)
        scaled_clf = scalers['classification'].transform(features)

        predictions = {}

        for name, model in reg_models.items():
            try:
                pred = model.predict(scaled_features)[0]
                predictions[name] = pred
            except:
                pass

        # ============================================================
        # FIXED MODEL SELECTION LOGIC
        # ============================================================

        if predictions:

            preferred_models = ["Linear Regression", "SVR"]

            best_name = None

            for pref in preferred_models:
                for model_name in predictions.keys():
                    if pref.lower() in model_name.lower():
                        best_name = model_name
                        break
                if best_name:
                    break

            if not best_name:
                best_name = list(predictions.keys())[0]

            quality_score = predictions[best_name]

            st.subheader(f"🎯 {st.session_state.system} Score & Classification")

            interpretation = get_quality_interpretation(
                quality_score,
                st.session_state.system
            )

            colA, colB = st.columns(2)

            with colA:
                st.markdown(f"""
                <div class="metric-box">
                    <h2>{quality_score:.2f}</h2>
                    <p>{interpretation['class']}</p>
                </div>
                """, unsafe_allow_html=True)

            with colB:
                st.markdown(f"""
                <div style="background-color:{interpretation['color']};
                            color:white;padding:20px;border-radius:10px;">
                    <h3>{interpretation['emoji']} {interpretation['class']}</h3>
                    <p>{interpretation['description']}</p>
                    <p><b>Action:</b> {interpretation['action']}</p>
                </div>
                """, unsafe_allow_html=True)

        # Show all regression predictions (UNCHANGED)
        st.subheader("📊 All Regression Model Predictions")

        pred_df = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Score': list(predictions.values())
        })

        st.dataframe(pred_df, use_container_width=True)

        # Classification (UNCHANGED)
        if clf_models:
            best_clf = list(clf_models.values())[0]
            class_pred = best_clf.predict(scaled_clf)[0]
            class_name = encoders['class_names'][class_pred]
            st.metric("Predicted Class", class_name)


if __name__ == "__main__":
    main()
