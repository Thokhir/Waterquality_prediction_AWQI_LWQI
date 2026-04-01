import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
                  padding: 25px; border-radius: 12px; text-align: center; margin: 10px 0; }
    .info-box { background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 15px; 
                margin: 10px 0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models(system_type):
    system = system_type.lower()
    model_dir = f"models/{system}"
    
    if not os.path.exists(model_dir):
        st.error(f"Models folder not found: {model_dir}")
        return None, None, None
    
    reg_models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('_reg.pkl'):
            name = fname.replace('_reg.pkl', '').replace('_', ' ').title()
            reg_models[name] = joblib.load(os.path.join(model_dir, fname))
    
    scaler = joblib.load(os.path.join(model_dir, 'scaler_regression.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    
    return reg_models, scaler, feature_names

def main():
    st.title("💧 Water Quality Prediction System")
    st.markdown("**Aquaculture (AWQI) & Livestock (LWQI)**")

    system = st.radio("Select System:", ["🐟 Aquaculture (AWQI)", "🐄 Livestock (LWQI)"], horizontal=True)
    system_type = "Aquaculture" if "Aquaculture" in system else "Livestock"

    reg_models, scaler, feature_names = load_models(system_type)
    if not reg_models:
        return

    st.header(f"{system_type} - Water Quality Prediction")

    # ==================== INPUT PARAMETERS ====================
    col1, col2, col3 = st.columns(3)

    if system_type == "Aquaculture":
        with col1:
            st.subheader("Input Parameters")
            tds = st.number_input("TDS (mg/L)", 0.0, 5000.0, 170.0)
            ph = st.number_input("pH", 0.0, 14.0, 7.8, 0.1)
            alkalinity = st.number_input("Alkalinity (mg/L)", 0.0, 2000.0, 70.0)

        with col2:
            do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0, 0.1)
            chlorides = st.number_input("Chlorides (mg/L)", 0.0, 3000.0, 25.0)
            ec = st.number_input("EC (µS/cm)", 0.0, 10000.0, 280.0, 10.0)

        with col3:
            nitrate = st.number_input("Nitrate (mg/L)", 0.0, 2000.0, 0.4, 0.1)
            th = st.number_input("Total Hardness (mg/L)", 0.0, 2000.0, 140.0)
            ammonia = st.number_input("Ammonia (mg/L)", 0.0, 100.0, 0.01, 0.001)

    else:  # Livestock - Fixed Layout as per your screenshot
        with col1:
            st.subheader("Input Parameters")
            do = st.number_input("DO (mg/L)", 0.0, 15.0, 5.0, 0.1)
            ph = st.number_input("pH", 0.0, 14.0, 7.8, 0.1)
            na = st.number_input("Sodium (mg/L)", 0.0, 500.0, 20.0, 0.5)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)   # Aligns with subheader
            nitrate = st.number_input("Nitrate (mg/L)", 0.0, 500.0, 0.5, 0.1)
            cah = st.number_input("Calcium Hardness (mg/L)", 0.0, 500.0, 8.0, 1.0)
            sulphates = st.number_input("Sulphates (mg/L)", 0.0, 500.0, 6.0, 0.1)

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            ec = st.number_input("EC (µS/cm)", 0.0, 2000.0, 300.0, 10.0)
            iron = st.number_input("Iron (mg/L)", 0.0, 100.0, 0.03, 0.1)

    # Time Input (Common for both)
    time_value = st.slider("Time of Day (hours)", 0, 23, 12)
    time_sin = np.sin(2 * np.pi * time_value / 12)
    time_cos = np.cos(2 * np.pi * time_value / 12)

    # Prepare input dictionary
    if system_type == "Aquaculture":
        input_dict = {
            'TDS': tds, 'DO': do, 'Nitrate': nitrate, 'TH': th, 'pH': ph,
            'Chlorides': chlorides, 'Alkalinity': alkalinity, 'EC': ec,
            'Ammonia': ammonia, 'Time_sin': time_sin, 'Time_cos': time_cos
        }
    else:
        input_dict = {
            'DO': do, 'Nitrate': nitrate, 'CaH': cah, 'pH': ph,
            'Sulphates': sulphates, 'Sodium': na, 'EC': ec, 'Iron': iron,
            'Time_sin': time_sin, 'Time_cos': time_cos
        }

    if st.button("🔍 Predict Water Quality", type="primary", use_container_width=True):
        try:
            features = pd.DataFrame([input_dict])[feature_names]
            scaled = scaler.transform(features)

            # Predict using best model (Linear Regression preferred)
            best_model_name = next((name for name in reg_models if "linear" in name.lower()), list(reg_models.keys())[0])
            model = reg_models[best_model_name]
            score = model.predict(scaled)[0]

            st.success(f"**{system_type} Score: {score:.2f}**")

            # Interpretation
            if system_type == "Aquaculture":
                quality = "Excellent" if score < 25 else "Good" if score < 50 else "Moderate" if score < 75 else "Poor"
            else:
                quality = "Good" if score < 40 else "Fair" if score < 80 else "Poor"

            st.info(f"**Water Quality Class: {quality}**")

            st.write(f"**Model Used:** {best_model_name} | **Time:** {time_value}:00")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    main()