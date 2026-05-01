import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import shap

# --- Constants & Advice ---
MARKET_PRICES_PER_QUINTAL = {
    'rice': 2200, 'maize': 2000, 'chickpea': 5300, 'kidneybeans': 6000,
    'pigeonpeas': 6600, 'mothbeans': 5500, 'mungbean': 7700, 'blackgram': 6000,
    'lentil': 6000, 'pomegranate': 8000, 'banana': 3000, 'mango': 5000,
    'grapes': 7000, 'watermelon': 1500, 'muskmelon': 1800, 'apple': 9000,
    'orange': 4000, 'papaya': 2500, 'coconut': 3500, 'cotton': 6000,
    'jute': 2800, 'coffee': 15000
}

AGRICULTURAL_ADVICE = {
    "Nitrogen": "Adjust your Nitrogen fertilizer application. If too low, add urea or compost. If too high, consider planting nitrogen-fixing cover crops temporarily.",
    "Phosphorus": "Your Phosphorus level is sub-optimal. Use bone meal or rock phosphate if low, or avoid P-heavy fertilizers if too high.",
    "Potassium": "Potassium is slightly off. Consider applying potash fertilizers or wood ash to balance it for this crop.",
    "Temperature": "Temperature is a negative factor. Use mulching or shade nets during hot peaks, or windbreaks/row covers if it's too cold.",
    "Humidity": "Humidity levels are fighting the crop. Ensure good air circulation, prune lower leaves, and avoid overhead watering if too humid.",
    "pH": "Soil pH is limiting the crop's potential. Apply agricultural lime to raise pH, or sulfur/organic matter to lower it towards neutral.",
    "Rainfall": "Water availability is an issue. Implement drip irrigation if rainfall is too low, or improve field drainage systems if it is too high."
}

# --- AI Logic (Merged from Backend) ---
@st.cache_resource
def load_model_assets():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'random_forest_model.joblib')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.joblib')
    data_path = os.path.join(base_dir, 'data', 'processed', 'crop_production_processed.csv')
    
    rf_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    explainer = shap.TreeExplainer(rf_model)
    
    # Load Yield Data
    df_prod = pd.read_csv(data_path)
    df_prod['Crop_Clean'] = df_prod['Crop'].str.lower().str.strip()
    average_yields = df_prod.groupby('Crop_Clean')['Yield'].mean().to_dict()
    
    return rf_model, label_encoder, explainer, average_yields

def get_prediction(n, p, k, temp, hum, ph, rain):
    rf_model, label_encoder, explainer, average_yields = load_model_assets()
    
    input_features = np.array([[n, p, k, temp, hum, ph, rain]])
    
    # Predict
    pred_encoded = rf_model.predict(input_features)[0]
    predicted_crop = label_encoder.inverse_transform([pred_encoded])[0]
    
    # Calculate SHAP values
    feature_names = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
    shap_vals = explainer.shap_values(input_features)
    
    if isinstance(shap_vals, list):
        impacts = shap_vals[pred_encoded][0]
    else:
        impacts = shap_vals[0, :, pred_encoded] if len(shap_vals.shape) == 3 else shap_vals[0]
        
    feature_importances = {name: float(imp) for name, imp in zip(feature_names, impacts)}
    
    # Suggestions
    suggestions = [f"**{f}**: {AGRICULTURAL_ADVICE[f]}" for f, imp in feature_importances.items() if imp < -0.01]
    
    # Economics
    est_yield = average_yields.get(predicted_crop.lower(), 5.0)
    est_price = MARKET_PRICES_PER_QUINTAL.get(predicted_crop.lower(), 2000.0)
    profit = (est_yield * 10) * est_price
    
    return {
        "crop": predicted_crop.capitalize(),
        "yield": round(est_yield, 2),
        "price": est_price,
        "profit": round(profit, 2),
        "importances": feature_importances,
        "suggestions": suggestions
    }

# --- Streamlit UI ---
st.set_page_config(page_title="AgriSmart AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #13151a 0%, #1a231f 100%); color: #ffffff; }
    header {visibility: hidden;}
    [data-testid="stSidebar"] { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .glass-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(16px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 30px; margin-bottom: 20px; text-align: center; }
    .metric-title { color: #94a3b8; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
    .metric-value { color: #ffffff; font-size: 3rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .profit-value { background: -webkit-linear-gradient(45deg, #fbbf24, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%); color: #111827 !important; border-radius: 12px; width: 100%; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 4rem; font-weight: 800; line-height: 1.1; margin-bottom: 20px;">Farm <span style="color: #4ade80;">Smarter.</span></h1>
        <p style="color: #94a3b8; font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
            Enter your soil nutrients and weather conditions in the sidebar for an AI-powered crop recommendation and profit estimation.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Field Parameters")
n_val = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
p_val = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
k_val = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temp_val = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 25.0, 0.1)
hum_val = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0, 0.1)
ph_val = st.sidebar.slider("pH Level", 3.0, 10.0, 6.5, 0.1)
rain_val = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 0.1)

if st.sidebar.button("Predict Optimal Crop"):
    with st.spinner("AI is analyzing your soil data..."):
        data = get_prediction(n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 40px 0;'>", unsafe_allow_html=True)
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(f'<div class="glass-card"><div class="metric-title">AI Recommendation</div><div class="metric-value">{data["crop"]}</div></div>', unsafe_allow_html=True)
        with res_col2:
            st.markdown(f'<div class="glass-card"><div class="metric-title">Est. Yield & Price</div><div style="font-size: 2rem; font-weight: 700; color: #e2e8f0;">{data["yield"]} T/Ha</div><div style="color: #94a3b8;">@ ₹{data["price"]}/Q</div></div>', unsafe_allow_html=True)
        with res_col3:
            st.markdown(f'<div class="glass-card" style="background: rgba(251, 191, 36, 0.05);"><div class="metric-title">Estimated Profit</div><div class="metric-value profit-value" style="font-size: 2.5rem;">₹{data["profit"]:,}</div><div style="color: #94a3b8;">per Hectare</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        xai_col, advice_col = st.columns([3, 2])
        
        with xai_col:
            st.markdown(f'<h3 style="color: #4ade80;">Understand the AI Decision</h3><p style="color: #cbd5e1; font-size: 0.95rem;">How each factor impacted the decision to recommend <b>{data["crop"]}</b>.</p>', unsafe_allow_html=True)
            df_imp = pd.DataFrame(list(data["importances"].items()), columns=['Factor', 'Impact'])
            df_imp = df_imp.sort_values(by='Impact', ascending=True)
            st.bar_chart(df_imp.set_index('Factor'), use_container_width=True)
            
        with advice_col:
            if data["suggestions"]:
                st.markdown(f'<h3 style="color: #fbbf24;">Actionable Advice</h3><p style="color: #cbd5e1; font-size: 0.95rem;">Steps to improve your field for <b>{data["crop"]}</b>:</p>', unsafe_allow_html=True)
                for s in data["suggestions"]:
                    st.warning(s)
