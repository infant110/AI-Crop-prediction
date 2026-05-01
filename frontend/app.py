import streamlit as st
import requests
import json

# Must be the first Streamlit command
st.set_page_config(
    page_title="AgriSmart AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism UI
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #13151a 0%, #1a231f 100%);
        color: #ffffff;
    }
    
    /* Hide top header */
    header {visibility: hidden;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] h1 {
        color: #4ade80;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(74, 222, 128, 0.2);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .profit-value {
        background: -webkit-linear-gradient(45deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
        color: #111827 !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4);
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
    }

    /* Input sliders */
    .stSlider > div > div > div > div {
        background-color: #4ade80 !important;
    }

</style>
""", unsafe_allow_html=True)

# Main App Layout
st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 4rem; font-weight: 800; line-height: 1.1; margin-bottom: 20px;">
            Farm <span style="color: #4ade80;">Smarter.</span>
        </h1>
        <p style="color: #94a3b8; font-size: 1.2rem; line-height: 1.6; max-width: 800px; margin: 0 auto;">
            Enter your soil nutrients and weather conditions in the sidebar, and let our AI determine the most profitable crop for your exact location.
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.title("Field Parameters")
st.sidebar.markdown("Adjust the sliders to match your field conditions.")

n_val = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
p_val = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
k_val = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temp_val = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 25.0, 0.1)
hum_val = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0, 0.1)
ph_val = st.sidebar.slider("pH Level", 3.0, 10.0, 6.5, 0.1)
rain_val = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 0.1)

# API Prediction Call
if st.sidebar.button("Predict Optimal Crop"):
    
    with st.spinner("AI is analyzing your soil data..."):
        try:
            # Prepare payload
            payload = {
                "N": n_val,
                "P": p_val,
                "K": k_val,
                "temperature": temp_val,
                "humidity": hum_val,
                "ph": ph_val,
                "rainfall": rain_val
            }
            
            # Send to FastAPI Backend
            # Note: Ensure the backend is running on port 8000
            response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 40px 0;'>", unsafe_allow_html=True)
                
                # Results Grid - Top Row
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.markdown(f"""
                        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                            <div class="metric-title">AI Recommendation</div>
                            <div class="metric-value">{data['crop']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown(f"""
                        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                            <div class="metric-title">Est. Yield & Price</div>
                            <div style="font-size: 2rem; font-weight: 700; color: #e2e8f0; margin-bottom: 5px;">
                                {data['estimated_yield_per_hectare']} <span style="font-size: 1rem; color: #94a3b8;">T/Ha</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #94a3b8;">
                                @ ₹{data['market_price_per_quintal']}/Q
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with res_col3:
                    st.markdown(f"""
                        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; background: rgba(251, 191, 36, 0.05);">
                            <div class="metric-title">Estimated Profit</div>
                            <div class="metric-value profit-value" style="font-size: 2.5rem;">₹{data['estimated_profit_per_hectare']:,}</div>
                            <div style="font-size: 1rem; color: #94a3b8; margin-top: 5px;">per Hectare</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                # XAI Section - Bottom Row
                if 'feature_importances' in data and data['feature_importances']:
                    st.markdown("<br>", unsafe_allow_html=True)
                    xai_col, advice_col = st.columns([3, 2])
                    
                    with xai_col:
                        st.markdown(f"""
                        <div style="margin-bottom: 20px;">
                            <h3 style="color: #4ade80; margin-bottom: 10px;">Understand the AI Decision</h3>
                            <p style="color: #cbd5e1; font-size: 0.95rem;">
                                This chart shows how each factor impacted the decision to recommend <b>{data['crop']}</b>. 
                                Positive values pushed <b>towards</b> this crop, negative values pushed <b>away</b>.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        import pandas as pd
                        import_data = data['feature_importances']
                        if "Error" not in import_data:
                            df_imp = pd.DataFrame(list(import_data.items()), columns=['Factor', 'Impact'])
                            df_imp['Abs_Impact'] = df_imp['Impact'].abs()
                            df_imp = df_imp.sort_values(by='Abs_Impact', ascending=True).drop('Abs_Impact', axis=1)
                            st.bar_chart(df_imp.set_index('Factor'), use_container_width=True)
                            
                    with advice_col:
                        if 'actionable_suggestions' in data and data['actionable_suggestions']:
                            st.markdown(f"""
                            <div style="margin-bottom: 20px;">
                                <h3 style="color: #fbbf24; margin-bottom: 10px;">Actionable Advice</h3>
                                <p style="color: #cbd5e1; font-size: 0.95rem;">
                                    Based on the negative factors identified, here is how you can improve your field for <b>{data['crop']}</b>:
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for suggestion in data['actionable_suggestions']:
                                st.warning(suggestion)
                    
            else:
                st.error(f"Backend Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not reach the backend. Is the FastAPI server running on port 8000?")
            st.info("Run `uvicorn backend.main:app --reload` in your terminal.")

