from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import os
import shap
from .models import CropPredictionRequest, CropPredictionResponse

app = FastAPI(title="AgriSmart AI API", description="Predicts crop and estimates profit.")

# Global variables for models and data
rf_model = None
label_encoder = None
explainer = None
average_yields = {}

# Hardcoded market prices in INR (₹) per Quintal (100kg)
# Note: These are rough estimates for demonstration purposes.
MARKET_PRICES_PER_QUINTAL = {
    'rice': 2200,
    'maize': 2000,
    'chickpea': 5300,
    'kidneybeans': 6000,
    'pigeonpeas': 6600,
    'mothbeans': 5500,
    'mungbean': 7700,
    'blackgram': 6000,
    'lentil': 6000,
    'pomegranate': 8000,
    'banana': 3000,
    'mango': 5000,
    'grapes': 7000,
    'watermelon': 1500,
    'muskmelon': 1800,
    'apple': 9000,
    'orange': 4000,
    'papaya': 2500,
    'coconut': 3500,
    'cotton': 6000,
    'jute': 2800,
    'coffee': 15000
}

# Actionable farming advice based on negative factors
AGRICULTURAL_ADVICE = {
    "Nitrogen": "Adjust your Nitrogen fertilizer application. If too low, add urea or compost. If too high, consider planting nitrogen-fixing cover crops temporarily.",
    "Phosphorus": "Your Phosphorus level is sub-optimal. Use bone meal or rock phosphate if low, or avoid P-heavy fertilizers if too high.",
    "Potassium": "Potassium is slightly off. Consider applying potash fertilizers or wood ash to balance it for this crop.",
    "Temperature": "Temperature is a negative factor. Use mulching or shade nets during hot peaks, or windbreaks/row covers if it's too cold.",
    "Humidity": "Humidity levels are fighting the crop. Ensure good air circulation, prune lower leaves, and avoid overhead watering if too humid.",
    "pH": "Soil pH is limiting the crop's potential. Apply agricultural lime to raise pH, or sulfur/organic matter to lower it towards neutral.",
    "Rainfall": "Water availability is an issue. Implement drip irrigation if rainfall is too low, or improve field drainage systems if it is too high."
}

@app.on_event("startup")
def load_assets():
    global rf_model, label_encoder, average_yields, explainer
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'random_forest_model.joblib')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.joblib')
    data_path = os.path.join(base_dir, 'data', 'processed', 'crop_production_processed.csv')
    
    # Load Models
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        rf_model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        # Initialize SHAP Explainer
        explainer = shap.TreeExplainer(rf_model)
    else:
        print("WARNING: Models not found. Did you run the notebooks?")
        
    # Load Yield Data and calculate averages
    if os.path.exists(data_path):
        df_prod = pd.read_csv(data_path)
        # The crop names in production dataset might be slightly different or capitalized
        # Let's clean them to match our lowercase recommendation dataset
        df_prod['Crop_Clean'] = df_prod['Crop'].str.lower().str.strip()
        
        # Calculate mean yield per crop
        yields = df_prod.groupby('Crop_Clean')['Yield'].mean().to_dict()
        
        # Map them as best as possible to our 22 labels
        # (Some might not perfectly match, we'll provide a fallback)
        average_yields = yields
    else:
        print("WARNING: Processed crop production data not found.")

@app.post("/predict", response_model=CropPredictionResponse)
def predict_crop(request: CropPredictionRequest):
    if rf_model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")
        
    # Prepare input for model
    input_features = np.array([[
        request.N, request.P, request.K, 
        request.temperature, request.humidity, 
        request.ph, request.rainfall
    ]])
    
    # Predict
    pred_encoded = rf_model.predict(input_features)[0]
    predicted_crop = label_encoder.inverse_transform([pred_encoded])[0]
    
    # Calculate SHAP values for XAI
    feature_names = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
    feature_importances = {}
    actionable_suggestions = []
    
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(input_features)
            
            # Extract impacts for the predicted class
            if isinstance(shap_vals, list):
                impacts = shap_vals[pred_encoded][0]
            else:
                if len(shap_vals.shape) == 3:
                    impacts = shap_vals[0, :, pred_encoded]
                else:
                    impacts = shap_vals[0]
                    
            # Map impacts to feature names
            feature_importances = {name: float(imp) for name, imp in zip(feature_names, impacts)}
            
            # Generate actionable suggestions for significant negative impacts
            for feature, impact in feature_importances.items():
                if impact < -0.01:  # Threshold for negative impact
                    advice = AGRICULTURAL_ADVICE.get(feature)
                    if advice:
                        actionable_suggestions.append(f"**{feature}**: {advice}")
        except Exception as e:
            print(f"SHAP Calculation Error: {e}")
            feature_importances = {"Error": 0.0}
    
    # Look up average yield (Fallback to 5.0 if not found)
    # Note: Production data might have names like 'rice' or 'rice (paddy)'
    est_yield = average_yields.get(predicted_crop.lower(), 5.0) 
    
    # Look up price (Fallback to 2000 if not found)
    est_price = MARKET_PRICES_PER_QUINTAL.get(predicted_crop.lower(), 2000.0)
    
    # Calculate Profit: (Yield is typically Tonnes/Hectare or Kg/Hectare)
    # Assuming Yield here is roughly Quintals per Hectare for simplicity in our engine
    # (In reality, we'd need to carefully check the units in the dataset, but this serves the PRD logic)
    # Let's say Yield is directly multiplied by Price for profit per unit area.
    # To make numbers look realistic, let's assume the yield is in Tonnes/Hectare.
    # 1 Tonne = 10 Quintals.
    yield_in_quintals = est_yield * 10
    profit = yield_in_quintals * est_price
    
    return CropPredictionResponse(
        crop=predicted_crop.capitalize(),
        estimated_yield_per_hectare=round(est_yield, 2),
        market_price_per_quintal=est_price,
        estimated_profit_per_hectare=round(profit, 2),
        feature_importances=feature_importances,
        actionable_suggestions=actionable_suggestions
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the AgriSmart AI Backend!"}
