# AgriSmart AI

AgriSmart AI is a smart agriculture system that predicts the most optimal crop for a given set of soil and weather conditions and estimates the potential profit per hectare. It also includes an Explainable AI (XAI) component using SHAP to help farmers understand the reasoning behind the recommendations.

## Features
- **Crop Recommendation:** Uses a Random Forest classifier to suggest the best crop.
- **Profit Estimation:** Calculates potential profit based on yield and market prices.
- **Explainable AI (XAI):** Visualizes the impact of different factors (N, P, K, etc.) on the AI's decision.
- **Actionable Advice:** Provides practical farming tips based on negative environmental factors.

## Tech Stack
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn, SHAP, Joblib
- **Data:** Kaggle (Crop Recommendation & India Crop Production datasets)

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd "AI based crop prediction"
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory and add your Kaggle API token:
```
KAGGLE_API_TOKEN=your_token_here
```

### 5. Download Data & Train Models
If the models are not included, run the download script and then the Jupyter notebooks in the `notebooks/` directory.

### 6. Run the Application
The application is now self-contained! You can run it with a single command:

```bash
streamlit run frontend/app.py
```

## Hosting on Streamlit Cloud
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app" and select this repository and the `frontend/app.py` file.
4. Your app will be live and shared with the world!


## Project Structure
- `backend/`: FastAPI application and business logic.
- `frontend/`: Streamlit UI.
- `notebooks/`: EDA and model training.
- `data/`: Data storage and download scripts.
- `models/`: Saved ML models.
