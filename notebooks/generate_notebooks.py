import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Phase 2: Exploratory Data Analysis & Preprocessing\nIn this notebook, we will explore the `Crop Recommendation` and `Crop Production` datasets, handle any missing values, engineer the `Yield` feature, and prepare the data for Phase 3 (Model Building)."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\nfrom IPython.display import display\n\n# Set plotting style\nsns.set_theme(style='whitegrid')"),
        nbf.v4.new_markdown_cell("## 1. Load the Datasets"),
        nbf.v4.new_code_cell("df_crop = pd.read_csv('../data/raw/Crop_recommendation.csv')\ndf_prod = pd.read_csv('../data/raw/crop_production.csv')"),
        nbf.v4.new_markdown_cell("## 2. Explore Crop Recommendation Data\nThis dataset contains the optimal soil and weather conditions for different crops."),
        nbf.v4.new_code_cell("display(df_crop.head())\ndisplay(df_crop.info())\ndisplay(df_crop.describe())"),
        nbf.v4.new_code_cell("# Check for missing values\ndf_crop.isnull().sum()"),
        nbf.v4.new_code_cell("# Let's look at the distribution of the target variable (Crop Label)\nplt.figure(figsize=(15, 6))\nsns.countplot(data=df_crop, x='label')\nplt.xticks(rotation=90)\nplt.title('Distribution of Crops')\nplt.show()"),
        nbf.v4.new_code_cell("# Correlation Matrix for numerical features\nplt.figure(figsize=(10, 8))\nnumeric_cols = df_crop.select_dtypes(include=[np.number]).columns\nsns.heatmap(df_crop[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Correlation Matrix (Crop Recommendation)')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 3. Explore & Preprocess Crop Production Data\nThis dataset contains historical production records. We need to handle missing values and calculate **Yield** (Production / Area)."),
        nbf.v4.new_code_cell("display(df_prod.head())\ndisplay(df_prod.info())"),
        nbf.v4.new_code_cell("# Check for missing values\ndf_prod.isnull().sum()"),
        nbf.v4.new_markdown_cell("We have some missing values in the `Production` column. Since Production is critical for calculating Yield, we will drop rows with missing production data."),
        nbf.v4.new_code_cell("df_prod.dropna(subset=['Production'], inplace=True)\nprint('Null values after dropping:')\nprint(df_prod.isnull().sum())"),
        nbf.v4.new_code_cell("# Feature Engineering: Calculate Yield\ndf_prod['Yield'] = df_prod['Production'] / df_prod['Area']\ndisplay(df_prod.head())"),
        nbf.v4.new_code_cell("# Average yield per crop\nplt.figure(figsize=(15, 6))\nyield_per_crop = df_prod.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(30)\nsns.barplot(x=yield_per_crop.index, y=yield_per_crop.values)\nplt.xticks(rotation=90)\nplt.title('Top 30 Crops by Average Yield')\nplt.ylabel('Average Yield')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 4. Save Processed Data\nNow we save the cleaned datasets into `data/processed/` for model training and the profit engine."),
        nbf.v4.new_code_cell("os.makedirs('../data/processed', exist_ok=True)\n\ndf_crop.to_csv('../data/processed/Crop_recommendation_processed.csv', index=False)\ndf_prod.to_csv('../data/processed/crop_production_processed.csv', index=False)\nprint('Processed datasets saved successfully!')")
    ]
    
    nb['cells'] = cells
    
    with open('notebooks/01_EDA_and_Preprocessing.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
def create_model_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Phase 3: Model Training\nIn this notebook, we will train a Random Forest and an XGBoost model on the Crop Recommendation dataset. We will evaluate both and potentially build an ensemble if needed."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nimport xgboost as xgb\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nfrom sklearn.preprocessing import LabelEncoder\nimport joblib\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\nfrom IPython.display import display"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/Crop_recommendation_processed.csv')\ndisplay(df.head())"),
        nbf.v4.new_markdown_cell("## 2. Prepare Training and Testing Sets"),
        nbf.v4.new_code_cell("X = df.drop('label', axis=1)\ny = df['label']\n\n# XGBoost requires target labels to be encoded as integers\nlabel_encoder = LabelEncoder()\ny_encoded = label_encoder.fit_transform(y)\n\n# Save the label encoder classes so we can decode predictions later\nnp.save('../models/label_classes.npy', label_encoder.classes_)\n\nX_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)\nprint(f'Training shape: {X_train.shape}')\nprint(f'Testing shape: {X_test.shape}')"),
        nbf.v4.new_markdown_cell("## 3. Train Random Forest"),
        nbf.v4.new_code_cell("rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\nrf_model.fit(X_train, y_train)\n\nrf_preds = rf_model.predict(X_test)\nrf_acc = accuracy_score(y_test, rf_preds)\nprint(f'Random Forest Accuracy: {rf_acc * 100:.2f}%')"),
        nbf.v4.new_markdown_cell("## 4. Train XGBoost"),
        nbf.v4.new_code_cell("xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)\nxgb_model.fit(X_train, y_train)\n\nxgb_preds = xgb_model.predict(X_test)\nxgb_acc = accuracy_score(y_test, xgb_preds)\nprint(f'XGBoost Accuracy: {xgb_acc * 100:.2f}%')"),
        nbf.v4.new_markdown_cell("## 5. Evaluation and Comparison"),
        nbf.v4.new_code_cell("print(\"Random Forest Classification Report:\")\nprint(classification_report(y_test, rf_preds, target_names=label_encoder.classes_))\n\nprint(\"\\n========================================\\n\")\n\nprint(\"XGBoost Classification Report:\")\nprint(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_))"),
        nbf.v4.new_code_cell("def plot_confusion_matrix(y_true, y_pred, title):\n    cm = confusion_matrix(y_true, y_pred)\n    plt.figure(figsize=(12, 10))\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)\n    plt.title(title)\n    plt.ylabel('True Label')\n    plt.xlabel('Predicted Label')\n    plt.show()\n\nplot_confusion_matrix(y_test, rf_preds, 'Random Forest Confusion Matrix')\nplot_confusion_matrix(y_test, xgb_preds, 'XGBoost Confusion Matrix')"),
        nbf.v4.new_markdown_cell("## 6. Save Best Model\nBoth models typically achieve >99% accuracy on this dataset. We will save the Random Forest model for simplicity, or an ensemble voting classifier if we want the best of both worlds. Here we save the Random Forest model."),
        nbf.v4.new_code_cell("os.makedirs('../models', exist_ok=True)\njoblib.dump(rf_model, '../models/random_forest_model.joblib')\njoblib.dump(xgb_model, '../models/xgboost_model.joblib')\njoblib.dump(label_encoder, '../models/label_encoder.joblib')\nprint('Models and Label Encoder saved successfully to /models directory!')")
    ]
    
    nb['cells'] = cells
    
    with open('notebooks/02_Model_Training.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_eda_notebook()
    create_model_notebook()
    print("Notebooks created successfully!")
