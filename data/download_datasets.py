import os
import subprocess
from dotenv import load_dotenv

def download_kaggle_dataset(dataset_name, download_path="raw"):
    # Load environment variables from .env file
    load_dotenv()
    
    print(f"Downloading {dataset_name} to {download_path}...")
    
    # Ensure the path exists
    os.makedirs(download_path, exist_ok=True)
    
    # Kaggle command to download dataset and unzip
    cmd = [
        "kaggle", "datasets", "download", "-d", dataset_name,
        "-p", download_path, "--unzip"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded {dataset_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_name}. Make sure you have configured your kaggle.json file.")
        print("Details:", e)
    except FileNotFoundError:
        print("Error: The 'kaggle' command was not found.")
        print("Please ensure you have installed the kaggle package (pip install kaggle).")

if __name__ == "__main__":
    # Primary Dataset: Crop Recommendation
    crop_dataset = "atharvaingle/crop-recommendation-dataset"
    
    # Secondary Dataset: India Crop Production
    production_dataset = "aravindpcoder/india-crop-production-state-wise"
    
    # Download paths
    raw_data_dir = os.path.join(os.path.dirname(__file__), "raw")
    
    print("--- Starting Dataset Downloads ---")
    download_kaggle_dataset(crop_dataset, raw_data_dir)
    print("-" * 30)
    download_kaggle_dataset(production_dataset, raw_data_dir)
    print("--- Finished ---")
