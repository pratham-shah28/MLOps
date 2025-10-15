import numpy as np
import joblib
import os
from train import run_training

# Ensure model is available (train if missing)
if not os.path.exists("../model/model.pkl"):
    os.makedirs("../model", exist_ok=True)
    run_training()

# Load the trained model
model = joblib.load("../model/model.pkl")

def predict_wine(features):
    """
    Predict wine class based on feature inputs.
    Expects a list or array matching the feature order of the wine dataset.
    """
    input_data = np.array([features])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    print("Model loaded successfully.")
    
    # Example input — 13 features of wine dataset
    example = [13.2, 2.77, 2.51, 18.5, 98, 2.2, 1.45, 0.42, 1.4, 4.0, 1.0, 3.08, 820]
    result = predict_wine(example)
    print("Predicted wine class:", result)
