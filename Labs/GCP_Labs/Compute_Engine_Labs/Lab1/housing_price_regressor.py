import time
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_path = "HousingData.csv"
data = pd.read_csv(data_path)

# Select features
features = [
    "CRIM", "ZN", "INDUS", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO"
]
target = "MEDV"

# Drop missing
cleaned = data.dropna(subset=features + [target])

# X and y
X = cleaned[features]
y = cleaned[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Training ---
start_train = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
end_train = time.time()

# --- Inference ---
start_pred = time.time()
predictions = model.predict(X_test)
end_pred = time.time()

# --- Metrics ---
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2_train = r2_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, predictions)

# --- Per-sample inference time ---
inference_time_per_sample = (end_pred - start_pred) / len(X_test)

# --- Save Model ---
with open("linear_reg_model.pkl", "wb") as f:
    pickle.dump(model, f)

model_size = os.path.getsize("linear_reg_model.pkl") / 1024  # KB

# --- Logs ---
print("===== Model Training & Evaluation =====")
print(f"Training Time: {end_train - start_train:.4f} seconds")
print(f"Inference Time: {end_pred - start_pred:.4f} seconds")
print(f"Inference Time per Sample: {inference_time_per_sample:.8f} seconds")

print(f"\nMean Squared Error: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test):  {r2_test:.4f}")

print(f"\nModel Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

print(f"\nModel saved as linear_reg_model.pkl ({model_size:.2f} KB)")
