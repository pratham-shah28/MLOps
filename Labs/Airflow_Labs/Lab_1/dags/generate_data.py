import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define paths
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.rename(columns={"target": "TARGET"}, inplace=True)

# Convert numeric targets to class names (optional)
target_names = {i: name for i, name in enumerate(iris.target_names)}
df["TARGET"] = df["TARGET"].map(target_names)

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save train and test CSVs
train_path = os.path.join(data_dir, "file.csv")
test_path = os.path.join(data_dir, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.drop(columns=["TARGET"]).to_csv(test_path, index=False)

print(f"✅ Created:\n- {train_path}\n- {test_path}")
