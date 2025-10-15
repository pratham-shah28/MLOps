import joblib
import os
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_training():
    """
    Train the model (Wine dataset + RandomForest, no external CSV)
    """
    # Load the dataset from sklearn
    data = load_wine(as_frame=True)
    X = data.data
    y = data.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=26
    )

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=26)
    model.fit(X_train, y_train)
    model.feature_names = X.columns

    # Persist the trained model
    if not os.path.exists("../model"):
        os.makedirs("../model")
    joblib.dump(model, "../model/model.pkl")

if __name__ == "__main__":
    run_training()