import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TARGET_COL = "TARGET"  


# -Helpers
def _paths() -> Dict[str, str]:
    base = os.path.dirname(__file__)
    project = os.path.abspath(os.path.join(base, ".."))
    return {
        "data_dir": os.path.join(project, "data"),
        "model_dir": os.path.join(project, "model"),
        "artifacts_dir": os.path.join(project, "artifacts"),
    }


def _ensure_dirs() -> Dict[str, str]:
    p = _paths()
    os.makedirs(p["model_dir"], exist_ok=True)
    os.makedirs(p["artifacts_dir"], exist_ok=True)
    return p


def _require_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------- Task 1 ----------
def load_training_data() -> Dict[str, str]:
    """
    Locate the training CSV. Return JSON-serializable dict for XCom.
    """
    p = _ensure_dirs()
    train_csv = os.path.join(p["data_dir"], "file.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found at {train_csv}")
    return {"train_csv": train_csv}


# ---------- Task 2 ----------
def preprocess_training_data(train_info: Dict[str, str]) -> Dict[str, str]:
    """
    Fit (imputer -> scaler) on training data, encode TARGET if non-numeric,
    save artifacts to /model and matrices to /artifacts.
    Returns paths needed by later tasks (JSON-serializable).
    """
    p = _ensure_dirs()
    train_csv = train_info["train_csv"]
    df = pd.read_csv(train_csv)

    _require_cols(df, [TARGET_COL])
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    if not feature_cols:
        raise ValueError("No feature columns detected (columns other than TARGET).")

    X = df[feature_cols]
    y_raw = df[TARGET_COL]

    # Encode label 
    le: Optional[LabelEncoder] = None
    if y_raw.dtype.kind in "ifu":
        y = y_raw.to_numpy()
    else:
        le = LabelEncoder().fit(y_raw.astype(str))
        y = le.transform(y_raw.astype(str))

    # Numeric preprocessing pipeline
    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
    )
    pre = ColumnTransformer([("num", num_pipe, feature_cols)], remainder="drop")
    X_proc = pre.fit_transform(X)

    # Persisting artifacts
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    imputer = pre.named_transformers_["num"].named_steps["imputer"]

    scaler_path = os.path.join(p["model_dir"], "minmax_scaler.pkl")
    imputer_path = os.path.join(p["model_dir"], "imputer.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)

    label_encoder_path = None
    if le is not None:
        label_encoder_path = os.path.join(p["model_dir"], "label_encoder.pkl")
        with open(label_encoder_path, "wb") as f:
            pickle.dump(le, f)

    features_path = os.path.join(p["artifacts_dir"], "feature_columns.txt")
    with open(features_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    Xn_path = os.path.join(p["artifacts_dir"], "X_train.npy")
    yn_path = os.path.join(p["artifacts_dir"], "y_train.npy")
    np.save(Xn_path, X_proc)
    np.save(yn_path, y)

    return {
        "X_path": Xn_path,
        "y_path": yn_path,
        "scaler_path": scaler_path,
        "imputer_path": imputer_path,
        "label_encoder_path": label_encoder_path or "",
        "feature_cols_path": features_path,
    }


# ---------- Task 3 ----------
def train_and_save_classifier(artifacts: Dict[str, str], filename: str) -> Dict[str, Any]:
    """
    Train RandomForest on preprocessed matrices and save model to /model/<filename>.
    Returns model_path and a quick train accuracy.
    """
    p = _ensure_dirs()
    X = np.load(artifacts["X_path"])
    y = np.load(artifacts["y_path"])

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    model_path = os.path.join(p["model_dir"], filename)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    acc = float(accuracy_score(y, clf.predict(X)))

    # Also drop a tiny metrics file
    metrics_path = os.path.join(p["artifacts_dir"], "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"train_accuracy": acc}, f)

    return {"model_path": model_path, "train_accuracy": acc}


# ---------- Task 4 ----------
def predict_with_saved_model(model_info: Dict[str, Any], artifacts: Dict[str, str]) -> Dict[str, Any]:
    """
    Load saved model + preprocessing artifacts; transform data/test.csv; save predictions CSV.
    Returns predictions_csv and the first prediction for quick inspection.
    """
    p = _ensure_dirs()
    test_csv = os.path.join(p["data_dir"], "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found at {test_csv}")

    # Load artifacts
    with open(model_info["model_path"], "rb") as f:
        clf = pickle.load(f)
    with open(artifacts["imputer_path"], "rb") as f:
        imputer = pickle.load(f)
    with open(artifacts["scaler_path"], "rb") as f:
        scaler = pickle.load(f)

    with open(artifacts["feature_cols_path"], "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    test_df = pd.read_csv(test_csv)
    _require_cols(test_df, feature_cols)

    Xt = test_df[feature_cols].to_numpy()
    Xt_imp = imputer.transform(Xt)
    Xt_scaled = scaler.transform(Xt_imp)

    preds = clf.predict(Xt_scaled)

    # Inversing label encoding if available
    le_path = artifacts.get("label_encoder_path") or ""
    if le_path and os.path.exists(le_path):
        with open(le_path, "rb") as f:
            le: LabelEncoder = pickle.load(f)
        try:
            preds = le.inverse_transform(preds.astype(int))
        except Exception:
            pass

    preds_csv = os.path.join(p["artifacts_dir"], "predictions.csv")
    pd.DataFrame({"prediction": preds}).to_csv(preds_csv, index=False)

    first = preds[0] if len(preds) else None
    return {"predictions_csv": preds_csv, "first_prediction": first}
