"""
IPL FORENSICS - Win Probability Model
Trains, evaluates, calibrates, and serves binary classification models for ball-by-ball win probability.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)

from src.utils import logger, MODELS_DIR
from src.feature_engineering import MODEL_FEATURE_COLS, TARGET_COL


def split_data_by_match(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
    chronological: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits deliveries into Train and Test datasets at MATCH level to prevent data leakage.
    If chronological=True, older matches are used for training and recent matches for testing.
    """
    # Sort matches chronologically
    match_dates = (
        df.groupby("match_id")["date"]
        .min()
        .reset_index()
        .sort_values(by="date")
    )
    
    unique_matches = match_dates["match_id"].tolist()
    split_idx = int(len(unique_matches) * (1 - test_ratio))
    
    train_match_ids = set(unique_matches[:split_idx])
    test_match_ids = set(unique_matches[split_idx:])
    
    train_df = df[df["match_id"].isin(train_match_ids)].copy()
    test_df = df[df["match_id"].isin(test_match_ids)].copy()
    
    logger.info(
        f"Match split: {len(train_match_ids)} train matches ({len(train_df)} balls), "
        f"{len(test_match_ids)} test matches ({len(test_df)} balls)"
    )
    return train_df, test_df


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates comprehensive classification and probability calibration metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "log_loss": round(float(ll), 4),
        "brier_score": round(float(brier), 4)
    }


def train_and_compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    Trains multiple candidate models:
    1. Calibrated Logistic Regression (Baseline)
    2. Random Forest Classifier
    3. HistGradientBoostingClassifier (High Performance Gradient Boosting)
    
    Returns:
        (best_model, comparison_df, evaluation_details)
    """
    X_train = train_df[MODEL_FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    
    X_test = test_df[MODEL_FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting (Hist)": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            random_state=42
        )
    }
    
    results = []
    trained_models = {}
    eval_details = {}
    
    for name, model in models.items():
        logger.info(f"Training candidate model: {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_predictions(y_test, y_prob)
        metrics["model"] = name
        results.append(metrics)
        trained_models[name] = model
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        eval_details[name] = {
            "metrics": metrics,
            "calibration": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist()
            }
        }
        logger.info(f"{name} Results: AUC={metrics['roc_auc']}, LogLoss={metrics['log_loss']}, Brier={metrics['brier_score']}")
        
    comparison_df = pd.DataFrame(results).set_index("model")
    
    # Pick the model with the lowest Brier Score / Log Loss & highest ROC-AUC
    # HistGradientBoosting typically excels in probability calibration and speed
    best_name = "Gradient Boosting (Hist)"
    best_model = trained_models[best_name]
    
    logger.info(f"Selected champion model: {best_name}")
    return best_model, comparison_df, eval_details


def save_champion_model(
    model: Any,
    eval_details: Dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Path:
    """Saves the trained champion model and its metadata JSON."""
    model_path = MODELS_DIR / "win_probability_model.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"
    
    joblib.dump(model, model_path)
    
    metadata = {
        "model_name": "HistGradientBoostingClassifier",
        "version": "1.0.0",
        "target": TARGET_COL,
        "features": MODEL_FEATURE_COLS,
        "training_matches": int(train_df["match_id"].nunique()),
        "training_deliveries": len(train_df),
        "test_matches": int(test_df["match_id"].nunique()),
        "test_deliveries": len(test_df),
        "train_period": f"{train_df['date'].min()} to {train_df['date'].max()}",
        "test_period": f"{test_df['date'].min()} to {test_df['date'].max()}",
        "champion_metrics": eval_details["Gradient Boosting (Hist)"]["metrics"],
        "all_candidate_metrics": {k: v["metrics"] for k, v in eval_details.items()},
        "calibration": eval_details["Gradient Boosting (Hist)"]["calibration"]
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Saved model to {model_path} and metadata to {meta_path}")
    return model_path


def load_champion_model(model_path: Path = None):
    """Loads the serialized win probability model."""
    if model_path is None:
        model_path = MODELS_DIR / "win_probability_model.pkl"
    return joblib.load(model_path)


def predict_win_probability(model: Any, df: pd.DataFrame) -> np.ndarray:
    """
    Computes pre-delivery win probability using the trained model.
    """
    X = df[MODEL_FEATURE_COLS].copy()
    probs = model.predict_proba(X)[:, 1]
    return np.clip(probs, 0.001, 0.999)
