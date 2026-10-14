"""
IPL FORENSICS - Feature Engineering
Constructs strict pre-delivery match state features with zero future leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from src.utils import logger


def create_pre_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs pre-delivery features:
    - score_before
    - wickets_before
    - balls_bowled_before
    - overs_completed_before
    - balls_remaining_before
    - current_run_rate_before
    - runs_required_before
    - required_run_rate_before
    - runs_last_6_before
    - wickets_last_6_before
    - fours_last_6_before
    - sixes_last_6_before
    
    All features represent the exact state BEFORE the bowler bowls the delivery.
    """
    df = df.copy()
    
    # Pre-delivery fundamental match state
    df["score_before"] = df["score"] - df["runs_total"]
    df["wickets_before"] = df["wickets_lost"] - df["is_wicket"]
    df["balls_bowled_before"] = np.maximum(0, df["legal_balls_bowled"] - df["is_legal_ball"])
    df["overs_completed_before"] = df["balls_bowled_before"] / 6.0
    df["balls_remaining_before"] = np.clip(120 - df["balls_bowled_before"], 0, 120)
    
    # Pre-delivery Current Run Rate
    df["current_run_rate_before"] = np.where(
        df["overs_completed_before"] > 0,
        df["score_before"] / df["overs_completed_before"],
        0.0
    )
    
    # Pre-delivery chasing state (Innings 2)
    df["runs_required_before"] = np.where(
        df["innings"] == 2,
        np.maximum(0, df["target_score"] - df["score_before"]),
        0.0
    )
    
    overs_rem_before = df["balls_remaining_before"] / 6.0
    df["required_run_rate_before"] = np.where(
        (df["innings"] == 2) & (overs_rem_before > 0),
        df["runs_required_before"] / overs_rem_before,
        np.where(
            (df["innings"] == 2) & (df["runs_required_before"] > 0),
            99.0,
            0.0
        )
    )
    
    # Rolling 6-ball momentum metrics strictly BEFORE the delivery
    # Using shift(1) within (match_id, innings) ensures current ball outcome is NEVER included
    grouped = df.groupby(["match_id", "innings"])
    
    df["runs_last_6_before"] = (
        grouped["runs_total"]
        .shift(1)
        .rolling(6, min_periods=1)
        .sum()
        .fillna(0.0)
    )
    
    df["wickets_last_6_before"] = (
        grouped["is_wicket"]
        .shift(1)
        .rolling(6, min_periods=1)
        .sum()
        .fillna(0.0)
    )
    
    df["fours_last_6_before"] = (
        grouped["is_four"]
        .shift(1)
        .rolling(6, min_periods=1)
        .sum()
        .fillna(0.0)
    )
    
    df["sixes_last_6_before"] = (
        grouped["is_six"]
        .shift(1)
        .rolling(6, min_periods=1)
        .sum()
        .fillna(0.0)
    )
    
    # Also maintain post-delivery rolling stats for general analytics
    df["runs_last_6"] = grouped["runs_total"].rolling(6, min_periods=1).sum().reset_index(drop=True)
    df["wickets_last_6"] = grouped["is_wicket"].rolling(6, min_periods=1).sum().reset_index(drop=True)
    df["fours_last_6"] = grouped["is_four"].rolling(6, min_periods=1).sum().reset_index(drop=True)
    df["sixes_last_6"] = grouped["is_six"].rolling(6, min_periods=1).sum().reset_index(drop=True)
    
    # Recent run rate (over last 6 balls)
    df["recent_run_rate"] = df["runs_last_6"]  # runs in last 6 balls = run rate per over in last 6 balls
    df["momentum_signal"] = df["recent_run_rate"] - df["current_run_rate"]
    
    return df


# Definition of features used by the ML model
MODEL_FEATURE_COLS = [
    "innings",
    "score_before",
    "wickets_before",
    "balls_bowled_before",
    "balls_remaining_before",
    "current_run_rate_before",
    "runs_required_before",
    "required_run_rate_before",
    "runs_last_6_before",
    "wickets_last_6_before",
    "fours_last_6_before",
    "sixes_last_6_before"
]

TARGET_COL = "batting_team_won"


def get_model_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts X and y for ML training with strict pre-delivery features."""
    X = df[MODEL_FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y
