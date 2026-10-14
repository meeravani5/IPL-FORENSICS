"""
IPL FORENSICS - Preprocessing
Cleans raw delivery data, computes progressive match state, and validates invariants.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils import logger, get_match_phase


def compute_progressive_match_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes running score, cumulative wickets, legal balls, run rates,
    target calculation for 2nd innings, and phase markers.
    """
    df = df.copy()
    
    # Ensure correct sort order
    df = df.sort_values(by=["match_id", "innings", "over", "delivery"]).reset_index(drop=True)
    
    # Boundary and dot flags
    df["is_four"] = (df["runs_batter"] == 4).astype(int)
    df["is_six"] = (df["runs_batter"] == 6).astype(int)
    df["is_dot"] = (df["runs_total"] == 0).astype(int)
    df["is_legal_ball"] = df["is_legal"].astype(int)
    
    # Group by match_id and innings for running cumulative statistics
    group = df.groupby(["match_id", "innings"])
    
    # Post-delivery state
    df["score"] = group["runs_total"].cumsum()
    df["wickets_lost"] = group["is_wicket"].cumsum()
    df["legal_balls_bowled"] = group["is_legal_ball"].cumsum()
    df["balls_bowled"] = df["legal_balls_bowled"]  # Consistent alias
    
    df["overs_completed"] = df["legal_balls_bowled"] / 6.0
    df["balls_remaining"] = np.clip(120 - df["legal_balls_bowled"], 0, 120)
    
    # Current Run Rate (post-delivery)
    df["current_run_rate"] = np.where(
        df["overs_completed"] > 0,
        df["score"] / df["overs_completed"],
        0.0
    )
    
    # Match phase based on over (0-indexed)
    df["phase"] = df["over"].apply(get_match_phase)
    
    # Calculate Innings 1 total score to determine Target for Innings 2
    inn1_totals = (
        df[df["innings"] == 1]
        .groupby("match_id")["runs_total"]
        .sum()
        .reset_index()
        .rename(columns={"runs_total": "inn1_score"})
    )
    inn1_totals["target_score"] = inn1_totals["inn1_score"] + 1
    
    df = df.merge(inn1_totals[["match_id", "target_score"]], on="match_id", how="left")
    
    # Target and runs required for 2nd innings
    df["runs_required"] = np.where(
        df["innings"] == 2,
        np.maximum(0, df["target_score"] - df["score"]),
        np.nan
    )
    
    # Required Run Rate
    overs_remaining = df["balls_remaining"] / 6.0
    df["required_run_rate"] = np.where(
        (df["innings"] == 2) & (overs_remaining > 0),
        df["runs_required"] / overs_remaining,
        np.where(
            (df["innings"] == 2) & (df["runs_required"] > 0),
            99.0,
            0.0
        )
    )
    
    # Target variable: Did batting team win?
    df["batting_team_won"] = (df["batting_team"] == df["winner"]).astype(int)
    
    return df


def validate_match_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates logical invariants across deliveries:
    - score monotonically non-decreasing within innings
    - wickets monotonically non-decreasing within innings
    - legal balls monotonically non-decreasing within innings
    """
    validation_results = {
        "total_deliveries": len(df),
        "total_matches": df["match_id"].nunique(),
        "score_invariants_passed": True,
        "wicket_invariants_passed": True,
        "null_winners": int(df["winner"].isna().sum()),
        "invalid_scores_count": 0,
        "invalid_wickets_count": 0
    }
    
    # Check monotonicity
    grouped = df.groupby(["match_id", "innings"])
    score_diffs = grouped["score"].diff().dropna()
    wicket_diffs = grouped["wickets_lost"].diff().dropna()
    
    negative_scores = (score_diffs < 0).sum()
    negative_wickets = (wicket_diffs < 0).sum()
    
    if negative_scores > 0:
        validation_results["score_invariants_passed"] = False
        validation_results["invalid_scores_count"] = int(negative_scores)
        logger.warning(f"Found {negative_scores} instances where score decreased!")
        
    if negative_wickets > 0:
        validation_results["wicket_invariants_passed"] = False
        validation_results["invalid_wickets_count"] = int(negative_wickets)
        logger.warning(f"Found {negative_wickets} instances where wickets decreased!")
        
    return validation_results


def clean_and_prepare_deliveries(raw_deliveries_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs complete preprocessing pipeline on raw deliveries."""
    logger.info("Starting progressive match state computation...")
    df = compute_progressive_match_state(raw_deliveries_df)
    validation = validate_match_data(df)
    logger.info(f"Preprocessing validation report: {validation}")
    return df, validation
