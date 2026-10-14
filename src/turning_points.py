"""
IPL FORENSICS - Turning Point Detection and Explanation Engine
Identifies high-leverage deliveries where win probability swings dramatically,
classifies the event type, and generates factual investigative autopsy reports.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.utils import logger


def calculate_probability_swings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes ball-by-ball win probability swings within each match and innings.
    probability_change = win_prob(t) - win_prob(t-1)
    turning_point_score = abs(probability_change) * 100
    """
    df = df.copy()
    grouped = df.groupby(["match_id", "innings"])
    
    # win_probability represents the state BEFORE the delivery
    # The probability AFTER the delivery is the pre-delivery probability of the NEXT delivery
    # Or for the final ball, 1.0 (if won) or 0.0 (if lost)
    df["win_probability_before"] = df["win_probability"]
    
    # Next ball's pre-delivery probability within the same match & innings
    next_prob = grouped["win_probability"].shift(-1)
    
    # For the last ball of the innings: if innings 2, final result determines outcome (1.0 or 0.0)
    final_outcome = df["batting_team_won"].astype(float)
    df["win_probability_after"] = next_prob.fillna(final_outcome)
    
    # Probability swing caused by this delivery
    df["probability_change"] = df["win_probability_after"] - df["win_probability_before"]
    df["turning_point_score"] = df["probability_change"].abs() * 100.0
    
    # Convenience aliases for UI compatibility
    df["turning_point_impact"] = df["turning_point_score"].round(2)
    df["impact"] = df["turning_point_score"].round(2)
    
    return df


def classify_delivery_event(row: pd.Series) -> str:
    """
    Classifies the tactical nature of the delivery:
    WICKET, SIX, FOUR, DOT BALL, SCORING BURST, COLLAPSE, RECOVERY, or OTHER.
    """
    is_wicket = row.get("is_wicket", 0) == 1
    runs_batter = row.get("runs_batter", 0)
    runs_total = row.get("runs_total", 0)
    prob_change = row.get("probability_change", 0.0)
    wickets_last_6 = row.get("wickets_last_6_before", 0)
    runs_last_6 = row.get("runs_last_6_before", 0)
    
    if is_wicket:
        if wickets_last_6 >= 2:
            return "COLLAPSE"
        return "WICKET"
        
    if runs_batter == 6:
        return "SIX"
        
    if runs_batter == 4:
        return "FOUR"
        
    if runs_total == 0:
        return "DOT BALL"
        
    if runs_last_6 >= 16 and prob_change > 0.08:
        return "SCORING BURST"
        
    if prob_change >= 0.12 and wickets_last_6 >= 1:
        return "RECOVERY"
        
    return "OTHER"


def generate_turning_point_explanation(row: pd.Series) -> str:
    """
    Generates a natural-language cricket forensic explanation of the turning point.
    """
    team = row.get("batting_team", "Batting team")
    bowler = row.get("bowler", "the bowler")
    batter = row.get("batter", "the batter")
    event = row.get("event", "Event")
    over = int(row.get("over", 0)) + 1
    ball = int(row.get("delivery", 1))
    score = int(row.get("score", 0))
    wickets = int(row.get("wickets_lost", 0))
    p_before = round(float(row.get("win_probability_before", 0.5)) * 100, 1)
    p_after = round(float(row.get("win_probability_after", 0.5)) * 100, 1)
    swing = round(abs(float(row.get("probability_change", 0.0))) * 100, 1)
    direction = "rose" if row.get("probability_change", 0.0) > 0 else "plummeted"
    phase = row.get("phase", "match")
    
    if event in ["WICKET", "COLLAPSE"]:
        player_out = row.get("player_out") or batter
        kind = row.get("wicket_kind", "dismissal")
        return (
            f"CRITICAL DISMISSAL ({over}.{ball} ov): {player_out} fell to {bowler} ({kind}) "
            f"with {team} at {score}/{wickets}. Under the pressure of the {phase.lower()} phase, "
            f"win probability {direction} from {p_before}% to {p_after}% (a {swing} percentage-point swing)."
        )
    elif event == "SIX":
        return (
            f"MAXIMUM IMPACT ({over}.{ball} ov): {batter} dispatched {bowler} for SIX. "
            f"This boundary shifted match momentum toward {team}, lifting win probability "
            f"from {p_before}% to {p_after}% (+{swing} pp)."
        )
    elif event == "FOUR":
        return (
            f"BOUNDARY RELEASE ({over}.{ball} ov): {batter} found the boundary off {bowler}. "
            f"The four relieved mounting pressure, swinging win probability from {p_before}% to {p_after}% (+{swing} pp)."
        )
    elif event == "DOT BALL":
        req_rr = round(float(row.get("required_run_rate_before", 0)), 1)
        return (
            f"PRESSURE DOT ({over}.{ball} ov): {bowler} delivered a vital dot ball to {batter}. "
            f"With required run rate at {req_rr}, the dot suffocated {team}'s chase, dropping probability "
            f"from {p_before}% to {p_after}% (-{swing} pp)."
        )
    elif event == "SCORING BURST":
        return (
            f"SCORING ONSLAUGHT ({over}.{ball} ov): {team} unleashed an aggressive scoring burst, "
            f"gaining massive momentum off {bowler}. Win probability surged from {p_before}% to {p_after}% (+{swing} pp)."
        )
    elif event == "RECOVERY":
        return (
            f"RESILIENT RECOVERY ({over}.{ball} ov): {team} counter-attacked effectively after earlier setbacks, "
            f"elevating their win probability from {p_before}% to {p_after}% (+{swing} pp)."
        )
    else:
        return (
            f"TACTICAL SHIFT ({over}.{ball} ov): A {swing} percentage-point swing occurred "
            f"as {team} moved to {score}/{wickets} facing {bowler}. Win probability transitioned from {p_before}% to {p_after}%."
        )


def detect_top_turning_points(
    df: pd.DataFrame,
    top_n_per_match: int = 15,
    min_score_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Computes probability changes, classifies events, generates explanations,
    and returns top turning points across all matches.
    """
    logger.info("Detecting turning points and computing probability shifts...")
    df_swings = calculate_probability_swings(df)
    
    # Classify each delivery event
    df_swings["event"] = df_swings.apply(classify_delivery_event, axis=1)
    df_swings["event_type"] = df_swings["event"]
    
    # Filter for significant turning points
    tp_candidates = df_swings[df_swings["turning_point_score"] >= min_score_threshold].copy()
    
    # Generate explanations
    logger.info(f"Generating explanations for {len(tp_candidates)} turning points...")
    tp_candidates["explanation"] = tp_candidates.apply(generate_turning_point_explanation, axis=1)
    
    # Rank by turning_point_score descending within each match
    tp_ranked = (
        tp_candidates
        .sort_values(by=["match_id", "turning_point_score"], ascending=[True, False])
        .groupby("match_id")
        .head(top_n_per_match)
        .reset_index(drop=True)
    )
    
    logger.info(f"Generated {len(tp_ranked)} top turning points across {tp_ranked['match_id'].nunique()} matches.")
    return tp_ranked, df_swings
