"""
IPL FORENSICS - Match Autopsy Engine
Performs post-mortem analytical breakdown explaining why and how matches were won or lost.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils import logger


def generate_single_match_autopsy(
    match_id: str,
    match_delivs: pd.DataFrame,
    match_tps: pd.DataFrame
) -> Dict[str, Any]:
    """
    Constructs a comprehensive post-mortem autopsy for a single match:
    - Result, winner, loser, margin
    - Strongest phase (max net win prob delta)
    - Weakest / pressure phase
    - Largest probability swing & decisive event
    - Major wicket dismissal
    - Natural language forensic clinical summary
    """
    if len(match_delivs) == 0:
        return {}
        
    winner = match_delivs["winner"].iloc[0]
    teams = match_delivs["batting_team"].dropna().unique().tolist()
    loser = [t for t in teams if t != winner]
    loser = loser[0] if loser else "Opposition"
    
    # Analyze phases (Powerplay, Middle, Death) by net probability swing
    phase_metrics = (
        match_delivs.groupby("phase")["probability_change"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    
    strongest_phase = "Middle"
    weakest_phase = "Death"
    if not phase_metrics.empty:
        best_row = phase_metrics.sort_values(by="sum", ascending=False).iloc[0]
        worst_row = phase_metrics.sort_values(by="sum", ascending=True).iloc[0]
        strongest_phase = best_row["phase"]
        weakest_phase = worst_row["phase"]
        
    # Pressure phase: phase with highest dot ball frequency and lowest scoring rate
    phase_dots = (
        match_delivs.groupby("phase")["is_dot"]
        .mean()
        .reset_index()
        .sort_values(by="is_dot", ascending=False)
    )
    pressure_phase = phase_dots.iloc[0]["phase"] if not phase_dots.empty else "Middle"
    
    # Largest turning point
    decisive_event = "Decisive sequence"
    decisive_batter = None
    decisive_bowler = None
    largest_swing = 0.0
    decisive_impact = 0.0
    major_wicket = "None recorded"
    
    if len(match_tps) > 0:
        top_tp = match_tps.sort_values(by="turning_point_score", ascending=False).iloc[0]
        decisive_event = f"{top_tp.get('event', 'Shift')} (Over {int(top_tp.get('over', 0)) + 1}.{int(top_tp.get('delivery', 1))})"
        decisive_batter = top_tp.get("batter")
        decisive_bowler = top_tp.get("bowler")
        decisive_impact = round(float(top_tp.get("turning_point_score", 0)), 2)
        largest_swing = decisive_impact
        
        # Find biggest wicket event
        wickets = match_tps[match_tps["is_wicket"] == 1]
        if not wickets.empty:
            top_w = wickets.sort_values(by="turning_point_score", ascending=False).iloc[0]
            player_out = top_w.get("player_out") or top_w.get("batter")
            major_wicket = f"{player_out} b/c {top_w.get('bowler')} in over {int(top_w.get('over', 0)) + 1} (-{round(float(top_w.get('turning_point_score', 0)), 1)} pp)"
    
    # Generate clinical forensic narrative summary
    summary = (
        f"MATCH AUTOPSY REPORT: {winner} prevailed over {loser}. "
        f"The decisive momentum shift occurred during the {strongest_phase.lower()} phase, "
        f"anchored by {decisive_event} involving {decisive_bowler or 'the bowling unit'} and {decisive_batter or 'the batters'}, "
        f"which catalyzed a {decisive_impact} percentage-point win probability swing. "
        f"{loser} faced their most stifling pressure in the {pressure_phase.lower()} overs, "
        f"where boundary drought and critical dismissals proved insurmountable."
    )
    
    return {
        "match_id": match_id,
        "winner": winner,
        "loser": loser,
        "summary": summary,
        "event": decisive_event,
        "batter": decisive_batter,
        "bowler": decisive_bowler,
        "turning_point_impact": decisive_impact,
        "largest_turning_point": decisive_event,
        "largest_probability_swing": largest_swing,
        "major_wicket": major_wicket,
        "best_scoring_phase": strongest_phase,
        "worst_scoring_phase": weakest_phase,
        "pressure_phase": pressure_phase,
        "decisive_moment": decisive_event,
        "final_explanation": summary
    }


def generate_all_match_autopsies(
    deliveries_df: pd.DataFrame,
    turning_points_df: pd.DataFrame
) -> pd.DataFrame:
    """Computes autopsy summaries for all unique matches in the deliveries dataset."""
    logger.info("Generating match autopsies for all matches...")
    unique_matches = deliveries_df["match_id"].unique()
    
    autopsies = []
    for match_id in unique_matches:
        m_delivs = deliveries_df[deliveries_df["match_id"] == match_id]
        m_tps = turning_points_df[turning_points_df["match_id"] == match_id] if "match_id" in turning_points_df.columns else pd.DataFrame()
        
        autopsy = generate_single_match_autopsy(match_id, m_delivs, m_tps)
        if autopsy:
            autopsies.append(autopsy)
            
    autopsy_df = pd.DataFrame(autopsies)
    logger.info(f"Generated {len(autopsy_df)} match autopsies.")
    return autopsy_df
