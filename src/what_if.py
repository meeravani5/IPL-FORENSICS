"""
IPL FORENSICS - What-If Scenario Simulator
Interactive counterfactual simulation engine for evaluating alternative match outcomes.
Simulates hypothetical events and computes updated win probabilities using the trained ML model.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.utils import logger
from src.feature_engineering import MODEL_FEATURE_COLS

DISCLAIMER_TEXT = "Scenario estimate based on mathematical state simulation — not proof of causal outcome."


def simulate_counterfactual(
    model: Any,
    row: pd.Series,
    scenario_type: str,
    delta_runs: int = 0
) -> Dict[str, Any]:
    """
    Simulates a counterfactual scenario on a given delivery:
    - 'NO_WICKET': What if the wicket had not fallen?
    - 'DOT_INSTEAD_OF_BOUNDARY': What if the ball was a dot (0 runs)?
    - 'SIX_INSTEAD_OF_DOT': What if the ball had been hit for SIX?
    - 'OVER_RUNS_BOOST': What if the over yielded +5 runs?
    - 'OVER_RUNS_DROP': What if the over yielded -5 runs?
    - 'EVENT_OVERRIDE': Specify alternative runs and wicket status.
    """
    sim_row = row.copy()
    orig_prob = float(row.get("win_probability_after", row.get("win_probability", 0.5)))
    
    # Extract baseline pre-delivery variables
    innings = int(row.get("innings", 1))
    score_before = float(row.get("score_before", 0))
    wickets_before = float(row.get("wickets_before", 0))
    balls_bowled = float(row.get("balls_bowled_before", 0)) + 1  # State after the ball
    balls_remaining = np.clip(120 - balls_bowled, 0, 120)
    overs_completed = balls_bowled / 6.0
    target_score = float(row.get("target_score", 0)) if "target_score" in row else 160.0
    runs_last_6_before = float(row.get("runs_last_6_before", 0))
    wickets_last_6_before = float(row.get("wickets_last_6_before", 0))
    fours_last_6_before = float(row.get("fours_last_6_before", 0))
    sixes_last_6_before = float(row.get("sixes_last_6_before", 0))
    
    sim_runs = int(row.get("runs_total", 0))
    sim_wicket = int(row.get("is_wicket", 0))
    scenario_desc = ""
    
    if scenario_type == "NO_WICKET" or scenario_type == "What if wicket did NOT occur?":
        sim_wicket = 0
        scenario_desc = "Counterfactual: The wicket does not fall on this delivery (team retains batsman)."
    elif scenario_type == "DOT_INSTEAD_OF_BOUNDARY" or scenario_type == "What if boundary became a DOT ball?":
        sim_runs = 0
        scenario_desc = "Counterfactual: Delivery produces 0 runs instead of a boundary."
    elif scenario_type == "SIX_INSTEAD_OF_DOT" or scenario_type == "What if ball was hit for SIX?":
        sim_runs = 6
        sim_wicket = 0
        scenario_desc = "Counterfactual: Batter hits a SIX instead of recorded delivery outcome."
    elif scenario_type == "OVER_RUNS_BOOST" or scenario_type == "What if the over scored 5 more runs?":
        sim_runs = sim_runs + 5
        scenario_desc = "Counterfactual: Batting team scores an additional 5 runs in this phase."
    elif scenario_type == "OVER_RUNS_DROP" or scenario_type == "What if the over scored 5 fewer runs?":
        sim_runs = max(0, sim_runs - 5)
        scenario_desc = "Counterfactual: Batting team scores 5 fewer runs in this phase."
    elif scenario_type == "DOT BALL":
        sim_runs = 0
        sim_wicket = 0
        scenario_desc = "Scenario: Ball produces a dot ball."
    elif scenario_type == "SINGLE" or scenario_type == "1 RUN":
        sim_runs = 1
        sim_wicket = 0
        scenario_desc = "Scenario: Batter takes a single."
    elif scenario_type == "DOUBLE" or scenario_type == "2 RUNS":
        sim_runs = 2
        sim_wicket = 0
        scenario_desc = "Scenario: Batters run two runs."
    elif scenario_type == "FOUR":
        sim_runs = 4
        sim_wicket = 0
        scenario_desc = "Scenario: Batter hits a four."
    elif scenario_type == "SIX":
        sim_runs = 6
        sim_wicket = 0
        scenario_desc = "Scenario: Batter hits a six."
    elif scenario_type == "WICKET":
        sim_runs = 0
        sim_wicket = 1
        scenario_desc = "Scenario: A wicket falls on this ball."
    else:
        sim_runs = delta_runs
        scenario_desc = f"Counterfactual: Custom scenario applied (+{delta_runs} runs)."
        
    # Reconstruct post-delivery feature vector
    new_score = score_before + sim_runs
    new_wickets = min(10.0, wickets_before + sim_wicket)
    new_crr = (new_score / overs_completed) if overs_completed > 0 else 0.0
    new_runs_req = max(0.0, target_score - new_score) if innings == 2 else 0.0
    new_rrr = (new_runs_req / (balls_remaining / 6.0)) if (innings == 2 and balls_remaining > 0) else (99.0 if new_runs_req > 0 else 0.0)
    
    new_runs_last_6 = runs_last_6_before + sim_runs
    new_wickets_last_6 = wickets_last_6_before + sim_wicket
    new_fours_last_6 = fours_last_6_before + (1 if sim_runs == 4 else 0)
    new_sixes_last_6 = sixes_last_6_before + (1 if sim_runs == 6 else 0)
    
    feature_dict = {
        "innings": innings,
        "score_before": new_score,
        "wickets_before": new_wickets,
        "balls_bowled_before": balls_bowled,
        "balls_remaining_before": balls_remaining,
        "current_run_rate_before": new_crr,
        "runs_required_before": new_runs_req,
        "required_run_rate_before": new_rrr,
        "runs_last_6_before": new_runs_last_6,
        "wickets_last_6_before": new_wickets_last_6,
        "fours_last_6_before": new_fours_last_6,
        "sixes_last_6_before": new_sixes_last_6
    }
    
    feature_df = pd.DataFrame([feature_dict])[MODEL_FEATURE_COLS]
    sim_prob = float(model.predict_proba(feature_df)[:, 1][0])
    sim_prob = float(np.clip(sim_prob, 0.001, 0.999))
    
    prob_diff = sim_prob - orig_prob
    swing_pp = round(prob_diff * 100, 2)
    
    return {
        "scenario": scenario_type,
        "scenario_description": scenario_desc,
        "original_probability": round(orig_prob * 100, 2),
        "simulated_probability": round(sim_prob * 100, 2),
        "probability_difference": swing_pp,
        "runs_simulated": sim_runs,
        "wickets_simulated": sim_wicket,
        "disclaimer": DISCLAIMER_TEXT
    }


def generate_default_what_if_dataset(model: Any, tp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates reference what-if scenarios based on high-leverage deliveries.
    Provides standard scenario table compatible with existing Streamlit UI components.
    """
    logger.info("Generating reference What-If scenario dataset...")
    
    # Pick a high-leverage delivery for baseline comparison
    if len(tp_df) > 0:
        ref_row = tp_df.sort_values(by="turning_point_score", ascending=False).iloc[0]
    else:
        ref_row = pd.Series({
            "innings": 2, "score_before": 150, "wickets_before": 4, "balls_bowled_before": 105,
            "target_score": 175, "runs_last_6_before": 10, "wickets_last_6_before": 0,
            "fours_last_6_before": 1, "sixes_last_6_before": 0, "win_probability_after": 0.50
        })
        
    scenarios = ["SIX", "1 RUN", "2 RUNS", "FOUR", "DOT BALL", "WICKET"]
    runs_map = {"SIX": 6, "1 RUN": 1, "2 RUNS": 2, "FOUR": 4, "DOT BALL": 0, "WICKET": 0}
    
    rows = []
    for rank, sc in enumerate(scenarios, start=1):
        sim = simulate_counterfactual(model, ref_row, sc)
        rows.append({
            "Rank": rank,
            "Scenario": sc,
            "Runs": runs_map.get(sc, 0),
            "Win Probability (%)": sim["simulated_probability"]
        })
        
    return pd.DataFrame(rows)
