"""
IPL FORENSICS - Data Loader
Reads Cricsheet JSON files and extracts match metadata and raw delivery rows.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from src.utils import logger, RAW_DATA_DIR, standardize_team_name


def load_single_match(file_path: Path) -> Optional[Dict[str, Any]]:
    """Loads a single Cricsheet match JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return None


def extract_match_meta(match_data: Dict[str, Any], match_id: str) -> Optional[Dict[str, Any]]:
    """Extracts match-level metadata and outcome info."""
    info = match_data.get("info", {})
    
    # Check outcome
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    result = outcome.get("result")  # e.g., 'no result', 'tie'
    
    # Identify winner if super over was played
    if not winner and outcome.get("eliminator"):
        winner = outcome.get("eliminator")
        
    dates = info.get("dates", [])
    match_date = dates[0] if dates else None
    
    teams = [standardize_team_name(t) for t in info.get("teams", [])]
    if len(teams) < 2:
        return None
        
    toss = info.get("toss", {})
    toss_winner = standardize_team_name(toss.get("winner"))
    toss_decision = toss.get("decision")
    
    by_info = outcome.get("by", {})
    win_by_runs = by_info.get("runs", 0)
    win_by_wickets = by_info.get("wickets", 0)
    
    venue = info.get("venue", "Unknown Venue")
    city = info.get("city", "")
    season = info.get("season", "")
    
    return {
        "match_id": match_id,
        "season": str(season),
        "date": match_date,
        "venue": venue,
        "city": city,
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": standardize_team_name(winner) if winner else None,
        "result": result,
        "win_by_runs": win_by_runs,
        "win_by_wickets": win_by_wickets,
        "player_of_match": info.get("player_of_match", [None])[0] if info.get("player_of_match") else None
    }


def extract_deliveries_from_match(match_data: Dict[str, Any], match_id: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses a single match JSON into a metadata dict and a list of raw delivery records.
    Only keeps innings 1 & 2 for standard T20 match analysis.
    """
    meta = extract_match_meta(match_data, match_id)
    if not meta:
        return None, []
        
    # If no valid winner exists (abandoned / no result), skip match
    if not meta["winner"]:
        return meta, []
        
    deliveries_list = []
    innings_list = match_data.get("innings", [])
    
    for innings_idx, innings_data in enumerate(innings_list, start=1):
        # We focus on the two main innings of the match
        if innings_idx > 2:
            break
            
        batting_team = standardize_team_name(innings_data.get("team", ""))
        # Determine bowling team
        bowling_team = meta["team2"] if batting_team == meta["team1"] else meta["team1"]
        
        for over_data in innings_data.get("overs", []):
            over_num = over_data.get("over", 0)
            
            for delivery_idx, delivery in enumerate(over_data.get("deliveries", []), start=1):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])
                
                # An illegal ball is wide or no-ball
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                is_legal = not (is_wide or is_noball)
                
                runs_batter = runs.get("batter", 0)
                runs_extras = runs.get("extras", 0)
                runs_total = runs.get("total", 0)
                
                is_wicket = 1 if wickets else 0
                wicket_kind = wickets[0].get("kind") if wickets else None
                player_out = wickets[0].get("player_out") if wickets else None
                
                deliveries_list.append({
                    "match_id": match_id,
                    "date": meta["date"],
                    "season": meta["season"],
                    "venue": meta["venue"],
                    "city": meta["city"],
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "innings": innings_idx,
                    "over": over_num,
                    "delivery": delivery_idx,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "non_striker": delivery.get("non_striker"),
                    "runs_batter": runs_batter,
                    "runs_extras": runs_extras,
                    "runs_total": runs_total,
                    "is_wicket": is_wicket,
                    "wicket_kind": wicket_kind,
                    "player_out": player_out,
                    "is_legal": is_legal,
                    "is_wide": is_wide,
                    "is_noball": is_noball,
                    "winner": meta["winner"]
                })
                
    return meta, deliveries_list


def load_all_raw_matches(raw_dir: Optional[Path] = None, max_matches: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Ingests all raw JSON files from raw_dir.
    Returns:
        (matches_df, deliveries_df, summary_stats)
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
        
    json_files = sorted(list(raw_dir.glob("*.json")))
    if max_matches:
        json_files = json_files[:max_matches]
        
    total_files = len(json_files)
    logger.info(f"Discovered {total_files} match files in {raw_dir}")
    
    successful_files = 0
    failed_files = 0
    skipped_matches = 0
    skip_reasons = {}
    
    matches_meta = []
    all_deliveries = []
    
    for fpath in json_files:
        match_id = fpath.stem
        match_data = load_single_match(fpath)
        if match_data is None:
            failed_files += 1
            skip_reasons["load_error"] = skip_reasons.get("load_error", 0) + 1
            continue
            
        meta, deliveries = extract_deliveries_from_match(match_data, match_id)
        if meta is None:
            failed_files += 1
            skip_reasons["invalid_metadata"] = skip_reasons.get("invalid_metadata", 0) + 1
            continue
            
        if not meta.get("winner") or len(deliveries) == 0:
            skipped_matches += 1
            reason = meta.get("result", "no_valid_winner")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
            
        matches_meta.append(meta)
        all_deliveries.extend(deliveries)
        successful_files += 1
        
    matches_df = pd.DataFrame(matches_meta)
    deliveries_df = pd.DataFrame(all_deliveries)
    
    summary = {
        "total_files_processed": total_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "skipped_matches": skipped_matches,
        "skip_reasons": skip_reasons,
        "total_matches": len(matches_df),
        "total_deliveries": len(deliveries_df)
    }
    
    logger.info(f"Ingestion complete: {summary}")
    return matches_df, deliveries_df, summary
