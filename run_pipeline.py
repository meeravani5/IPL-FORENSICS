"""
IPL FORENSICS - Master End-to-End Execution Pipeline
Executes data ingestion -> preprocessing -> feature engineering -> ML training
-> probability estimation -> turning point detection -> match autopsy -> what-if simulation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import logger, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.data_loader import load_all_raw_matches
from src.preprocessing import clean_and_prepare_deliveries
from src.feature_engineering import create_pre_delivery_features
from src.model import (
    split_data_by_match,
    train_and_compare_models,
    save_champion_model,
    predict_win_probability
)
from src.turning_points import detect_top_turning_points
from src.autopsy import generate_all_match_autopsies
from src.what_if import generate_default_what_if_dataset


def run_full_pipeline():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("🏏 STARTING IPL FORENSICS COMPLETE DATA SCIENCE PIPELINE")
    logger.info("=" * 70)
    
    # 1. DATA INGESTION
    logger.info("Phase 1: Ingesting raw Cricsheet JSON matches from data/raw/...")
    matches_df, raw_deliveries_df, ingestion_summary = load_all_raw_matches(RAW_DATA_DIR)
    logger.info(f"Ingested {len(matches_df)} valid matches, {len(raw_deliveries_df)} deliveries.")
    
    # 2. PREPROCESSING
    logger.info("Phase 2: Preprocessing and progressive match state computation...")
    prep_df, val_report = clean_and_prepare_deliveries(raw_deliveries_df)
    
    # 3. FEATURE ENGINEERING
    logger.info("Phase 3: Strict pre-delivery feature engineering (Zero-leakage)...")
    featured_df = create_pre_delivery_features(prep_df)
    logger.info(f"Feature engineering complete. Total columns: {len(featured_df.columns)}")
    
    # 4. MACHINE LEARNING & EVALUATION
    logger.info("Phase 4: ML model training with chronological match-level split...")
    train_df, test_df = split_data_by_match(featured_df, test_ratio=0.20, chronological=True)
    champion_model, comparison_df, eval_details = train_and_compare_models(train_df, test_df)
    
    logger.info("\n" + "=" * 50)
    logger.info("MODEL COMPARISON RESULTS:")
    logger.info("\n" + comparison_df.to_string())
    logger.info("=" * 50 + "\n")
    
    save_champion_model(champion_model, eval_details, train_df, test_df)
    
    # 5. WIN PROBABILITY INFERENCE
    logger.info("Phase 5: Generating win probabilities for all deliveries...")
    featured_df["win_probability"] = predict_win_probability(champion_model, featured_df)
    
    # Save primary deliveries dataset
    deliv_csv_path = PROCESSED_DATA_DIR / "ipl_deliveries.csv"
    logger.info(f"Saving deliveries dataset to {deliv_csv_path}...")
    featured_df.to_csv(deliv_csv_path, index=False)
    
    # 6. TURNING POINT DETECTION
    logger.info("Phase 6: Detecting and classifying turning points...")
    top_tps_df, df_with_swings = detect_top_turning_points(featured_df, top_n_per_match=15)
    tp_csv_path = PROCESSED_DATA_DIR / "top_turning_points.csv"
    top_tps_df.to_csv(tp_csv_path, index=False)
    logger.info(f"Saved {len(top_tps_df)} turning points to {tp_csv_path}")
    
    # 7. MATCH AUTOPSY
    logger.info("Phase 7: Generating match autopsies for all matches...")
    autopsy_df = generate_all_match_autopsies(df_with_swings, top_tps_df)
    autopsy_csv_path = PROCESSED_DATA_DIR / "match_autopsy_summary.csv"
    autopsy_df.to_csv(autopsy_csv_path, index=False)
    logger.info(f"Saved {len(autopsy_df)} match autopsies to {autopsy_csv_path}")
    
    # 8. WHAT-IF SCENARIOS
    logger.info("Phase 8: Generating What-If reference scenario dataset...")
    what_if_df = generate_default_what_if_dataset(champion_model, top_tps_df)
    what_if_csv_path = PROCESSED_DATA_DIR / "what_if_scenarios.csv"
    what_if_df.to_csv(what_if_csv_path, index=False)
    logger.info(f"Saved what-if scenarios to {what_if_csv_path}")
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    logger.info(f"Matches Processed: {len(matches_df)}")
    logger.info(f"Deliveries: {len(featured_df)}")
    logger.info(f"Turning Points: {len(top_tps_df)}")
    logger.info(f"Autopsies: {len(autopsy_df)}")
    logger.info(f"Champion Model ROC-AUC: {eval_details['Gradient Boosting (Hist)']['metrics']['roc_auc']}")
    logger.info(f"Champion Model Brier Score: {eval_details['Gradient Boosting (Hist)']['metrics']['brier_score']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
