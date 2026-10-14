"""
IPL FORENSICS - Utilities and Constants
Central configuration, paths, team mappings, and helper functions.
"""

from pathlib import Path
import logging

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("IPL-FORENSICS")

# Standardized IPL Franchise Names
TEAM_NAME_STANDARDIZATION = {
    "Delhi Daredevils": "Delhi Capitals",
    "Delhi Capitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Deccan Chargers": "Deccan Chargers",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
    "Mumbai Indians": "Mumbai Indians",
    "Chennai Super Kings": "Chennai Super Kings",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Rajasthan Royals": "Rajasthan Royals",
    "Gujarat Lions": "Gujarat Lions",
    "Gujarat Titans": "Gujarat Titans",
    "Lucknow Super Giants": "Lucknow Super Giants",
    "Pune Warriors": "Pune Warriors",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala"
}

def standardize_team_name(name: str) -> str:
    """Standardizes franchise name if found in mapping, otherwise returns original."""
    if not name or not isinstance(name, str):
        return name
    return TEAM_NAME_STANDARDIZATION.get(name.strip(), name.strip())

def get_match_phase(over: int) -> str:
    """
    Returns match phase based on over index (0-indexed: 0-19):
    0-5: Powerplay (overs 1-6)
    6-14: Middle Overs (overs 7-15)
    15-19: Death Overs (overs 16-20)
    """
    if over < 6:
        return "Powerplay"
    elif over < 15:
        return "Middle"
    else:
        return "Death"
