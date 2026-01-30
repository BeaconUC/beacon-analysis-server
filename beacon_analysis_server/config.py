from pathlib import Path
import tomllib

from dotenv import load_dotenv
from loguru import logger
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CONFIG_PATH = PROJ_ROOT / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

SENTIMENT_MAP = cfg["sentiment"]["mapping"]
PHRASES = cfg["urgency"]["phrases"]
URGENCY_THRESHOLD = cfg["urgency"]["threshold"]
TECHNICAL_CANDIDATES = cfg["rca"]["candidates"]
RCA_DIVERSITY = cfg["rca"]["diversity"]
RCA_TOP_N = cfg["rca"]["top_n"]

TAGALOG_STOP_WORDS = cfg["language"]["tagalog_stop_words"]
EXTENDED_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(TAGALOG_STOP_WORDS))

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
