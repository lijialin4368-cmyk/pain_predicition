"""Naive regression baseline configuration."""

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "data_vectorized.csv"
TARGET_COLUMN = "术后第一天_静息痛"
MANUAL_FEATURE_COLUMNS = None

ENABLE_TEMPORAL_FILTER = True
STRICT_PAST_ONLY = True

TEST_SIZE = 0.1
RANDOM_STATE = 42

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "raw"
