"""Linear regression baseline configuration."""

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

RIDGE_ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
ELASTICNET_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTICNET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
CV_SPLITS = 4
