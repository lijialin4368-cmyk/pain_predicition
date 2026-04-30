"""BorderlineSMOTE augmentation configuration."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_DIR / "data" / "processed" / "data_vectorized.csv"
SPLIT_FILE = PROJECT_DIR / "splits" / "reference_splits_seed_42.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

TARGET_COLUMN = "术后第一天_静息痛"
TARGET_THRESHOLD = 4.0
SPLIT_SEED = 42
RANDOM_STATE = 42

K_NEIGHBORS = 5
M_NEIGHBORS = 10
SAMPLING_STRATEGY = "auto"
KIND = "borderline-1"

META_PREFIX = "__meta_"
META_IS_GENERATED_COL = "__meta_is_generated"
META_SPLIT_COL = "__meta_dataset_split"
META_SOURCE_ROW_ID_COL = "__meta_source_row_id"
META_AUGMENTATION_METHOD_COL = "__meta_augmentation_method"
META_TARGET_COLUMN_COL = "__meta_target_column"
META_TARGET_THRESHOLD_COL = "__meta_target_threshold"
META_BINARY_TARGET_COL = "__meta_binary_target"

TRAIN_SPLIT_VALUE = "train"
VALIDATION_SPLIT_VALUE = "validation"
TEST_SPLIT_VALUE = "test"

AUGMENTED_DATASET_FILENAME = "augmented_dataset.csv"
GENERATED_ONLY_FILENAME = "generated_only.csv"
TRAIN_ORIGINAL_FILENAME = "train_original.csv"
VALIDATION_ORIGINAL_FILENAME = "validation_original.csv"
TEST_ORIGINAL_FILENAME = "test_original.csv"
SUMMARY_FILENAME = "borderline_smote_summary.json"
