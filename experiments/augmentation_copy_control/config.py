from pathlib import Path

BASE_AUGMENTATION_DIR = Path(__file__).resolve().parents[1] / "augmentation"
REFERENCE_OUTPUT_DIR = BASE_AUGMENTATION_DIR / "generated"

TRAIN_ORIGINAL_PATH = REFERENCE_OUTPUT_DIR / "train_original.csv"
VALIDATION_ORIGINAL_PATH = REFERENCE_OUTPUT_DIR / "validation_original.csv"
TEST_ORIGINAL_PATH = REFERENCE_OUTPUT_DIR / "test_original.csv"
REFERENCE_GENERATED_PATH = REFERENCE_OUTPUT_DIR / "generated_only.csv"
REFERENCE_SUMMARY_PATH = REFERENCE_OUTPUT_DIR / "augmentation_summary.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

META_IS_GENERATED_COL = "__meta_is_generated"
META_SPLIT_COL = "__meta_dataset_split"
META_SOURCE_ROW_ID_COL = "__meta_source_row_id"
META_RULE_COL = "__meta_generation_rule"
META_TRIGGER_COL = "__meta_trigger_reason"
META_PAIN_SHIFT_COL = "__meta_pain_shift"
META_AGE_DELTA_COL = "__meta_age_delta"
META_WEIGHT_DELTA_COL = "__meta_weight_delta"
META_VARIANT_NAME_COL = "__meta_variant_name"

TRAIN_SPLIT_VALUE = "train"
VALIDATION_SPLIT_VALUE = "validation"
TEST_SPLIT_VALUE = "test"

DUPLICATION_RULE_VALUE = "direct_duplicate"
DUPLICATION_TRIGGER_VALUE = "matched_reference_augmentation_count"
DUPLICATION_VARIANT_PREFIX = "duplicate"

TRAIN_ORIGINAL_FILENAME = "train_original.csv"
VALIDATION_ORIGINAL_FILENAME = "validation_original.csv"
TEST_ORIGINAL_FILENAME = "test_original.csv"
GENERATED_ONLY_FILENAME = "generated_only.csv"
AUGMENTED_DATASET_FILENAME = "augmented_dataset.csv"
SUMMARY_FILENAME = "augmentation_summary.json"
DUPLICATION_COUNTS_FILENAME = "duplication_counts_by_source.csv"
