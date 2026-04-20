from pathlib import Path

# 输入 / 输出
DATA_PATH = Path(__file__).resolve().parents[1] / "data_vectorized.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

# 随机性与切分
RANDOM_STATE = 42
TEST_SIZE = 0.30
GENERATED_KEEP_FRACTION = 1.0
DEDUPLICATE_GENERATED_ROWS = True

# 核心列
GENDER_COL = "性别_num"
AGE_COL = "年龄"
WEIGHT_COL = "体重"
MALE_CODE = 1
AGE_HIGH_RISK_THRESHOLD = 60

REST_PAIN_COLS = [
    "手术当天_静息痛",
    "术后第一天_静息痛",
    "术后第二天_静息痛",
    "术后第三天_静息痛",
]
MOVEMENT_PAIN_COLS = [
    "手术当天_活动痛",
    "术后第一天_活动痛",
    "术后第二天_活动痛",
    "术后第三天_活动痛",
]
ALL_PAIN_COLS = REST_PAIN_COLS + MOVEMENT_PAIN_COLS

# 高痛抬升规则
SAME_DAY_TRIGGER_COLS = [
    "手术当天_静息痛",
    "手术当天_活动痛",
]
SAME_DAY_HIGH_PAIN_THRESHOLD = 3
PAIN_UPSHIFT_DELTA = 1
DEMOGRAPHIC_DELTAS = (-2, -1, 0, 1, 2)
HIGH_PAIN_KEEP_COUNT = 10

# 四天波动规则
WAVE_TRIGGER_THRESHOLD = 4
WAVE_TRIGGER_MIN_COUNT = 2
WAVE_DELTA = 1
PAIN_MIN_VALUE = 0
PAIN_MAX_VALUE = None

# Metadata 列
META_PREFIX = "__meta_"
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
TEST_SPLIT_VALUE = "test"

# 产物文件名
AUGMENTED_DATASET_FILENAME = "augmented_dataset.csv"
TRAIN_ORIGINAL_FILENAME = "train_original.csv"
TEST_ORIGINAL_FILENAME = "test_original.csv"
GENERATED_ONLY_FILENAME = "generated_only.csv"
SUMMARY_FILENAME = "augmentation_summary.json"
RULE_COUNTS_FILENAME = "augmentation_rule_counts.csv"
