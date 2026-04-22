"""标准随机森林回归基线配置。

这个目录专门用于“直接拿原始 data_vectorized.csv 训练随机森林”，
不依赖增强数据，也不写入其他实验目录。
"""

from pathlib import Path

# =========================
# 1) 数据路径配置
# =========================
DATA_PATH = Path(__file__).resolve().parents[1] / "data_vectorized.csv"

# =========================
# 2) 目标列配置
# =========================
TARGET_COLUMN = "术后第一天_静息痛"

# 如果你希望手动指定特征列，可在这里填列名列表；默认 None 表示自动使用“除目标列外的所有列”。
MANUAL_FEATURE_COLUMNS = None

# 是否启用“时间防泄漏”过滤。
ENABLE_TEMPORAL_FILTER = True

# True 表示严格只用过去时点；False 表示允许用到目标当天时点。
STRICT_PAST_ONLY = True

# =========================
# 3) 训练/验证拆分配置
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# 4) 随机森林超参数
# =========================
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
}

USE_HYPERPARAM_SEARCH = True
RF_SEARCH_N_ITER = 16
RF_SEARCH_CV = 4
RF_SEARCH_SCORING = "neg_mean_absolute_error"
RF_SEARCH_PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [None, 8, 12, 16, 24],
    "min_samples_split": [2, 4, 8, 12],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
    "max_samples": [None, 0.7, 0.85],
}

# =========================
# 5) 输出目录配置
# =========================
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
