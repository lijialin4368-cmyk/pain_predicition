"""XGBoost 回归任务配置文件。

你通常只需要改这个文件，不需要直接改训练逻辑。
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
# 4) XGBoost 超参数
# =========================
XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

USE_HYPERPARAM_SEARCH = True
XGB_SEARCH_N_ITER = 16
XGB_SEARCH_CV = 4
XGB_SEARCH_SCORING = "neg_mean_absolute_error"
XGB_SEARCH_PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 700, 900],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

# =========================
# 5) 输出目录配置
# =========================
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
