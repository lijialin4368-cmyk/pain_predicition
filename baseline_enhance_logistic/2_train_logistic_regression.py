import os
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

OUTCOME_DAYS = ["手术当天", "术后第一天", "术后第二天", "术后第三天"]
OUTCOME_METRICS = ["静息痛", "活动痛", "镇静评分", "活动状态", "恶心呕吐"]
SHARED_BACKBONE_METRICS = {"静息痛", "活动痛", "镇静评分", "恶心呕吐"}
PAIN_TYPES = {"rest": "静息痛", "movement": "活动痛"}

DAY_EN = {
    "手术当天": "SurgeryDay",
    "术后第一天": "POD1",
    "术后第二天": "POD2",
    "术后第三天": "POD3",
}

METRIC_EN = {
    "静息痛": "RestPain",
    "活动痛": "MovementPain",
    "镇静评分": "SedationScore",
    "活动状态": "ActivityStatus",
    "恶心呕吐": "NauseaVomiting",
}

METRIC_BINARY_SPECS = {
    "静息痛": {
        "positive_min": 4.0,
        "low_label": "Class 0 (0-3)",
        "high_label": "Class 1 (4-10)",
    },
    "活动痛": {
        "positive_min": 4.0,
        "low_label": "Class 0 (0-3)",
        "high_label": "Class 1 (4-10)",
    },
    "镇静评分": {
        "positive_min": 3.0,
        "low_label": "Class 0 (1-2)",
        "high_label": "Class 1 (3-5)",
    },
    "活动状态": {
        "positive_min": 3.0,
        "low_label": "Class 0 (1-2)",
        "high_label": "Class 1 (3-4)",
    },
    "恶心呕吐": {
        "positive_min": 2.0,
        "low_label": "Class 0 (0-1)",
        "high_label": "Class 1 (2-3)",
    },
}

TOTAL_OUTPUT_FILES = {
    "prediction_overview_all_targets.csv",
    "confusion_matrix_prob_all_targets.csv",
    "confusion_matrix_prob_all_targets.png",
    "training_acc_all_targets.png",
}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    baseline_dir = script_dir.parent
    project_dir = baseline_dir.parent

    default_input_candidates = [
        project_dir / "data_vectorized.csv",
        baseline_dir / "data_vectorized.csv",
        script_dir / "data_vectorized.csv",
    ]
    default_input = next((p for p in default_input_candidates if p.exists()), default_input_candidates[0])

    parser = argparse.ArgumentParser(
        description="Train enhanced 3-layer logistic network model(s) for outcome binary classification."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Input CSV path.")
    parser.add_argument(
        "--day",
        type=str,
        default="all",
        help='Outcome day prefix, e.g. "术后第一天"; use "all" for all days.',
    )
    parser.add_argument(
        "--pain-type",
        type=str,
        choices=["rest", "movement", "both", "all"],
        default="all",
        help="rest/movement/both for pain-only, or all for all metrics.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["strict", "all", "temporal", "strong_signal_temporal"],
        default="strong_signal_temporal",
        help=(
            "strict: drop all outcomes from features; "
            "all: use all columns except current target (may leak); "
            "temporal/strong_signal_temporal: use non-outcomes + all earlier-day outcomes."
        ),
    )
    parser.add_argument(
        "--feature-impute",
        type=str,
        choices=["median", "zero"],
        default="median",
        help="How to impute missing feature values.",
    )
    parser.add_argument(
        "--pain-threshold",
        type=float,
        default=4.0,
        help="Backward-compatible threshold for pain metrics (0-3 vs >=4).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["focal", "bce"],
        default="focal",
        help="Training loss type. focal is default for imbalanced labels.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help="Positive-class alpha in focal loss (0,1).",
    )
    parser.add_argument("--hidden-size-1", type=int, default=96, help="Hidden layer 1 width.")
    parser.add_argument("--hidden-size-2", type=int, default=48, help="Hidden layer 2 width.")
    parser.add_argument("--patience", type=int, default=120, help="Early stop patience on train loss.")
    parser.add_argument(
        "--positive-weight-mode",
        type=str,
        choices=["balanced", "none", "sqrt_balanced"],
        default="balanced",
        help="Class weighting for positive class in training batches.",
    )
    parser.add_argument(
        "--disable-high-pain-oversampling",
        action="store_true",
        help="Disable oversampling for high-pain samples (> high-pain-score-threshold). Recommended with focal loss.",
    )
    parser.add_argument(
        "--high-pain-oversample-factor",
        type=float,
        default=2.5,
        help="Oversampling factor for high-pain samples in train set.",
    )
    parser.add_argument(
        "--high-pain-loss-weight",
        type=float,
        default=2.0,
        help="Extra loss weight for high-pain samples in train set.",
    )
    parser.add_argument(
        "--high-pain-score-threshold",
        type=float,
        default=3.0,
        help="Samples with score > threshold are treated as high pain for enhancement.",
    )
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        choices=["fixed", "tune", "day_relaxed", "task_adaptive"],
        default="task_adaptive",
        help=(
            "fixed: use --decision-threshold; "
            "tune: choose threshold on validation set; "
            "day_relaxed: use day-specific thresholds from --day-thresholds; "
            "task_adaptive: choose per-task threshold to maximize accuracy while keeping recall near baseline."
        ),
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for fixed mode.")
    parser.add_argument(
        "--day-thresholds",
        type=str,
        default="0.55,0.50,0.45,0.40",
        help="Comma-separated thresholds for [SurgeryDay,POD1,POD2,POD3], progressively relaxed.",
    )
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split from train set for threshold tuning.")
    parser.add_argument("--fpr-reduction-target", type=float, default=0.35, help="Target FPR reduction ratio vs threshold=0.5 on validation.")
    parser.add_argument("--acc-drop-tolerance", type=float, default=0.02, help="Max allowed validation accuracy drop when tuning threshold.")
    parser.add_argument("--min-recall-ratio", type=float, default=0.75, help="Keep recall above baseline_recall * ratio when tuning.")
    parser.add_argument(
        "--recall-keep-ratio",
        type=float,
        default=1.0,
        help="For task_adaptive: keep recall >= baseline_recall * ratio.",
    )
    parser.add_argument(
        "--recall-keep-abs-drop",
        type=float,
        default=0.0,
        help="For task_adaptive: keep recall >= baseline_recall - abs_drop.",
    )
    parser.add_argument(
        "--task-adaptive-min-val-pos",
        type=int,
        default=8,
        help="For task_adaptive: if validation positives < this count, keep baseline threshold.",
    )
    parser.add_argument(
        "--task-adaptive-max-shift",
        type=float,
        default=0.12,
        help="For task_adaptive: max absolute shift from baseline threshold.",
    )
    parser.add_argument("--threshold-grid-size", type=int, default=181, help="Threshold candidates in [0.05, 0.95].")
    parser.add_argument("--fp-penalty", type=float, default=0.25, help="Fallback threshold objective penalty on FPR.")
    parser.add_argument(
        "--clinical-cost-weight",
        type=float,
        default=0.0,
        help="Extra penalty weight for missing high-pain positives; 0 disables clinical cost.",
    )
    parser.add_argument(
        "--clinical-high-pain-threshold",
        type=float,
        default=7.0,
        help="Pain score >= this value is treated as high-pain for clinical cost.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output_prediction_report_en",
        help="Directory to save confusion matrices and optional predictions.",
    )
    return parser.parse_args()


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Cannot read CSV (file not found): {path}")
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {path}")


def get_outcome_columns(df: pd.DataFrame):
    cols = []
    for day in OUTCOME_DAYS:
        for metric in OUTCOME_METRICS:
            col = f"{day}_{metric}"
            if col in df.columns:
                cols.append(col)
    return cols


def get_outcome_day_index(col: str):
    for i, day in enumerate(OUTCOME_DAYS):
        if col.startswith(f"{day}_"):
            return i
    return None


def target_to_english_name(target_col: str):
    for day in OUTCOME_DAYS:
        prefix = f"{day}_"
        if target_col.startswith(prefix):
            metric = target_col[len(prefix) :]
            day_en = DAY_EN.get(day, day)
            metric_en = METRIC_EN.get(metric, metric)
            return f"{day_en}_{metric_en}"
    return target_col


def split_day_metric(target_col: str):
    for day in OUTCOME_DAYS:
        prefix = f"{day}_"
        if target_col.startswith(prefix):
            return day, target_col[len(prefix) :]
    return "", target_col


def is_rest_or_movement_target(target_col: str):
    _, metric = split_day_metric(target_col)
    return metric in ("静息痛", "活动痛")


def get_metric_binary_spec(metric: str):
    if metric not in METRIC_BINARY_SPECS:
        raise ValueError(f"Unsupported metric for binary classification: {metric}")
    return METRIC_BINARY_SPECS[metric]


def make_binary_target(target_col: str, y_score: np.ndarray, args):
    _, metric = split_day_metric(target_col)
    spec = get_metric_binary_spec(metric)

    positive_min = float(spec["positive_min"])
    if metric in ("静息痛", "活动痛"):
        positive_min = float(args.pain_threshold)

    y_bin = (y_score >= positive_min).astype(int)
    low_label = spec["low_label"]
    high_label = spec["high_label"]
    return y_bin, positive_min, low_label, high_label


def select_feature_columns(df: pd.DataFrame, target_col: str, feature_mode: str):
    outcome_cols = get_outcome_columns(df)

    if feature_mode == "strict":
        if target_col in outcome_cols:
            return [c for c in df.columns if c not in outcome_cols]
        return [c for c in df.columns if c != target_col]

    if feature_mode == "all":
        return [c for c in df.columns if c != target_col]

    if feature_mode not in ("temporal", "strong_signal_temporal"):
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    # temporal / strong_signal_temporal mode:
    # use all non-outcome features + all outcome features from earlier days.
    # Example: predict POD2 -> include SurgeryDay + POD1 outcomes as strong signals.
    target_day_idx = get_outcome_day_index(target_col)
    feature_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if c not in outcome_cols:
            feature_cols.append(c)
            continue
        col_day_idx = get_outcome_day_index(c)
        if target_day_idx is not None and col_day_idx is not None and col_day_idx < target_day_idx:
            feature_cols.append(c)
    return feature_cols


def prepare_features(df: pd.DataFrame, feature_cols, feature_impute: str):
    x = df[feature_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    if feature_impute == "median":
        x = x.fillna(x.median(numeric_only=True))
        x = x.fillna(0.0)
    else:
        x = x.fillna(0.0)
    return x


def split_train_test_stratified(y: np.ndarray, test_size: float, random_state: int):
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if len(y) < 2:
        raise ValueError("Need at least 2 samples for train/test split.")

    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Target has only one class after dropping missing values.")

    train_idx = []
    test_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test_c = int(round(len(idx) * test_size))
        n_test_c = max(1, n_test_c)
        n_test_c = min(n_test_c, len(idx) - 1)
        test_idx.extend(idx[:n_test_c].tolist())
        train_idx.extend(idx[n_test_c:].tolist())

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Invalid split result. Try adjusting test_size.")

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def split_train_val_stratified(y: np.ndarray, val_size: float, random_state: int):
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be in [0,1), got {val_size}")
    if val_size == 0.0:
        idx = np.arange(len(y))
        return idx, np.array([], dtype=int)

    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    train_idx = []
    val_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val_c = int(round(len(idx) * val_size))
        n_val_c = max(1, n_val_c)
        n_val_c = min(n_val_c, len(idx) - 1)
        val_idx.extend(idx[:n_val_c].tolist())
        train_idx.extend(idx[n_val_c:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def sigmoid(z: np.ndarray):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def relu(x: np.ndarray):
    return np.maximum(x, 0.0)


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray):
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def weighted_binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray):
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    sw = np.clip(sample_weight.astype(float), 0.0, None)
    sw_sum = float(np.sum(sw))
    if sw_sum <= 0.0:
        return binary_log_loss(y_true, y_prob)
    loss_vec = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.sum(loss_vec * sw) / sw_sum)


def build_high_pain_training_weights(args, target_col: str, train_scores: np.ndarray):
    n = len(train_scores)
    loss_w = np.ones(n, dtype=float)
    over_w = np.ones(n, dtype=float)
    high_mask = np.zeros(n, dtype=bool)

    if not is_rest_or_movement_target(target_col):
        return loss_w, over_w, high_mask

    high_mask = train_scores > args.high_pain_score_threshold
    if np.any(high_mask):
        loss_w[high_mask] = max(1.0, float(args.high_pain_loss_weight))
        over_w[high_mask] = max(1.0, float(args.high_pain_oversample_factor))

    return loss_w, over_w, high_mask


def init_three_layer_params(n_features: int, hidden_size_1: int, hidden_size_2: int, rng: np.random.Generator):
    h1 = max(4, int(hidden_size_1))
    h2 = max(4, int(hidden_size_2))
    w1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, h1))
    b1 = np.zeros(h1, dtype=float)
    w2 = rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2))
    b2 = np.zeros(h2, dtype=float)
    w3 = rng.normal(0.0, np.sqrt(2.0 / h2), size=(h2, 1))
    b3 = np.zeros(1, dtype=float)
    return w1, b1, w2, b2, w3, b3


def forward_three_layer(x: np.ndarray, params):
    w1, b1, w2, b2, w3, b3 = params
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    logits = (a2 @ w3 + b3).reshape(-1)
    prob = sigmoid(logits)
    return prob, (z1, a1, z2, a2)


def build_shared_feature_columns(df: pd.DataFrame, target_cols, feature_mode: str):
    if len(target_cols) == 0:
        return []
    per_target_cols = [set(select_feature_columns(df, target_col=t, feature_mode=feature_mode)) for t in target_cols]
    shared = set.intersection(*per_target_cols)
    # keep original column order for stable behavior
    return [c for c in df.columns if c in shared]


def split_train_test_multitask(y_mat: np.ndarray, test_size: float, random_state: int, max_tries: int = 128):
    n = y_mat.shape[0]
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if n < 2:
        raise ValueError("Need at least 2 samples for split.")

    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)
    rng = np.random.default_rng(random_state)

    for _ in range(max_tries):
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        ok = True
        for h in range(y_mat.shape[1]):
            y_tr = y_mat[train_idx, h]
            y_te = y_mat[test_idx, h]
            y_tr = y_tr[~np.isnan(y_tr)].astype(int)
            y_te = y_te[~np.isnan(y_te)].astype(int)

            if len(y_tr) == 0 or len(np.unique(y_tr)) < 2:
                ok = False
                break
            if len(y_te) == 0:
                ok = False
                break

        if ok:
            return train_idx.astype(int), test_idx.astype(int)

    raise ValueError("Could not create a valid train/test split for multi-head labels.")


def split_train_val_multitask(y_train_mat: np.ndarray, val_size: float, random_state: int, max_tries: int = 128):
    n = y_train_mat.shape[0]
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be in [0,1), got {val_size}")
    if val_size == 0.0:
        idx = np.arange(n, dtype=int)
        return idx, np.array([], dtype=int)

    n_val = max(1, int(round(n * val_size)))
    n_val = min(n_val, n - 1)
    rng = np.random.default_rng(random_state)

    for _ in range(max_tries):
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        ok = True
        for h in range(y_train_mat.shape[1]):
            y_tr = y_train_mat[tr_idx, h]
            y_tr = y_tr[~np.isnan(y_tr)].astype(int)
            if len(y_tr) == 0 or len(np.unique(y_tr)) < 2:
                ok = False
                break
        if ok:
            return tr_idx.astype(int), val_idx.astype(int)

    idx = np.arange(n, dtype=int)
    return idx, np.array([], dtype=int)


def forward_multi_head(x: np.ndarray, params):
    w1, b1, w2, b2, w3, b3 = params
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    logits = a2 @ w3 + b3
    prob = sigmoid(logits)
    return prob, (z1, a1, z2, a2)


def compute_binary_element_grad(y_elem: np.ndarray, p_elem: np.ndarray, mask_elem: np.ndarray, args):
    eps = 1e-12
    y = y_elem
    p = np.clip(p_elem, eps, 1.0 - eps)
    z_grad = np.zeros_like(p, dtype=float)
    loss_elem = np.zeros_like(p, dtype=float)

    if args.loss_type == "focal":
        alpha = float(np.clip(args.focal_alpha, 1e-6, 1.0 - 1e-6))
        gamma = max(0.0, float(args.focal_gamma))

        pos = y == 1.0
        neg = y == 0.0

        if np.any(pos):
            pp = p[pos]
            loss_elem[pos] = -alpha * np.power(1.0 - pp, gamma) * np.log(pp)
            dldp = alpha * (gamma * np.power(1.0 - pp, gamma - 1.0) * np.log(pp) - np.power(1.0 - pp, gamma) / pp)
            z_grad[pos] = dldp * pp * (1.0 - pp)

        if np.any(neg):
            pn = p[neg]
            a_neg = 1.0 - alpha
            loss_elem[neg] = -a_neg * np.power(pn, gamma) * np.log(1.0 - pn)
            dldp = a_neg * (-gamma * np.power(pn, gamma - 1.0) * np.log(1.0 - pn) + np.power(pn, gamma) / (1.0 - pn))
            z_grad[neg] = dldp * pn * (1.0 - pn)
    else:
        loss_elem = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        z_grad = p - y

    z_grad[~mask_elem] = 0.0
    loss_elem[~mask_elem] = 0.0
    return loss_elem, z_grad


def fit_shared_backbone_multitask(
    x_train: np.ndarray,
    y_train_mat: np.ndarray,
    raw_score_train_mat: np.ndarray,
    target_cols,
    args,
):
    n_samples, n_features = x_train.shape
    n_heads = y_train_mat.shape[1]
    rng = np.random.default_rng(args.random_state)

    w1, b1, w2, b2, _, _ = init_three_layer_params(n_features, args.hidden_size_1, args.hidden_size_2, rng)
    w3 = rng.normal(0.0, np.sqrt(2.0 / w2.shape[1]), size=(w2.shape[1], n_heads))
    b3 = np.zeros(n_heads, dtype=float)
    params = (w1, b1, w2, b2, w3, b3)

    label_mask = ~np.isnan(y_train_mat)
    y_filled = np.where(label_mask, y_train_mat, 0.0).astype(float)
    raw_filled = np.where(np.isnan(raw_score_train_mat), -np.inf, raw_score_train_mat)

    pos_cnt = np.sum((y_filled == 1.0) & label_mask, axis=0).astype(float)
    neg_cnt = np.sum((y_filled == 0.0) & label_mask, axis=0).astype(float)
    if args.positive_weight_mode == "balanced":
        pos_weight_vec = np.minimum(neg_cnt / (pos_cnt + 1e-8), 10.0)
    elif args.positive_weight_mode == "sqrt_balanced":
        pos_weight_vec = np.minimum(np.sqrt(neg_cnt / (pos_cnt + 1e-8)), 10.0)
    else:
        pos_weight_vec = np.ones(n_heads, dtype=float)
    pos_weight_vec = np.where(pos_cnt > 0.0, pos_weight_vec, 1.0).astype(float)

    # With focal loss, oversampling is usually unnecessary; keep disabled by default.
    use_high_pain_oversampling = (not args.disable_high_pain_oversampling) and (args.loss_type != "focal")
    oversample_weight = np.ones(n_samples, dtype=float)
    if use_high_pain_oversampling:
        for h, tcol in enumerate(target_cols):
            if is_rest_or_movement_target(tcol):
                high_mask = raw_filled[:, h] > args.high_pain_score_threshold
                oversample_weight[high_mask] = np.maximum(oversample_weight[high_mask], args.high_pain_oversample_factor)
        oversample_weight = np.clip(oversample_weight, 1.0, None)
    over_prob = oversample_weight / np.sum(oversample_weight)

    best_loss = np.inf
    best_params = tuple(p.copy() for p in params)
    stale = 0
    history = []
    batch_size = max(1, min(int(args.batch_size), n_samples))

    for epoch in range(1, int(args.epochs) + 1):
        if use_high_pain_oversampling:
            order = rng.choice(n_samples, size=n_samples, replace=True, p=over_prob)
        else:
            order = rng.permutation(n_samples)

        w1, b1, w2, b2, w3, b3 = params
        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_filled[idx]
            mb = label_mask[idx]
            rb = raw_filled[idx]

            prob, (z1, a1, z2, a2) = forward_multi_head(xb, params)
            loss_elem, dz = compute_binary_element_grad(yb, prob, mb, args)
            class_w = np.where(yb == 1.0, pos_weight_vec.reshape(1, -1), 1.0) * mb.astype(float)
            loss_elem *= class_w
            dz *= class_w

            if args.clinical_cost_weight > 0.0:
                for h, tcol in enumerate(target_cols):
                    if not is_rest_or_movement_target(tcol):
                        continue
                    high_pos = mb[:, h] & (yb[:, h] == 1.0) & (rb[:, h] >= args.clinical_high_pain_threshold)
                    if np.any(high_pos):
                        p_h = np.clip(prob[high_pos, h], 1e-12, 1.0 - 1e-12)
                        extra_loss = args.clinical_cost_weight * np.square(1.0 - p_h)
                        loss_elem[high_pos, h] += extra_loss
                        dz[high_pos, h] += (-2.0 * args.clinical_cost_weight * (1.0 - p_h)) * p_h * (1.0 - p_h)

            weight = mb.astype(float)
            weight_sum = float(np.sum(weight))
            if weight_sum <= 0.0:
                continue
            dz = dz * weight / weight_sum

            grad_w3 = a2.T @ dz + float(args.l2) * w3
            grad_b3 = dz.sum(axis=0)
            da2 = dz @ w3.T
            dz2 = da2 * (z2 > 0.0)
            grad_w2 = a1.T @ dz2 + float(args.l2) * w2
            grad_b2 = dz2.sum(axis=0)
            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0.0)
            grad_w1 = xb.T @ dz1 + float(args.l2) * w1
            grad_b1 = dz1.sum(axis=0)

            lr = float(args.learning_rate)
            w3 -= lr * grad_w3
            b3 -= lr * grad_b3
            w2 -= lr * grad_w2
            b2 -= lr * grad_b2
            w1 -= lr * grad_w1
            b1 -= lr * grad_b1
            params = (w1, b1, w2, b2, w3, b3)

        prob_all, _ = forward_multi_head(x_train, params)
        loss_elem_all, _ = compute_binary_element_grad(y_filled, prob_all, label_mask, args)
        train_loss = float(np.sum(loss_elem_all) / max(1.0, float(np.sum(label_mask))))
        reg_term = 0.5 * float(args.l2) * float(np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3))
        train_loss += reg_term

        pred_all = (prob_all >= 0.5).astype(float)
        correct = (pred_all == y_filled) & label_mask
        train_acc = float(np.sum(correct) / max(1.0, float(np.sum(label_mask))))
        head_correct = np.sum(correct, axis=0).astype(float)
        head_total = np.sum(label_mask, axis=0).astype(float)
        head_acc = np.divide(head_correct, np.maximum(1.0, head_total))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "head_train_acc": head_acc.tolist(),
            }
        )

        if args.verbose and (epoch == 1 or epoch % 100 == 0 or epoch == int(args.epochs)):
            print(f"[epoch {epoch:4d}] train_loss={train_loss:.6f} train_acc={train_acc:.4f}")

        if best_loss - train_loss > 1e-8:
            best_loss = train_loss
            best_params = tuple(p.copy() for p in params)
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break

    return best_params, best_loss, history, use_high_pain_oversampling


def fit_three_layer_logistic_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_weight: np.ndarray,
    oversample_weight: np.ndarray,
    use_high_pain_oversampling: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    patience: int,
    random_state: int,
    positive_weight_mode: str,
    hidden_size_1: int,
    hidden_size_2: int,
    verbose: bool = False,
):
    n_samples, n_features = x_train.shape
    rng = np.random.default_rng(random_state)
    w1, b1, w2, b2, w3, b3 = init_three_layer_params(n_features, hidden_size_1, hidden_size_2, rng)

    best_loss = np.inf
    best_params = (
        w1.copy(),
        b1.copy(),
        w2.copy(),
        b2.copy(),
        w3.copy(),
        b3.copy(),
    )
    stale = 0
    history = []

    batch_size = max(1, min(batch_size, n_samples))
    over_prob = np.clip(oversample_weight.astype(float), 0.0, None)
    if float(np.sum(over_prob)) <= 0:
        over_prob = np.ones(n_samples, dtype=float)
    over_prob = over_prob / np.sum(over_prob)

    for epoch in range(1, epochs + 1):
        if use_high_pain_oversampling:
            order = rng.choice(n_samples, size=n_samples, replace=True, p=over_prob)
        else:
            order = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            sw = train_sample_weight[idx]

            z1 = xb @ w1 + b1
            a1 = relu(z1)
            z2 = a1 @ w2 + b2
            a2 = relu(z2)
            logits = (a2 @ w3 + b3).reshape(-1)
            prob = sigmoid(logits)

            pos = np.sum(yb == 1)
            neg = np.sum(yb == 0)
            if positive_weight_mode == "balanced":
                pos_weight = min(neg / (pos + 1e-8), 10.0)
            elif positive_weight_mode == "sqrt_balanced":
                pos_weight = min(np.sqrt(neg / (pos + 1e-8)), 10.0)
            else:
                pos_weight = 1.0

            class_weight = np.where(yb == 1, pos_weight, 1.0)
            weight = class_weight * sw
            weight_sum = float(np.sum(weight))
            if weight_sum <= 0.0:
                weight = np.ones_like(weight)
                weight_sum = float(len(weight))

            dlogits = ((prob - yb) * weight / weight_sum).reshape(-1, 1)
            grad_w3 = a2.T @ dlogits + l2 * w3
            grad_b3 = dlogits.sum(axis=0)

            da2 = dlogits @ w3.T
            dz2 = da2 * (z2 > 0.0)
            grad_w2 = a1.T @ dz2 + l2 * w2
            grad_b2 = dz2.sum(axis=0)

            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0.0)
            grad_w1 = xb.T @ dz1 + l2 * w1
            grad_b1 = dz1.sum(axis=0)

            w3 -= learning_rate * grad_w3
            b3 -= learning_rate * grad_b3
            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2
            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1

        train_prob, _ = forward_three_layer(x_train, (w1, b1, w2, b2, w3, b3))
        pos_all = np.sum(y_train == 1)
        neg_all = np.sum(y_train == 0)
        if positive_weight_mode == "balanced":
            pos_weight_all = min(neg_all / (pos_all + 1e-8), 10.0)
        elif positive_weight_mode == "sqrt_balanced":
            pos_weight_all = min(np.sqrt(neg_all / (pos_all + 1e-8)), 10.0)
        else:
            pos_weight_all = 1.0
        class_weight_all = np.where(y_train == 1, pos_weight_all, 1.0)
        eval_weight = train_sample_weight * class_weight_all
        reg_term = 0.5 * l2 * float(np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3))
        train_loss = weighted_binary_log_loss(y_train, train_prob, eval_weight) + reg_term
        train_acc = float(np.mean((train_prob >= 0.5).astype(int) == y_train))
        history.append({"epoch": int(epoch), "train_loss": float(train_loss), "train_acc": train_acc})

        if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == epochs):
            print(f"[epoch {epoch:4d}] train_loss={train_loss:.6f} train_acc={train_acc:.4f}")

        if best_loss - train_loss > 1e-8:
            best_loss = train_loss
            best_params = (
                w1.copy(),
                b1.copy(),
                w2.copy(),
                b2.copy(),
                w3.copy(),
                b3.copy(),
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return best_params, best_loss, history


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = y_true.astype(int)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_prob)
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = compute_auc(y_true, y_prob)
    log_loss = binary_log_loss(y_true, y_prob)
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1),
        "auc": float(auc),
        "log_loss": float(log_loss),
        "brier": float(brier),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "y_pred": y_pred,
    }


def parse_day_thresholds(threshold_text: str):
    parts = [p.strip() for p in str(threshold_text).split(",") if p.strip() != ""]
    if len(parts) != len(OUTCOME_DAYS):
        raise ValueError(
            f"--day-thresholds expects {len(OUTCOME_DAYS)} values for {OUTCOME_DAYS}, got: {threshold_text}"
        )
    values = []
    for p in parts:
        t = float(p)
        if not (0.0 < t < 1.0):
            raise ValueError(f"Each day threshold must be in (0,1), got {t}")
        values.append(t)
    return {day: values[i] for i, day in enumerate(OUTCOME_DAYS)}


def get_day_relaxed_threshold(target_col: str, day_threshold_map: dict, default_threshold: float):
    day, _ = split_day_metric(target_col)
    return float(day_threshold_map.get(day, default_threshold))


def choose_threshold_by_validation(args, y_val: np.ndarray, prob_val: np.ndarray):
    base = classification_metrics(y_val, prob_val, threshold=args.decision_threshold)

    fpr_limit = base["fpr"] * max(0.0, (1.0 - args.fpr_reduction_target))
    acc_floor = base["accuracy"] - args.acc_drop_tolerance
    recall_floor = base["recall"] * args.min_recall_ratio

    candidates = np.linspace(0.05, 0.95, args.threshold_grid_size)
    feasible = []
    for t in candidates:
        m = classification_metrics(y_val, prob_val, threshold=float(t))
        if m["fpr"] <= fpr_limit and m["accuracy"] >= acc_floor and m["recall"] >= recall_floor:
            feasible.append((float(t), m))

    if feasible:
        feasible.sort(
            key=lambda x: (
                x[1]["f1"],
                x[1]["precision"],
                x[1]["recall"],
                x[1]["accuracy"],
                -x[1]["fpr"],
            ),
            reverse=True,
        )
        best_t, best_m = feasible[0]
        return best_t, best_m, base, "constrained_search"

    # Fallback: optimize a soft objective if no candidate satisfies all constraints.
    best = None
    for t in candidates:
        m = classification_metrics(y_val, prob_val, threshold=float(t))
        score = m["f1"] + 0.20 * m["precision"] + 0.10 * m["recall"] - args.fp_penalty * m["fpr"]
        if best is None or score > best[0]:
            best = (score, float(t), m)

    return best[1], best[2], base, "fallback_objective"


def choose_threshold_task_adaptive(args, y_val: np.ndarray, prob_val: np.ndarray, base_threshold: float):
    pos_count = int(np.sum(y_val == 1))
    if pos_count < int(args.task_adaptive_min_val_pos):
        base = classification_metrics(y_val, prob_val, threshold=float(base_threshold))
        return float(base_threshold), base, base, "task_adaptive_low_val_pos_keep_base"

    base = classification_metrics(y_val, prob_val, threshold=float(base_threshold))
    base_recall = float(base["recall"])
    recall_floor = max(
        0.0,
        min(
            1.0,
            max(
                base_recall * float(args.recall_keep_ratio),
                base_recall - float(args.recall_keep_abs_drop),
            ),
        ),
    )

    candidates = np.linspace(0.05, 0.95, args.threshold_grid_size)
    candidates = np.unique(np.concatenate([candidates, np.array([float(base_threshold)])]))
    max_shift = max(0.0, float(args.task_adaptive_max_shift))
    candidates = np.array([t for t in candidates if abs(float(t) - float(base_threshold)) <= max_shift + 1e-12], dtype=float)
    if len(candidates) == 0:
        candidates = np.array([float(base_threshold)])
    feasible = []
    for t in candidates:
        m = classification_metrics(y_val, prob_val, threshold=float(t))
        if m["recall"] >= recall_floor:
            feasible.append((float(t), m))

    if len(feasible) > 0:
        feasible.sort(
            key=lambda x: (
                x[1]["accuracy"],
                x[1]["recall"],
                -abs(x[1]["recall"] - base_recall),
                x[1]["precision"],
                -x[1]["fpr"],
            ),
            reverse=True,
        )
        best_t, best_m = feasible[0]

        # If no accuracy gain under recall guard, keep baseline threshold.
        if best_m["accuracy"] + 1e-12 < base["accuracy"]:
            return float(base_threshold), base, base, "task_adaptive_keep_base"
        return best_t, best_m, base, "task_adaptive_recall_guard"

    return float(base_threshold), base, base, "task_adaptive_fallback_base"


def row_normalize_cm(cm: np.ndarray):
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0.0] = 1.0
    return cm.astype(float) / row_sum


def plot_confusion_matrix_prob(cm_prob: np.ndarray, low_label: str, high_label: str, title: str, out_file: Path):
    if not HAS_MATPLOTLIB:
        return False

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_prob, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {low_label}", f"Pred {high_label}"])
    ax.set_yticklabels([f"True {low_label}", f"True {high_label}"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_prob[i, j] * 100:.1f}%", ha="center", va="center", color="black", fontsize=11)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close(fig)
    return True


def plot_training_accuracy(history, title: str, out_file: Path):
    if not HAS_MATPLOTLIB:
        return False
    if len(history) == 0:
        return False

    epochs = [h["epoch"] for h in history]
    accs = [h["train_acc"] for h in history]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(epochs, accs, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close(fig)
    return True


def plot_training_accuracy_grid(results, out_file: Path):
    if not HAS_MATPLOTLIB:
        return False
    if len(results) == 0:
        return False

    n = len(results)
    n_cols = min(4, n)
    n_rows = int(math.ceil(n / n_cols))

    fig_w = max(9.0, 4.8 * n_cols)
    fig_h = max(5.0, 4.0 * n_rows + 0.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for k, result in enumerate(results):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r, c]

        history = result.get("train_history", [])
        if len(history) == 0:
            ax.set_title(result["target_en"], fontsize=11, pad=8)
            ax.text(0.5, 0.5, "No history", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue

        epochs = [h["epoch"] for h in history]
        accs = [h["train_acc"] for h in history]
        ax.plot(epochs, accs, color="#1f77b4", linewidth=1.8)
        ax.set_title(result["target_en"], fontsize=11, pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Train Accuracy", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)

    for k in range(n, n_rows * n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r, c].axis("off")

    fig.suptitle("Training Accuracy Curves (All Targets)", fontsize=14)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    return True


def plot_confusion_matrices_grid(results, out_file: Path):
    if not HAS_MATPLOTLIB:
        return False
    if len(results) == 0:
        return False

    n = len(results)
    n_cols = min(4, n)
    n_rows = int(math.ceil(n / n_cols))

    fig_w = max(9.0, 4.6 * n_cols)
    fig_h = max(4.8, 4.2 * n_rows + 0.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    im = None
    for k, result in enumerate(results):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r, c]

        cm_prob = result["cm_prob"]
        low_label = result["low_label"]
        high_label = result["high_label"]
        im = ax.imshow(cm_prob, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(result["target_en"], fontsize=11, pad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred {low_label}", f"Pred {high_label}"], fontsize=8, rotation=16, ha="right")
        ax.set_yticklabels([f"True {low_label}", f"True {high_label}"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_prob[i, j] * 100:.1f}%", ha="center", va="center", color="black", fontsize=10)

    for k in range(n, n_rows * n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r, c].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
        cbar.set_label("Row Probability", fontsize=10)

    fig.suptitle("Confusion Matrices (Row-normalized Probabilities)", fontsize=14)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    return True


def save_combined_confusion_csv(results, out_file: Path):
    rows = []
    for result in results:
        cm_prob = result["cm_prob"]
        cm_count = result["cm_count"]
        m = result["metrics"]
        rows.append(
            {
                "target": result["target_en"],
                "class_0_label": result["low_label"],
                "class_1_label": result["high_label"],
                "p_true_low_pred_low": float(cm_prob[0, 0]),
                "p_true_low_pred_high": float(cm_prob[0, 1]),
                "p_true_high_pred_low": float(cm_prob[1, 0]),
                "p_true_high_pred_high": float(cm_prob[1, 1]),
                "count_true_low_pred_low": int(cm_count[0, 0]),
                "count_true_low_pred_high": int(cm_count[0, 1]),
                "count_true_high_pred_low": int(cm_count[1, 0]),
                "count_true_high_pred_high": int(cm_count[1, 1]),
                "auc": float(m["auc"]),
                "accuracy": float(m["accuracy"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "specificity": float(m["specificity"]),
                "fpr": float(m["fpr"]),
                "f1": float(m["f1"]),
                "log_loss": float(m["log_loss"]),
                "brier": float(m["brier"]),
            }
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_file, index=False, encoding="utf-8-sig")
    return out_file


def run_shared_backbone_targets(args, df: pd.DataFrame, targets):
    if len(targets) == 0:
        raise ValueError("No targets provided for shared-backbone training.")

    shared_feature_cols = build_shared_feature_columns(df, targets, args.feature_mode)
    if len(shared_feature_cols) == 0:
        raise ValueError("No shared feature columns found across selected targets.")

    y_score_cols = []
    y_bin_cols = []
    for t in targets:
        y_raw = pd.to_numeric(df[t], errors="coerce")
        y_np = y_raw.to_numpy(dtype=float)
        y_score_cols.append(y_np)

        y_bin, _, _, _ = make_binary_target(t, np.nan_to_num(y_np, nan=-1e9), args)
        y_bin = y_bin.astype(float)
        y_bin[np.isnan(y_np)] = np.nan
        y_bin_cols.append(y_bin)

    y_score_mat = np.stack(y_score_cols, axis=1)
    y_bin_mat = np.stack(y_bin_cols, axis=1)

    any_label_mask = np.any(~np.isnan(y_bin_mat), axis=1)
    if not np.any(any_label_mask):
        raise ValueError("All selected targets are missing for all rows.")

    x_df = prepare_features(df.loc[any_label_mask].copy(), shared_feature_cols, args.feature_impute)
    y_score_mat = y_score_mat[any_label_mask]
    y_bin_mat = y_bin_mat[any_label_mask]

    train_idx, test_idx = split_train_test_multitask(y_bin_mat, args.test_size, args.random_state)
    x_all = x_df.to_numpy(dtype=float)
    x_train_all, x_test = x_all[train_idx], x_all[test_idx]
    y_train_all, y_test = y_bin_mat[train_idx], y_bin_mat[test_idx]
    score_train_all, score_test = y_score_mat[train_idx], y_score_mat[test_idx]

    tr_sub_idx, val_idx = split_train_val_multitask(y_train_all, args.val_size, args.random_state + 17)
    x_train, y_train = x_train_all[tr_sub_idx], y_train_all[tr_sub_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]
    score_train = score_train_all[tr_sub_idx]

    x_train_std, x_test_std = standardize_train_test(x_train, x_test)
    _, x_val_std = standardize_train_test(x_train, x_val)

    best_params, train_loss, train_history, oversampling_used = fit_shared_backbone_multitask(
        x_train=x_train_std,
        y_train_mat=y_train,
        raw_score_train_mat=score_train,
        target_cols=targets,
        args=args,
    )

    test_prob_mat, _ = forward_multi_head(x_test_std, best_params)
    val_prob_mat, _ = forward_multi_head(x_val_std, best_params) if len(val_idx) > 0 else (np.zeros((0, len(targets))), None)

    results = []
    for h, target_col in enumerate(targets):
        y_test_h = y_test[:, h]
        test_mask_h = ~np.isnan(y_test_h)
        if not np.any(test_mask_h):
            continue
        y_test_vec = y_test_h[test_mask_h].astype(int)
        prob_test_vec = test_prob_mat[test_mask_h, h]

        y_val_h = y_val[:, h] if len(val_idx) > 0 else np.array([])
        val_mask_h = (~np.isnan(y_val_h)) if len(val_idx) > 0 else np.array([], dtype=bool)
        y_val_vec = y_val_h[val_mask_h].astype(int) if len(val_idx) > 0 else np.array([])
        prob_val_vec = val_prob_mat[val_mask_h, h] if len(val_idx) > 0 else np.array([])

        if args.threshold_strategy == "day_relaxed":
            chosen_threshold = get_day_relaxed_threshold(target_col, args.day_threshold_map, args.decision_threshold)
            val_base = (
                classification_metrics(y_val_vec, prob_val_vec, threshold=args.decision_threshold)
                if len(y_val_vec) > 0
                else None
            )
            val_selected = (
                classification_metrics(y_val_vec, prob_val_vec, threshold=chosen_threshold) if len(y_val_vec) > 0 else None
            )
            threshold_selection_mode = "day_relaxed"
        elif args.threshold_strategy == "task_adaptive" and len(y_val_vec) > 0 and len(np.unique(y_val_vec)) >= 2:
            base_threshold = get_day_relaxed_threshold(target_col, args.day_threshold_map, args.decision_threshold)
            chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_task_adaptive(
                args, y_val_vec, prob_val_vec, base_threshold=base_threshold
            )
        elif args.threshold_strategy == "fixed" or len(y_val_vec) == 0 or len(np.unique(y_val_vec)) < 2:
            chosen_threshold = args.decision_threshold
            val_base = (
                classification_metrics(y_val_vec, prob_val_vec, threshold=args.decision_threshold)
                if len(y_val_vec) > 0
                else None
            )
            val_selected = (
                classification_metrics(y_val_vec, prob_val_vec, threshold=chosen_threshold) if len(y_val_vec) > 0 else None
            )
            threshold_selection_mode = "fixed_threshold"
        else:
            chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_by_validation(
                args, y_val_vec, prob_val_vec
            )

        metrics = classification_metrics(y_test_vec, prob_test_vec, threshold=chosen_threshold)
        cm_count = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
        cm_prob = row_normalize_cm(cm_count)
        _, metric = split_day_metric(target_col)
        _, positive_min, low_label, high_label = make_binary_target(target_col, np.array([0.0, 10.0]), args)

        train_label_h = y_train[:, h]
        train_mask_h = ~np.isnan(train_label_h)
        train_score_h = score_train[:, h]
        high_mask_h = np.zeros_like(train_mask_h, dtype=bool)
        if is_rest_or_movement_target(target_col):
            high_mask_h = train_mask_h & (train_score_h > args.high_pain_score_threshold)

        task_train_history = []
        for hh in train_history:
            head_acc_list = hh.get("head_train_acc", None)
            if isinstance(head_acc_list, list) and h < len(head_acc_list):
                task_acc = float(head_acc_list[h])
            else:
                task_acc = float(hh.get("train_acc", 0.0))
            task_train_history.append(
                {
                    "epoch": int(hh["epoch"]),
                    "train_loss": float(hh["train_loss"]),
                    "train_acc": task_acc,
                }
            )

        results.append(
            {
                "target_col": target_col,
                "n_total_non_missing": int(np.sum(~np.isnan(y_bin_mat[:, h]))),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_pos_train": int(np.sum(train_label_h[train_mask_h] == 1.0)),
                "n_high_pain_train": int(np.sum(high_mask_h)),
                "n_val": int(np.sum(val_mask_h)) if len(val_idx) > 0 else 0,
                "n_pos_val": int(np.sum(y_val_vec == 1)) if len(y_val_vec) > 0 else 0,
                "n_pos_test": int(np.sum(y_test_vec == 1)),
                "avg_train_sample_weight": 1.0,
                "class_1_min_score": float(positive_min),
                "low_label": low_label,
                "high_label": high_label,
                "feature_count": int(len(shared_feature_cols)),
                "train_loss": float(train_loss),
                "chosen_threshold": float(chosen_threshold),
                "threshold_selection_mode": threshold_selection_mode,
                "val_metrics_base": val_base,
                "val_metrics_selected": val_selected,
                "metrics": metrics,
                "target_en": target_to_english_name(target_col),
                "cm_count": cm_count,
                "cm_prob": cm_prob,
                "train_history": task_train_history,
                "metric_name": metric,
                "oversampling_used": bool(oversampling_used),
            }
        )

    if len(results) == 0:
        raise ValueError("No evaluable heads found on test split.")

    return results


def run_one_target(args, df: pd.DataFrame, target_col: str, output_dir: Path):
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    feature_cols = select_feature_columns(df, target_col=target_col, feature_mode=args.feature_mode)

    x_df = prepare_features(df, feature_cols, args.feature_impute)
    y_raw = pd.to_numeric(df[target_col], errors="coerce")

    # Requirement #2: drop samples with missing outcomes before training/batching.
    valid_mask = y_raw.notna()
    x_df = x_df.loc[valid_mask].reset_index(drop=True)
    y_score = y_raw.loc[valid_mask].to_numpy(dtype=float)
    y_bin, positive_min, low_label, high_label = make_binary_target(target_col, y_score, args)

    if len(np.unique(y_bin)) < 2:
        raise ValueError(f"Target {target_col} has only one class after missing drop.")

    x = x_df.to_numpy(dtype=float)
    train_idx, test_idx = split_train_test_stratified(y_bin, args.test_size, args.random_state)
    x_train_all, x_test = x[train_idx], x[test_idx]
    y_train_all, y_test = y_bin[train_idx], y_bin[test_idx]
    score_train_all = y_score[train_idx]

    # Validation split for threshold tuning to reduce false positives without overfitting test.
    tr_sub_idx, val_idx = split_train_val_stratified(y_train_all, args.val_size, args.random_state + 17)
    x_train, y_train = x_train_all[tr_sub_idx], y_train_all[tr_sub_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]
    score_train = score_train_all[tr_sub_idx]
    train_sample_weight, oversample_weight, high_mask_train = build_high_pain_training_weights(
        args, target_col=target_col, train_scores=score_train
    )

    x_train_std, x_test_std = standardize_train_test(x_train, x_test)
    _, x_val_std = standardize_train_test(x_train, x_val)
    best_params, train_loss, train_history = fit_three_layer_logistic_batch(
        x_train=x_train_std,
        y_train=y_train,
        train_sample_weight=train_sample_weight,
        oversample_weight=oversample_weight,
        use_high_pain_oversampling=(not args.disable_high_pain_oversampling),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        random_state=args.random_state,
        positive_weight_mode=args.positive_weight_mode,
        hidden_size_1=args.hidden_size_1,
        hidden_size_2=args.hidden_size_2,
        verbose=args.verbose,
    )

    test_prob, _ = forward_three_layer(x_test_std, best_params)
    val_prob, _ = forward_three_layer(x_val_std, best_params) if len(y_val) > 0 else (np.array([]), None)

    if args.threshold_strategy == "day_relaxed":
        chosen_threshold = get_day_relaxed_threshold(target_col, args.day_threshold_map, args.decision_threshold)
        val_base = classification_metrics(y_val, val_prob, threshold=args.decision_threshold) if len(y_val) > 0 else None
        val_selected = classification_metrics(y_val, val_prob, threshold=chosen_threshold) if len(y_val) > 0 else None
        threshold_selection_mode = "day_relaxed"
    elif args.threshold_strategy == "task_adaptive" and len(y_val) > 0 and len(np.unique(y_val)) >= 2:
        base_threshold = get_day_relaxed_threshold(target_col, args.day_threshold_map, args.decision_threshold)
        chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_task_adaptive(
            args, y_val, val_prob, base_threshold=base_threshold
        )
    elif args.threshold_strategy == "fixed" or len(y_val) == 0:
        chosen_threshold = args.decision_threshold
        val_base = classification_metrics(y_val, val_prob, threshold=args.decision_threshold) if len(y_val) > 0 else None
        val_selected = classification_metrics(y_val, val_prob, threshold=chosen_threshold) if len(y_val) > 0 else None
        threshold_selection_mode = "fixed_threshold"
    else:
        chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_by_validation(
            args, y_val, val_prob
        )

    metrics = classification_metrics(y_test, test_prob, threshold=chosen_threshold)
    target_en = target_to_english_name(target_col)

    cm_count = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    cm_prob = row_normalize_cm(cm_count)

    return {
        "target_col": target_col,
        "n_total_non_missing": int(len(y_bin)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_pos_train": int(np.sum(y_train)),
        "n_high_pain_train": int(np.sum(high_mask_train)),
        "n_val": int(len(y_val)),
        "n_pos_val": int(np.sum(y_val)) if len(y_val) > 0 else 0,
        "n_pos_test": int(np.sum(y_test)),
        "avg_train_sample_weight": float(np.mean(train_sample_weight)) if len(train_sample_weight) > 0 else 1.0,
        "class_1_min_score": float(positive_min),
        "low_label": low_label,
        "high_label": high_label,
        "feature_count": int(len(feature_cols)),
        "train_loss": float(train_loss),
        "chosen_threshold": float(chosen_threshold),
        "threshold_selection_mode": threshold_selection_mode,
        "val_metrics_base": val_base,
        "val_metrics_selected": val_selected,
        "metrics": metrics,
        "target_en": target_en,
        "cm_count": cm_count,
        "cm_prob": cm_prob,
        "train_history": train_history,
    }


def print_result(result: dict):
    m = result["metrics"]
    print("=" * 82)
    print(f"Target: {result['target_col']} ({result['target_en']})")
    print("=" * 82)
    print(f"Samples (non-missing)    : {result['n_total_non_missing']}")
    print(f"Train / Val / Test       : {result['n_train']} / {result['n_val']} / {result['n_test']}")
    print(f"Pos in Train/Val/Test    : {result['n_pos_train']} / {result['n_pos_val']} / {result['n_pos_test']}")
    print(f"Class split              : {result['low_label']} vs {result['high_label']}")
    print(f"High pain in train (>3)  : {result['n_high_pain_train']}")
    print(f"Avg train sample weight  : {result['avg_train_sample_weight']:.3f}")
    print(f"Feature count            : {result['feature_count']}")
    print(f"Train loss (BCE+L2)      : {result['train_loss']:.6f}")
    print(
        f"Threshold                : {result['chosen_threshold']:.3f} ({result['threshold_selection_mode']})"
    )
    if result["val_metrics_base"] is not None and result["val_metrics_selected"] is not None:
        vb = result["val_metrics_base"]
        vs = result["val_metrics_selected"]
        print(
            "Val @base->selected      : "
            f"acc {vb['accuracy']:.3f}->{vs['accuracy']:.3f}, "
            f"fpr {vb['fpr']:.3f}->{vs['fpr']:.3f}, "
            f"pre {vb['precision']:.3f}->{vs['precision']:.3f}, "
            f"rec {vb['recall']:.3f}->{vs['recall']:.3f}, "
            f"f1 {vb['f1']:.3f}->{vs['f1']:.3f}"
        )
    print("-" * 82)
    print(f"AUC                      : {m['auc']:.4f}")
    print(f"Accuracy                 : {m['accuracy']:.4f}")
    print(f"Precision                : {m['precision']:.4f}")
    print(f"Recall (Sensitivity)     : {m['recall']:.4f}")
    print(f"Specificity              : {m['specificity']:.4f}")
    print(f"FPR                      : {m['fpr']:.4f}")
    print(f"F1 Score                 : {m['f1']:.4f}")
    print(f"Log Loss                 : {m['log_loss']:.4f}")
    print(f"Brier Score              : {m['brier']:.4f}")
    print("-" * 82)
    print(
        "Confusion Matrix Prob "
        f"(rows=true [{result['low_label']}, {result['high_label']}], "
        f"cols=pred [{result['low_label']}, {result['high_label']}])"
    )
    cm_prob = result["cm_prob"]
    print(f"[[{cm_prob[0,0]:.4f}, {cm_prob[0,1]:.4f}],")
    print(f" [{cm_prob[1,0]:.4f}, {cm_prob[1,1]:.4f}]]")
    print("=" * 82)


def cleanup_output_dir_keep_totals(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.iterdir():
        if p.is_file() and p.name not in TOTAL_OUTPUT_FILES:
            p.unlink()


def main():
    args = parse_args()
    args.day_threshold_map = (
        parse_day_thresholds(args.day_thresholds)
        if args.threshold_strategy in ("day_relaxed", "task_adaptive")
        else {}
    )
    df = read_csv_with_fallback(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.day == "all":
        day_candidates = OUTCOME_DAYS
    else:
        if args.day not in OUTCOME_DAYS:
            raise ValueError(f'Unknown day: {args.day}. Use one of {OUTCOME_DAYS} or "all".')
        day_candidates = [args.day]

    if args.pain_type == "rest":
        selected_metrics = ["静息痛"]
    elif args.pain_type == "movement":
        selected_metrics = ["活动痛"]
    elif args.pain_type == "both":
        selected_metrics = ["静息痛", "活动痛"]
    else:
        selected_metrics = list(OUTCOME_METRICS)

    target_candidates = [f"{day}_{metric}" for day in day_candidates for metric in selected_metrics]
    targets = [c for c in target_candidates if c in df.columns]
    missing_targets = [c for c in target_candidates if c not in df.columns]
    if len(targets) == 0:
        raise ValueError("No valid target columns found for current settings.")

    print(f"Input file               : {args.input}")
    print(f"Pain threshold rule      : 0-3 -> class 0, >= {args.pain_threshold} -> class 1")
    print("Other metric bins        : Sedation 1-2/3-5, Activity 1-2/3-4, Nausea 0-1/2-3")
    print("Missing outcome handling : drop missing rows before train/test split and batch training")
    print(f"Positive weight mode     : {args.positive_weight_mode}")
    print(f"Loss type                : {args.loss_type}")
    if args.loss_type == "focal":
        print(f"Focal (alpha,gamma)      : ({args.focal_alpha}, {args.focal_gamma})")
    print(f"Model layers             : 3 (input -> {args.hidden_size_1} -> {args.hidden_size_2} -> sigmoid)")
    print(f"High pain oversampling   : {not args.disable_high_pain_oversampling}")
    print(f"High pain score threshold: > {args.high_pain_score_threshold}")
    print(f"High pain oversample fac.: {args.high_pain_oversample_factor}")
    print(f"High pain loss weight    : {args.high_pain_loss_weight}")
    print(f"Threshold strategy       : {args.threshold_strategy}")
    print(f"Clinical cost weight     : {args.clinical_cost_weight}")
    print(f"Clinical high-pain >=    : {args.clinical_high_pain_threshold}")
    if args.threshold_strategy in ("day_relaxed", "task_adaptive"):
        print(f"Day thresholds           : {args.day_threshold_map}")
    if args.threshold_strategy == "task_adaptive":
        print(
            f"Recall guard             : max(base*{args.recall_keep_ratio}, base-{args.recall_keep_abs_drop})"
        )
        print(
            f"Adaptive guardrails      : min_val_pos={args.task_adaptive_min_val_pos}, max_shift={args.task_adaptive_max_shift}"
        )
    print(f"Feature mode             : {args.feature_mode}")
    if args.feature_mode == "all":
        print("Note                     : Using other outcomes as features may inflate metrics.")
    if args.feature_mode in ("temporal", "strong_signal_temporal"):
        print("Note                     : Temporal strong-signal mode uses all earlier-day outcomes to avoid leakage.")
    print(f"Targets                  : {targets}")
    if missing_targets:
        print(f"Missing targets skipped  : {missing_targets}")
    print(f"Output dir               : {args.output_dir}")
    skipped_not_shared = [t for t in targets if split_day_metric(t)[1] not in SHARED_BACKBONE_METRICS]
    if skipped_not_shared:
        print(f"Skipped (not shared set) : {skipped_not_shared}")
    shared_targets = [t for t in targets if split_day_metric(t)[1] in SHARED_BACKBONE_METRICS]
    print(f"Shared-backbone targets  : {shared_targets}")
    if len(shared_targets) == 0:
        raise ValueError("No targets left for shared-backbone training. Adjust --pain-type/--day settings.")

    all_rows = []
    all_results = run_shared_backbone_targets(args, df, shared_targets)
    if len(all_results) > 0:
        print(f"Oversampling applied     : {all_results[0].get('oversampling_used', False)}")
    for result in all_results:
        print_result(result)

        m = result["metrics"]
        all_rows.append(
            {
                "target": result["target_en"],
                "class_0_label": result["low_label"],
                "class_1_label": result["high_label"],
                "auc": m["auc"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "fpr": m["fpr"],
                "f1": m["f1"],
                "log_loss": m["log_loss"],
                "brier": m["brier"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "n_total_non_missing": result["n_total_non_missing"],
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_test": result["n_test"],
                "chosen_threshold": result["chosen_threshold"],
                "train_loss": result["train_loss"],
                "p_true_low_pred_low": result["cm_prob"][0, 0],
                "p_true_low_pred_high": result["cm_prob"][0, 1],
                "p_true_high_pred_low": result["cm_prob"][1, 0],
                "p_true_high_pred_high": result["cm_prob"][1, 1],
            }
        )

    summary_df = pd.DataFrame(all_rows)
    summary_file = args.output_dir / "prediction_overview_all_targets.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Saved summary metrics    : {summary_file}")

    combined_cm_csv = args.output_dir / "confusion_matrix_prob_all_targets.csv"
    save_combined_confusion_csv(all_results, combined_cm_csv)
    print(f"Saved combined CM CSV    : {combined_cm_csv}")

    combined_cm_fig = args.output_dir / "confusion_matrix_prob_all_targets.png"
    grid_plotted = plot_confusion_matrices_grid(all_results, combined_cm_fig)
    if grid_plotted:
        print(f"Saved combined CM figure : {combined_cm_fig}")
    else:
        print("Saved combined CM figure : skipped (matplotlib unavailable)")

    combined_acc_fig = args.output_dir / "training_acc_all_targets.png"
    acc_grid_plotted = plot_training_accuracy_grid(all_results, combined_acc_fig)
    if acc_grid_plotted:
        print(f"Saved combined ACC figure: {combined_acc_fig}")
    else:
        print("Saved combined ACC figure: skipped (matplotlib unavailable)")

    cleanup_output_dir_keep_totals(args.output_dir)
    print(f"Cleaned output dir       : kept only total CSV/PNG files in {args.output_dir}")


if __name__ == "__main__":
    main()
