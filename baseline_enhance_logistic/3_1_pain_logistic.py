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
PAIN_METRICS = ["静息痛", "活动痛"]
PAIN_TYPES = {"rest": "静息痛", "movement": "活动痛"}
PAIN_SCORE_MAX = 10.0
DAY_SENSITIVITY_SCALE = {
    "手术当天": 1.35,
    "术后第一天": 1.25,
    "术后第二天": 1.12,
    "术后第三天": 1.08,
}

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

TOTAL_OUTPUT_FILES = {
    "prediction_overview_all_targets.csv",
    "confusion_matrix_prob_all_targets.csv",
    "confusion_matrix_prob_all_targets.png",
    "training_acc_all_targets.png",
    "training_loss_all_targets.png",
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
        description=(
            "Train pain-only score-aware logistic network model(s): regress raw pain scores, "
            "add high-pain auxiliary classification loss, then map predicted scores to "
            "high-pain probability for final 0/1 decision."
        )
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
        choices=["rest", "movement", "both"],
        default="both",
        help="rest/movement/both for pain-only targets.",
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
        help="Clinical high-pain threshold used for derived 0/1 labels and probability mapping.",
    )
    parser.add_argument(
        "--prob-temperature",
        type=float,
        default=1.0,
        help="Temperature for mapping predicted score to high-pain probability.",
    )
    parser.add_argument(
        "--aux-cls-weight",
        type=float,
        default=1.0,
        help="Weight for the auxiliary high-pain BCE loss during training.",
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
        choices=["huber", "mse"],
        default="huber",
        help="Score regression loss type.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta on the 0-10 score scale.")
    parser.add_argument("--hidden-size-1", type=int, default=96, help="Hidden layer 1 width.")
    parser.add_argument("--hidden-size-2", type=int, default=48, help="Hidden layer 2 width.")
    parser.add_argument("--patience", type=int, default=120, help="Early stop patience on train loss.")
    parser.add_argument(
        "--positive-weight-mode",
        type=str,
        choices=["balanced", "none", "sqrt_balanced"],
        default="balanced",
        help="Use derived high-pain classes to weight the regression loss.",
    )
    parser.add_argument(
        "--disable-high-pain-oversampling",
        action="store_true",
        help="Disable oversampling for high-pain samples (> high-pain-score-threshold).",
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
        choices=["fixed", "tune", "day_relaxed", "conservative_tune"],
        default="conservative_tune",
        help=(
            "fixed: use --decision-threshold; "
            "tune: validation constrained search; "
            "day_relaxed: use day-specific thresholds from --day-thresholds; "
            "conservative_tune: small validation-based adjustment around day-relaxed baseline."
        ),
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for fixed mode.")
    parser.add_argument(
        "--day-thresholds",
        type=str,
        default="0.55,0.50,0.45,0.40",
        help="Comma-separated thresholds for [SurgeryDay,POD1,POD2,POD3].",
    )
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split from train set for threshold tuning.")
    parser.add_argument("--fpr-reduction-target", type=float, default=0.35, help="Target FPR reduction ratio vs threshold=0.5 on validation.")
    parser.add_argument("--acc-drop-tolerance", type=float, default=0.02, help="Max allowed validation accuracy drop when tuning threshold.")
    parser.add_argument("--min-recall-ratio", type=float, default=0.75, help="Keep recall above baseline_recall * ratio when tuning.")
    parser.add_argument("--threshold-grid-size", type=int, default=181, help="Threshold candidates in [0.05, 0.95].")
    parser.add_argument("--fp-penalty", type=float, default=0.25, help="Fallback threshold objective penalty on FPR.")
    parser.add_argument(
        "--rare-positive-target-count",
        type=float,
        default=48.0,
        help="Heads with positive count below this target receive extra rare-positive sensitivity boost.",
    )
    parser.add_argument(
        "--rare-positive-boost-power",
        type=float,
        default=0.5,
        help="Power used to scale rare-positive sensitivity from positive-count shortage.",
    )
    parser.add_argument(
        "--rare-positive-max-boost",
        type=float,
        default=3.0,
        help="Maximum multiplicative boost for rare-positive head weighting.",
    )
    parser.add_argument(
        "--rare-positive-oversample-boost",
        type=float,
        default=1.5,
        help="Extra oversampling multiplier for samples positive on rare heads.",
    )
    parser.add_argument(
        "--conservative-min-val-pos",
        type=int,
        default=10,
        help="For conservative_tune: if validation positives < this count, keep day-relaxed baseline.",
    )
    parser.add_argument(
        "--conservative-max-shift",
        type=float,
        default=0.05,
        help="For conservative_tune: max absolute threshold shift from day-relaxed baseline.",
    )
    parser.add_argument(
        "--conservative-min-acc-gain",
        type=float,
        default=0.003,
        help="For conservative_tune: minimum validation accuracy gain required to change threshold.",
    )
    parser.add_argument(
        "--conservative-max-recall-drop",
        type=float,
        default=0.02,
        help="For conservative_tune: maximum allowed validation recall drop from the day-relaxed baseline.",
    )
    parser.add_argument(
        "--conservative-min-f1-delta",
        type=float,
        default=0.0,
        help="For conservative_tune: minimum allowed validation F1 delta vs day-relaxed baseline.",
    )
    parser.add_argument(
        "--conservative-low-val-pos-shift",
        type=float,
        default=0.08,
        help="For conservative_tune: maximum downward threshold shift explored when validation positives are scarce.",
    )
    parser.add_argument(
        "--conservative-low-val-pos-max-fpr-rise",
        type=float,
        default=0.10,
        help="For conservative_tune: max allowed FPR increase over day-relaxed baseline when val positives are scarce.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "3_1_output_prediction_report_en_scoreprob_auxcls",
        help="Directory to save confusion matrices and reports.",
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


def is_pain_target(target_col: str):
    _, metric = split_day_metric(target_col)
    return metric in PAIN_METRICS


def make_binary_target_from_score(y_score: np.ndarray, threshold: float):
    return (y_score >= float(threshold)).astype(int)


def get_class_labels(threshold: float):
    thr_text = int(threshold) if abs(float(threshold) - round(float(threshold))) < 1e-12 else float(threshold)
    low_label = f"Class 0 (0-{thr_text - 1 if isinstance(thr_text, int) else f'<{thr_text}'})"
    if isinstance(thr_text, int):
        high_label = f"Class 1 ({thr_text}-10)"
    else:
        high_label = f"Class 1 (>={thr_text})"
    return low_label, high_label


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
            if len(y_tr) > 0 and len(np.unique(y_tr)) < 2:
                ok = False
                break
            if len(y_te) > 0 and len(np.unique(y_te)) < 2:
                ok = False
                break
        if ok:
            return train_idx.astype(int), test_idx.astype(int)

    raise ValueError("Could not create a valid train/test split for multi-head labels.")


def split_train_val_multitask(y_train_mat: np.ndarray, val_size: float, random_state: int, max_tries: int = 128):
    n = y_train_mat.shape[0]
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be in [0,1), got {val_size}")
    if val_size == 0.0 or n <= 1:
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


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std < 1e-8] = 1.0
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std
    return x_train_std, x_test_std


def sigmoid(z: np.ndarray):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def relu(x: np.ndarray):
    return np.maximum(0.0, x)


def init_three_layer_params(n_features: int, hidden_size_1: int, hidden_size_2: int, rng: np.random.Generator):
    h1 = max(4, int(hidden_size_1))
    h2 = max(4, int(hidden_size_2))
    w1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, h1))
    b1 = np.zeros(h1, dtype=float)
    w2 = rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2))
    b2 = np.zeros(h2, dtype=float)
    return w1, b1, w2, b2


def forward_multi_head_scores(x: np.ndarray, params):
    w1, b1, w2, b2, w3, b3 = params
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    logits = a2 @ w3 + b3
    unit_prob = sigmoid(logits)
    score_pred = PAIN_SCORE_MAX * unit_prob
    return score_pred, unit_prob, (z1, a1, z2, a2)


def score_to_probability(score_pred: np.ndarray, threshold: float, temperature: float):
    temp = max(1e-6, float(temperature))
    return sigmoid((score_pred - float(threshold)) / temp)


def compute_score_element_grad(y_score: np.ndarray, pred_score: np.ndarray, mask: np.ndarray, args):
    err = pred_score - y_score
    loss_elem = np.zeros_like(pred_score, dtype=float)
    grad_score = np.zeros_like(pred_score, dtype=float)

    if args.loss_type == "mse":
        loss_elem = 0.5 * np.square(err)
        grad_score = err
    else:
        delta = max(1e-6, float(args.huber_delta))
        abs_err = np.abs(err)
        quad = abs_err <= delta
        loss_elem[quad] = 0.5 * np.square(err[quad])
        loss_elem[~quad] = delta * (abs_err[~quad] - 0.5 * delta)
        grad_score[quad] = err[quad]
        grad_score[~quad] = delta * np.sign(err[~quad])

    grad_score[~mask] = 0.0
    loss_elem[~mask] = 0.0
    return loss_elem, grad_score


def weighted_score_loss(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray, loss_type: str, huber_delta: float):
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    y = y_true[mask]
    p = y_pred[mask]
    w = np.clip(sample_weight[mask].astype(float), 0.0, None)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w)
    err = p - y
    if loss_type == "mse":
        loss = 0.5 * np.square(err)
    else:
        delta = max(1e-6, float(huber_delta))
        abs_err = np.abs(err)
        loss = np.where(abs_err <= delta, 0.5 * np.square(err), delta * (abs_err - 0.5 * delta))
    return float(np.sum(loss * w) / np.sum(w))


def compute_aux_cls_element_grad(y_true_bin: np.ndarray, prob_high: np.ndarray, mask: np.ndarray, temperature: float):
    eps = 1e-12
    temp = max(1e-6, float(temperature))
    p = np.clip(prob_high, eps, 1.0 - eps)
    loss_elem = -(y_true_bin * np.log(p) + (1.0 - y_true_bin) * np.log(1.0 - p))
    grad_score = (p - y_true_bin) / temp
    grad_score[~mask] = 0.0
    loss_elem[~mask] = 0.0
    return loss_elem, grad_score


def weighted_binary_log_loss_masked(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    sample_weight: np.ndarray,
):
    valid = mask.astype(bool)
    if not np.any(valid):
        return 0.0
    eps = 1e-12
    y = y_true[valid].astype(float)
    p = np.clip(y_prob[valid].astype(float), eps, 1.0 - eps)
    w = np.clip(sample_weight[valid].astype(float), 0.0, None)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.sum(loss * w) / np.sum(w))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err))) if len(err) > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(err)))) if len(err) > 0 else float("nan")
    return {"mae": mae, "rmse": rmse}


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


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray):
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = compute_auc(y_true, y_prob)
    log_loss = binary_log_loss(y_true, y_prob)
    brier = float(np.mean((y_prob - y_true) ** 2)) if len(y_true) > 0 else float("nan")

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


def get_day_sensitivity_scale(target_col: str):
    day, _ = split_day_metric(target_col)
    return float(DAY_SENSITIVITY_SCALE.get(day, 1.0))


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

    best = None
    for t in candidates:
        m = classification_metrics(y_val, prob_val, threshold=float(t))
        score = m["f1"] + 0.20 * m["precision"] + 0.10 * m["recall"] - args.fp_penalty * m["fpr"]
        if best is None or score > best[0]:
            best = (score, float(t), m)

    return best[1], best[2], base, "fallback_objective"


def choose_threshold_conservative(args, target_col: str, y_val: np.ndarray, prob_val: np.ndarray):
    base_threshold = get_day_relaxed_threshold(target_col, args.day_threshold_map, args.decision_threshold)
    day_scale = get_day_sensitivity_scale(target_col)
    base = classification_metrics(y_val, prob_val, threshold=base_threshold)
    pos_count = int(np.sum(y_val == 1))
    if pos_count < int(args.conservative_min_val_pos):
        low_shift = max(0.0, float(args.conservative_low_val_pos_shift)) * day_scale
        fpr_cap = min(1.0, base["fpr"] + max(0.0, float(args.conservative_low_val_pos_max_fpr_rise)) * day_scale)
        candidates = np.linspace(max(0.05, float(base_threshold) - low_shift), float(base_threshold), args.threshold_grid_size)
        candidates = np.unique(np.concatenate([candidates, np.array([float(base_threshold)])]))
        feasible = []
        for t in candidates:
            m = classification_metrics(y_val, prob_val, threshold=float(t))
            if m["fpr"] <= fpr_cap:
                feasible.append((float(t), m))

        if len(feasible) > 0:
            feasible.sort(
                key=lambda x: (
                    x[1]["recall"],
                    x[1]["f1"],
                    x[1]["precision"],
                    x[1]["accuracy"],
                    -x[1]["fpr"],
                ),
                reverse=True,
            )
            best_t, best_m = feasible[0]
            if best_m["recall"] > base["recall"] + 1e-12:
                return best_t, best_m, base, "conservative_low_val_pos_recall_boost"
        return float(base_threshold), base, base, "conservative_low_val_pos_keep_day_relaxed"

    candidates = np.linspace(0.05, 0.95, args.threshold_grid_size)
    candidates = np.unique(np.concatenate([candidates, np.array([float(base_threshold)])]))
    max_shift = max(0.0, float(args.conservative_max_shift)) * day_scale
    candidates = np.array([
        t for t in candidates if abs(float(t) - float(base_threshold)) <= max_shift + 1e-12
    ])
    if len(candidates) == 0:
        candidates = np.array([float(base_threshold)])

    feasible = []
    recall_floor = max(0.0, base["recall"] - float(args.conservative_max_recall_drop) * day_scale)
    f1_floor = base["f1"] + float(args.conservative_min_f1_delta)
    acc_floor = base["accuracy"] + float(args.conservative_min_acc_gain) / max(day_scale, 1e-6)

    for t in candidates:
        m = classification_metrics(y_val, prob_val, threshold=float(t))
        if m["accuracy"] >= acc_floor and m["recall"] >= recall_floor and m["f1"] >= f1_floor:
            feasible.append((float(t), m))

    if len(feasible) == 0:
        return float(base_threshold), base, base, "conservative_keep_day_relaxed"

    feasible.sort(
        key=lambda x: (
            x[1]["accuracy"],
            x[1]["f1"],
            x[1]["precision"],
            -abs(float(x[0]) - float(base_threshold)),
            -x[1]["fpr"],
        ),
        reverse=True,
    )
    best_t, best_m = feasible[0]
    return best_t, best_m, base, "conservative_small_shift"


def row_normalize_cm(cm: np.ndarray):
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0.0] = 1.0
    return cm.astype(float) / row_sum


def plot_training_accuracy_grid(results, out_file: Path):
    if not HAS_MATPLOTLIB or len(results) == 0:
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

    fig.suptitle("Training Accuracy Curves (Derived High-Pain Labels)", fontsize=14)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    return True


def plot_training_loss(history, out_file: Path):
    if not HAS_MATPLOTLIB or len(history) == 0:
        return False

    epochs = [h["epoch"] for h in history]
    total_loss = [h["train_loss"] for h in history]
    reg_loss = [h.get("train_reg_loss", float("nan")) for h in history]
    aux_loss = [h.get("train_aux_cls_loss", float("nan")) for h in history]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, total_loss, label="Total Loss", color="#1f77b4", linewidth=2.0)
    ax.plot(epochs, reg_loss, label="Regression Loss", color="#2ca02c", linewidth=1.8)
    ax.plot(epochs, aux_loss, label="Aux Classification Loss", color="#d62728", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close(fig)
    return True


def plot_confusion_matrices_grid(results, out_file: Path):
    if not HAS_MATPLOTLIB or len(results) == 0:
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
        r = result["regression_metrics"]
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
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
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


def fit_shared_backbone_score_multitask(
    x_train: np.ndarray,
    y_train_score_mat: np.ndarray,
    target_cols,
    args,
):
    n_samples, n_features = x_train.shape
    n_heads = y_train_score_mat.shape[1]
    rng = np.random.default_rng(args.random_state)

    w1, b1, w2, b2 = init_three_layer_params(n_features, args.hidden_size_1, args.hidden_size_2, rng)
    w3 = rng.normal(0.0, np.sqrt(2.0 / w2.shape[1]), size=(w2.shape[1], n_heads))
    b3 = np.zeros(n_heads, dtype=float)
    params = (w1, b1, w2, b2, w3, b3)

    label_mask = ~np.isnan(y_train_score_mat)
    y_filled = np.where(label_mask, y_train_score_mat, 0.0).astype(float)
    y_high = (y_filled >= float(args.pain_threshold)) & label_mask
    low_mask = (y_filled < float(args.pain_threshold)) & label_mask

    pos_cnt = np.sum(y_high, axis=0).astype(float)
    neg_cnt = np.sum(low_mask, axis=0).astype(float)
    if args.positive_weight_mode == "balanced":
        pos_weight_vec = np.minimum(neg_cnt / (pos_cnt + 1e-8), 10.0)
    elif args.positive_weight_mode == "sqrt_balanced":
        pos_weight_vec = np.minimum(np.sqrt(neg_cnt / (pos_cnt + 1e-8)), 10.0)
    else:
        pos_weight_vec = np.ones(n_heads, dtype=float)
    pos_weight_vec = np.where(pos_cnt > 0.0, pos_weight_vec, 1.0).astype(float)
    day_scale_vec = np.array([get_day_sensitivity_scale(t) for t in target_cols], dtype=float)

    rare_target = max(1.0, float(args.rare_positive_target_count))
    rare_power = max(0.0, float(args.rare_positive_boost_power))
    rare_max = max(1.0, float(args.rare_positive_max_boost))
    shortage_ratio = np.where(pos_cnt > 0.0, rare_target / np.maximum(pos_cnt, 1e-8), 1.0)
    rare_head_boost = np.where(
        pos_cnt < rare_target,
        np.minimum(np.power(np.maximum(shortage_ratio, 1.0), rare_power), rare_max),
        1.0,
    ).astype(float)
    pos_weight_vec *= rare_head_boost * day_scale_vec

    oversample_weight = np.ones(n_samples, dtype=float)
    if not args.disable_high_pain_oversampling:
        sample_high_mask = np.any((y_filled > args.high_pain_score_threshold) & label_mask, axis=1)
        oversample_weight[sample_high_mask] = np.maximum(oversample_weight[sample_high_mask], args.high_pain_oversample_factor)
        rare_head_mask = (pos_cnt > 0.0) & (pos_cnt < rare_target)
        if np.any(rare_head_mask):
            rare_sample_mask = np.any(y_high[:, rare_head_mask], axis=1)
            rare_head_scale = float(np.max(day_scale_vec[rare_head_mask])) if np.any(rare_head_mask) else 1.0
            rare_over = (
                float(args.high_pain_oversample_factor)
                * max(1.0, float(args.rare_positive_oversample_boost))
                * rare_head_scale
            )
            oversample_weight[rare_sample_mask] = np.maximum(oversample_weight[rare_sample_mask], rare_over)
    oversample_weight = np.clip(oversample_weight, 1.0, None)
    over_prob = oversample_weight / np.sum(oversample_weight)

    best_loss = np.inf
    best_params = tuple(p.copy() for p in params)
    stale = 0
    history = []
    batch_size = max(1, min(int(args.batch_size), n_samples))

    for epoch in range(1, int(args.epochs) + 1):
        if not args.disable_high_pain_oversampling:
            order = rng.choice(n_samples, size=n_samples, replace=True, p=over_prob)
        else:
            order = rng.permutation(n_samples)

        w1, b1, w2, b2, w3, b3 = params
        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_filled[idx]
            mb = label_mask[idx]
            yb_high = y_high[idx].astype(float)

            score_pred, unit_prob, (z1, a1, z2, a2) = forward_multi_head_scores(xb, params)
            prob_high = score_to_probability(score_pred, args.pain_threshold, args.prob_temperature)
            reg_loss_elem, reg_grad_score = compute_score_element_grad(yb, score_pred, mb, args)
            aux_loss_elem, aux_grad_score = compute_aux_cls_element_grad(
                yb_high,
                prob_high,
                mb,
                args.prob_temperature,
            )

            class_w = np.where(yb >= float(args.pain_threshold), pos_weight_vec.reshape(1, -1), 1.0) * mb.astype(float)
            high_pain_w = np.where(yb > float(args.high_pain_score_threshold), float(args.high_pain_loss_weight), 1.0) * mb.astype(float)
            total_w = class_w * high_pain_w
            reg_loss_elem *= total_w
            reg_grad_score *= total_w
            aux_scale_vec = max(0.0, float(args.aux_cls_weight)) * day_scale_vec.reshape(1, -1)
            aux_loss_elem *= total_w * aux_scale_vec
            aux_grad_score *= total_w * aux_scale_vec

            weight_sum = float(np.sum(total_w))
            if weight_sum <= 0.0:
                continue
            grad_score = (reg_grad_score + aux_grad_score) / weight_sum
            dscore_dlogits = PAIN_SCORE_MAX * unit_prob * (1.0 - unit_prob)
            dlogits = grad_score * dscore_dlogits

            grad_w3 = a2.T @ dlogits + float(args.l2) * w3
            grad_b3 = dlogits.sum(axis=0)
            da2 = dlogits @ w3.T
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

        score_all, _, _ = forward_multi_head_scores(x_train, params)
        prob_high_all = score_to_probability(score_all, args.pain_threshold, args.prob_temperature)
        eval_weight = np.where(y_filled >= float(args.pain_threshold), pos_weight_vec.reshape(1, -1), 1.0)
        eval_weight *= np.where(y_filled > float(args.high_pain_score_threshold), float(args.high_pain_loss_weight), 1.0)
        eval_weight *= label_mask.astype(float)
        train_reg_loss = weighted_score_loss(y_filled, score_all, eval_weight, args.loss_type, args.huber_delta)
        train_aux_cls_loss = weighted_binary_log_loss_masked(
            y_high.astype(float),
            prob_high_all,
            label_mask,
            eval_weight,
        )
        reg_term = 0.5 * float(args.l2) * float(np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3))
        train_loss = train_reg_loss + max(0.0, float(args.aux_cls_weight)) * train_aux_cls_loss + reg_term

        pred_high_all = (prob_high_all >= args.decision_threshold).astype(float)
        correct = (pred_high_all == y_high.astype(float)) & label_mask
        train_acc = float(np.sum(correct) / max(1.0, float(np.sum(label_mask))))
        head_correct = np.sum(correct, axis=0).astype(float)
        head_total = np.sum(label_mask, axis=0).astype(float)
        head_acc = np.divide(head_correct, np.maximum(1.0, head_total))
        abs_err = np.abs(score_all - y_filled) * label_mask.astype(float)
        head_mae = np.divide(np.sum(abs_err, axis=0), np.maximum(1.0, head_total))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_reg_loss": float(train_reg_loss),
                "train_aux_cls_loss": float(train_aux_cls_loss),
                "train_acc": float(train_acc),
                "head_train_acc": head_acc.tolist(),
                "head_train_mae": head_mae.tolist(),
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

    return best_params, best_loss, history, (not args.disable_high_pain_oversampling)


def run_shared_backbone_targets(args, df: pd.DataFrame, targets):
    if len(targets) == 0:
        raise ValueError("No targets provided for training.")

    shared_feature_cols = None
    for t in targets:
        cols = set(select_feature_columns(df, target_col=t, feature_mode=args.feature_mode))
        shared_feature_cols = cols if shared_feature_cols is None else shared_feature_cols.intersection(cols)
    shared_feature_cols = [c for c in df.columns if c in (shared_feature_cols or set())]
    if len(shared_feature_cols) == 0:
        raise ValueError("No shared feature columns found across selected targets.")

    y_score_cols = []
    for t in targets:
        y_raw = pd.to_numeric(df[t], errors="coerce")
        y_score_cols.append(y_raw.to_numpy(dtype=float))

    y_score_mat = np.stack(y_score_cols, axis=1)
    y_bin_mat = make_binary_target_from_score(np.nan_to_num(y_score_mat, nan=-1e9), args.pain_threshold).astype(float)
    y_bin_mat[np.isnan(y_score_mat)] = np.nan

    any_label_mask = np.any(~np.isnan(y_score_mat), axis=1)
    if not np.any(any_label_mask):
        raise ValueError("All selected targets are missing for all rows.")

    x_df = prepare_features(df.loc[any_label_mask].copy(), shared_feature_cols, args.feature_impute)
    y_score_mat = y_score_mat[any_label_mask]
    y_bin_mat = y_bin_mat[any_label_mask]

    train_idx, test_idx = split_train_test_multitask(y_bin_mat, args.test_size, args.random_state)
    x_all = x_df.to_numpy(dtype=float)
    x_train_all, x_test = x_all[train_idx], x_all[test_idx]
    y_score_train_all, y_score_test = y_score_mat[train_idx], y_score_mat[test_idx]
    y_bin_train_all, y_bin_test = y_bin_mat[train_idx], y_bin_mat[test_idx]

    tr_sub_idx, val_idx = split_train_val_multitask(y_bin_train_all, args.val_size, args.random_state + 17)
    x_train, y_score_train = x_train_all[tr_sub_idx], y_score_train_all[tr_sub_idx]
    x_val, y_score_val = x_train_all[val_idx], y_score_train_all[val_idx]
    y_bin_val = y_bin_train_all[val_idx]

    x_train_std, x_test_std = standardize_train_test(x_train, x_test)
    _, x_val_std = standardize_train_test(x_train, x_val)

    best_params, train_loss, train_history, oversampling_used = fit_shared_backbone_score_multitask(
        x_train=x_train_std,
        y_train_score_mat=y_score_train,
        target_cols=targets,
        args=args,
    )

    test_score_pred_mat, _, _ = forward_multi_head_scores(x_test_std, best_params)
    val_score_pred_mat, _, _ = (
        forward_multi_head_scores(x_val_std, best_params) if len(val_idx) > 0 else (np.zeros((0, len(targets))), None, None)
    )
    test_prob_mat = score_to_probability(test_score_pred_mat, args.pain_threshold, args.prob_temperature)
    val_prob_mat = score_to_probability(val_score_pred_mat, args.pain_threshold, args.prob_temperature)

    low_label, high_label = get_class_labels(args.pain_threshold)
    results = []
    for h, target_col in enumerate(targets):
        score_test_h = y_score_test[:, h]
        test_mask_h = ~np.isnan(score_test_h)
        if not np.any(test_mask_h):
            continue

        y_test_score_vec = score_test_h[test_mask_h].astype(float)
        y_test_vec = make_binary_target_from_score(y_test_score_vec, args.pain_threshold)
        score_pred_test_vec = test_score_pred_mat[test_mask_h, h]
        prob_test_vec = test_prob_mat[test_mask_h, h]

        if len(val_idx) > 0:
            score_val_h = y_score_val[:, h]
            val_mask_h = ~np.isnan(score_val_h)
            y_val_score_vec = score_val_h[val_mask_h].astype(float)
            y_val_vec = make_binary_target_from_score(y_val_score_vec, args.pain_threshold)
            prob_val_vec = val_prob_mat[val_mask_h, h]
        else:
            val_mask_h = np.array([], dtype=bool)
            y_val_score_vec = np.array([])
            y_val_vec = np.array([])
            prob_val_vec = np.array([])

        if args.threshold_strategy == "day_relaxed":
            chosen_threshold = get_day_relaxed_threshold(target_col, args.day_threshold_map, args.decision_threshold)
            val_base = classification_metrics(y_val_vec, prob_val_vec, threshold=args.decision_threshold) if len(y_val_vec) > 0 else None
            val_selected = classification_metrics(y_val_vec, prob_val_vec, threshold=chosen_threshold) if len(y_val_vec) > 0 else None
            threshold_selection_mode = "day_relaxed"
        elif args.threshold_strategy == "conservative_tune" and len(y_val_vec) > 0 and len(np.unique(y_val_vec)) >= 2:
            chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_conservative(
                args, target_col, y_val_vec, prob_val_vec
            )
        elif args.threshold_strategy == "fixed" or len(y_val_vec) == 0 or len(np.unique(y_val_vec)) < 2:
            chosen_threshold = args.decision_threshold
            val_base = classification_metrics(y_val_vec, prob_val_vec, threshold=args.decision_threshold) if len(y_val_vec) > 0 else None
            val_selected = classification_metrics(y_val_vec, prob_val_vec, threshold=chosen_threshold) if len(y_val_vec) > 0 else None
            threshold_selection_mode = "fixed_threshold"
        else:
            chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_by_validation(
                args, y_val_vec, prob_val_vec
            )

        metrics = classification_metrics(y_test_vec, prob_test_vec, threshold=chosen_threshold)
        reg_metrics = regression_metrics(y_test_score_vec, score_pred_test_vec)
        cm_count = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
        cm_prob = row_normalize_cm(cm_count)

        train_score_h = y_score_train[:, h]
        train_mask_h = ~np.isnan(train_score_h)
        high_mask_h = train_mask_h & (train_score_h > args.high_pain_score_threshold)

        task_train_history = []
        for hh in train_history:
            head_acc_list = hh.get("head_train_acc", [])
            head_mae_list = hh.get("head_train_mae", [])
            task_train_history.append(
                {
                    "epoch": int(hh["epoch"]),
                    "train_loss": float(hh["train_loss"]),
                    "train_reg_loss": float(hh.get("train_reg_loss", hh["train_loss"])),
                    "train_aux_cls_loss": float(hh.get("train_aux_cls_loss", 0.0)),
                    "train_acc": float(head_acc_list[h]) if h < len(head_acc_list) else float(hh.get("train_acc", 0.0)),
                    "train_mae": float(head_mae_list[h]) if h < len(head_mae_list) else float("nan"),
                }
            )

        results.append(
            {
                "target_col": target_col,
                "target_en": target_to_english_name(target_col),
                "n_total_non_missing": int(np.sum(~np.isnan(y_score_mat[:, h]))),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_val": int(np.sum(val_mask_h)) if len(val_idx) > 0 else 0,
                "n_pos_train": int(np.sum(make_binary_target_from_score(train_score_h[train_mask_h], args.pain_threshold))),
                "n_pos_val": int(np.sum(y_val_vec)) if len(y_val_vec) > 0 else 0,
                "n_pos_test": int(np.sum(y_test_vec)),
                "n_high_pain_train": int(np.sum(high_mask_h)),
                "feature_count": int(len(shared_feature_cols)),
                "train_loss": float(train_loss),
                "train_reg_loss": float(task_train_history[-1]["train_reg_loss"]) if len(task_train_history) > 0 else float("nan"),
                "train_aux_cls_loss": float(task_train_history[-1]["train_aux_cls_loss"]) if len(task_train_history) > 0 else float("nan"),
                "chosen_threshold": float(chosen_threshold),
                "threshold_selection_mode": threshold_selection_mode,
                "val_metrics_base": val_base,
                "val_metrics_selected": val_selected,
                "metrics": metrics,
                "regression_metrics": reg_metrics,
                "mean_true_score_test": float(np.mean(y_test_score_vec)),
                "mean_pred_score_test": float(np.mean(score_pred_test_vec)),
                "low_label": low_label,
                "high_label": high_label,
                "cm_count": cm_count,
                "cm_prob": cm_prob,
                "train_history": task_train_history,
                "shared_train_history": train_history,
                "oversampling_used": bool(oversampling_used),
            }
        )

    if len(results) == 0:
        raise ValueError("No evaluable heads found on test split.")

    return results


def print_result(result: dict):
    m = result["metrics"]
    r = result["regression_metrics"]
    print("=" * 82)
    print(f"Target: {result['target_col']} ({result['target_en']})")
    print("=" * 82)
    print(f"Samples (non-missing)    : {result['n_total_non_missing']}")
    print(f"Train / Val / Test       : {result['n_train']} / {result['n_val']} / {result['n_test']}")
    print(f"Pos in Train/Val/Test    : {result['n_pos_train']} / {result['n_pos_val']} / {result['n_pos_test']}")
    print(f"Class split              : {result['low_label']} vs {result['high_label']}")
    print(f"High pain in train (>3)  : {result['n_high_pain_train']}")
    print(f"Feature count            : {result['feature_count']}")
    print(f"Train loss               : {result['train_loss']:.6f}")
    print(f"Train reg / aux cls loss : {result['train_reg_loss']:.6f} / {result['train_aux_cls_loss']:.6f}")
    print(f"Threshold                : {result['chosen_threshold']:.3f} ({result['threshold_selection_mode']})")
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
    print(f"Regression MAE           : {r['mae']:.4f}")
    print(f"Regression RMSE          : {r['rmse']:.4f}")
    print(f"Mean true/pred score     : {result['mean_true_score_test']:.3f} / {result['mean_pred_score_test']:.3f}")
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
        if args.threshold_strategy in ("day_relaxed", "conservative_tune")
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
    else:
        selected_metrics = list(PAIN_METRICS)

    target_candidates = [f"{day}_{metric}" for day in day_candidates for metric in selected_metrics]
    targets = [c for c in target_candidates if c in df.columns]
    missing_targets = [c for c in target_candidates if c not in df.columns]
    if len(targets) == 0:
        raise ValueError("No valid pain target columns found for current settings.")

    low_label, high_label = get_class_labels(args.pain_threshold)
    print(f"Input file               : {args.input}")
    print("Training target          : raw pain score regression + auxiliary high-pain classification, pain-only")
    print(f"Derived class rule       : {low_label} vs {high_label}")
    print(f"Probability temperature  : {args.prob_temperature}")
    print(f"Aux cls loss weight      : {args.aux_cls_weight}")
    print("Missing outcome handling : drop missing rows head-wise before evaluation")
    print(f"Positive weight mode     : {args.positive_weight_mode}")
    print(f"Regression loss          : {args.loss_type}")
    if args.loss_type == "huber":
        print(f"Huber delta              : {args.huber_delta}")
    print(f"Model layers             : 3 (input -> {args.hidden_size_1} -> {args.hidden_size_2} -> score)")
    print(f"High pain oversampling   : {not args.disable_high_pain_oversampling}")
    print(f"High pain score threshold: > {args.high_pain_score_threshold}")
    print(f"High pain oversample fac.: {args.high_pain_oversample_factor}")
    print(f"High pain loss weight    : {args.high_pain_loss_weight}")
    print(f"Threshold strategy       : {args.threshold_strategy}")
    if args.threshold_strategy in ("day_relaxed", "conservative_tune"):
        print(f"Day thresholds           : {args.day_threshold_map}")
    if args.threshold_strategy == "conservative_tune":
        print(
            "Conservative guards      : "
            f"min_val_pos={args.conservative_min_val_pos}, "
            f"max_shift={args.conservative_max_shift}, "
            f"min_acc_gain={args.conservative_min_acc_gain}, "
            f"max_recall_drop={args.conservative_max_recall_drop}, "
            f"min_f1_delta={args.conservative_min_f1_delta}, "
            f"low_val_pos_shift={args.conservative_low_val_pos_shift}, "
            f"low_val_pos_max_fpr_rise={args.conservative_low_val_pos_max_fpr_rise}"
        )
    print(
        "Rare positive boost      : "
        f"target_count={args.rare_positive_target_count}, "
        f"power={args.rare_positive_boost_power}, "
        f"max_boost={args.rare_positive_max_boost}, "
        f"oversample_boost={args.rare_positive_oversample_boost}"
    )
    print(f"Day sensitivity scale    : {DAY_SENSITIVITY_SCALE}")
    print(f"Feature mode             : {args.feature_mode}")
    if args.feature_mode == "all":
        print("Note                     : Using other outcomes as features may inflate metrics.")
    if args.feature_mode in ("temporal", "strong_signal_temporal"):
        print("Note                     : Temporal strong-signal mode uses all earlier-day outcomes to avoid leakage.")
    print(f"Targets                  : {targets}")
    if missing_targets:
        print(f"Missing targets skipped  : {missing_targets}")
    print(f"Output dir               : {args.output_dir}")

    all_rows = []
    all_results = run_shared_backbone_targets(args, df, targets)
    if len(all_results) > 0:
        print(f"Oversampling applied     : {all_results[0].get('oversampling_used', False)}")
    for result in all_results:
        print_result(result)
        m = result["metrics"]
        r = result["regression_metrics"]
        all_rows.append(
            {
                "target": result["target_en"],
                "class_0_label": result["low_label"],
                "class_1_label": result["high_label"],
                "mae": r["mae"],
                "rmse": r["rmse"],
                "mean_true_score_test": result["mean_true_score_test"],
                "mean_pred_score_test": result["mean_pred_score_test"],
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
                "train_reg_loss": result["train_reg_loss"],
                "train_aux_cls_loss": result["train_aux_cls_loss"],
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

    combined_loss_fig = args.output_dir / "training_loss_all_targets.png"
    loss_plotted = plot_training_loss(
        all_results[0].get("shared_train_history", []),
        combined_loss_fig,
    ) if len(all_results) > 0 else False
    if loss_plotted:
        print(f"Saved combined loss figure: {combined_loss_fig}")
    else:
        print("Saved combined loss figure: skipped (matplotlib unavailable or no history)")

    cleanup_output_dir_keep_totals(args.output_dir)
    print(f"Cleaned output dir       : kept only total CSV/PNG files in {args.output_dir}")


if __name__ == "__main__":
    main()
