import os
import argparse
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
PAIN_TYPES = {"rest": "静息痛", "movement": "活动痛"}


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train logistic regression model(s) for pain classification (0-3 vs 4-10)."
    )
    parser.add_argument("--input", type=Path, default=base_dir / "data_vectorized.csv", help="Input CSV path.")
    parser.add_argument(
        "--day",
        type=str,
        default="术后第一天",
        help='Outcome day prefix, e.g. "术后第一天"; use "all" for all outcome days.',
    )
    parser.add_argument(
        "--pain-type",
        type=str,
        choices=["rest", "movement", "both"],
        default="both",
        help="Train on rest pain, movement pain, or both.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["strict", "all", "temporal"],
        default="strict",
        help=(
            "strict: drop all outcomes from features; "
            "all: use all columns except current target (may leak); "
            "temporal: use non-outcomes + only earlier-day outcomes."
        ),
    )
    parser.add_argument(
        "--feature-impute",
        type=str,
        choices=["median", "zero"],
        default="median",
        help="How to impute missing feature values.",
    )
    parser.add_argument("--pain-threshold", type=float, default=4.0, help=">= threshold means class 1.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--patience", type=int, default=120, help="Early stop patience on train loss.")
    parser.add_argument(
        "--positive-weight-mode",
        type=str,
        choices=["balanced", "none", "sqrt_balanced"],
        default="balanced",
        help="Class weighting for positive class in training batches.",
    )
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        choices=["fixed", "tune"],
        default="tune",
        help="fixed: use --decision-threshold; tune: choose threshold on validation set.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for fixed mode.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split from train set for threshold tuning.")
    parser.add_argument("--fpr-reduction-target", type=float, default=0.35, help="Target FPR reduction ratio vs threshold=0.5 on validation.")
    parser.add_argument("--acc-drop-tolerance", type=float, default=0.02, help="Max allowed validation accuracy drop when tuning threshold.")
    parser.add_argument("--min-recall-ratio", type=float, default=0.75, help="Keep recall above baseline_recall * ratio when tuning.")
    parser.add_argument("--threshold-grid-size", type=int, default=181, help="Threshold candidates in [0.05, 0.95].")
    parser.add_argument("--fp-penalty", type=float, default=0.25, help="Fallback threshold objective penalty on FPR.")
    parser.add_argument("--verbose", action="store_true", help="Print training logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "baseline" / "logistic_outputs",
        help="Directory to save confusion matrices and optional predictions.",
    )
    parser.add_argument("--save-predictions", action="store_true", help="Save per-sample test predictions CSV.")
    return parser.parse_args()


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
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


def select_feature_columns(df: pd.DataFrame, target_col: str, feature_mode: str):
    outcome_cols = get_outcome_columns(df)

    if feature_mode == "strict":
        if target_col in outcome_cols:
            return [c for c in df.columns if c not in outcome_cols]
        return [c for c in df.columns if c != target_col]

    if feature_mode == "all":
        return [c for c in df.columns if c != target_col]

    # temporal mode:
    # use all non-outcome features + only outcome features from earlier days.
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


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray):
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_logistic_regression_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    patience: int,
    random_state: int,
    positive_weight_mode: str,
    x_eval: np.ndarray = None,
    y_eval: np.ndarray = None,
    verbose: bool = False,
):
    n_samples, n_features = x_train.shape
    rng = np.random.default_rng(random_state)
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    best_loss = np.inf
    best_w = w.copy()
    best_b = b
    stale = 0
    history = []

    batch_size = max(1, min(batch_size, n_samples))
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            prob = sigmoid(xb @ w + b)

            pos = np.sum(yb == 1)
            neg = np.sum(yb == 0)
            if positive_weight_mode == "balanced":
                pos_weight = min(neg / (pos + 1e-8), 10.0) #改
            elif positive_weight_mode == "sqrt_balanced":
                pos_weight = min(np.sqrt(neg / (pos + 1e-8)), 10.0) #改
            else:
                pos_weight = 1.0

            weight = np.where(yb == 1, pos_weight, 1.0)
            weight = weight / (pos_weight + 1.0) #改
            error = (prob - yb) * weight

            grad_w = (xb.T @ error) / len(idx) + l2 * w
            grad_b = float(np.mean(error))

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

        train_prob = sigmoid(x_train @ w + b)
        train_loss = binary_log_loss(y_train, train_prob) + 0.5 * l2 * float(np.sum(w * w))
        train_acc = float(np.mean((train_prob >= 0.5).astype(int) == y_train))

        if x_eval is not None and y_eval is not None and len(y_eval) > 0:
            eval_prob = sigmoid(x_eval @ w + b)
            eval_loss = binary_log_loss(y_eval, eval_prob)
            eval_acc = float(np.mean((eval_prob >= 0.5).astype(int) == y_eval))
        else:
            eval_loss = float("nan")
            eval_acc = float("nan")

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": train_acc,
                "test_loss": float(eval_loss),
                "test_acc": float(eval_acc),
            }
        )

        if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == epochs):
            if np.isnan(eval_acc):
                print(f"[epoch {epoch:4d}] train_loss={train_loss:.6f} train_acc={train_acc:.4f}")
            else:
                print(
                    f"[epoch {epoch:4d}] train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
                    f"test_acc={eval_acc:.4f}"
                )

        if best_loss - train_loss > 1e-8:
            best_loss = train_loss
            best_w = w.copy()
            best_b = b
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return best_w, best_b, best_loss, history


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


def plot_confusion_matrix(cm: np.ndarray, title: str, out_file: Path):
    if not HAS_MATPLOTLIB:
        return False

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=12)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close(fig)
    return True


def plot_train_test_accuracy(history, title: str, out_file: Path):
    if not HAS_MATPLOTLIB:
        return False
    if len(history) == 0:
        return False

    epochs = [h["epoch"] for h in history]
    train_accs = [h["train_acc"] for h in history]
    test_accs = [h["test_acc"] for h in history]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(epochs, train_accs, color="#1f77b4", linewidth=2.0, label="Train Acc")
    if not np.all(np.isnan(np.asarray(test_accs, dtype=float))):
        ax.plot(epochs, test_accs, color="#ff7f0e", linewidth=2.0, label="Test Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close(fig)
    return True


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
    y_bin = (y_score >= args.pain_threshold).astype(int)

    if len(np.unique(y_bin)) < 2:
        raise ValueError(f"Target {target_col} has only one class after missing drop.")

    x = x_df.to_numpy(dtype=float)
    train_idx, test_idx = split_train_test_stratified(y_bin, args.test_size, args.random_state)
    x_train_all, x_test = x[train_idx], x[test_idx]
    y_train_all, y_test = y_bin[train_idx], y_bin[test_idx]
    score_test = y_score[test_idx]

    # Validation split for threshold tuning to reduce false positives without overfitting test.
    tr_sub_idx, val_idx = split_train_val_stratified(y_train_all, args.val_size, args.random_state + 17)
    x_train, y_train = x_train_all[tr_sub_idx], y_train_all[tr_sub_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]

    x_train_std, x_test_std = standardize_train_test(x_train, x_test)
    _, x_val_std = standardize_train_test(x_train, x_val)
    w, b, train_loss, train_history = fit_logistic_regression_batch(
        x_train=x_train_std,
        y_train=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        random_state=args.random_state,
        positive_weight_mode=args.positive_weight_mode,
        x_eval=x_test_std,
        y_eval=y_test,
        verbose=args.verbose,
    )

    test_prob = sigmoid(x_test_std @ w + b)
    val_prob = sigmoid(x_val_std @ w + b) if len(y_val) > 0 else np.array([])

    if args.threshold_strategy == "fixed" or len(y_val) == 0:
        chosen_threshold = args.decision_threshold
        val_base = classification_metrics(y_val, val_prob, threshold=args.decision_threshold) if len(y_val) > 0 else None
        val_selected = classification_metrics(y_val, val_prob, threshold=chosen_threshold) if len(y_val) > 0 else None
        threshold_selection_mode = "fixed_threshold"
    else:
        chosen_threshold, val_selected, val_base, threshold_selection_mode = choose_threshold_by_validation(
            args, y_val, val_prob
        )

    metrics = classification_metrics(y_test, test_prob, threshold=chosen_threshold)

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    cm_file = output_dir / f"confusion_matrix_{target_col}.png"
    plotted = plot_confusion_matrix(cm, title="Confusion Matrix", out_file=cm_file)
    acc_curve_file = output_dir / f"train_test_acc_{target_col}.png"
    acc_curve_plotted = plot_train_test_accuracy(
        train_history, title=f"Train/Test Accuracy - {target_col}", out_file=acc_curve_file
    )

    if args.save_predictions:
        pred_df = pd.DataFrame(
            {
                "target_score": score_test,
                "y_true": y_test,
                "y_prob": test_prob,
                "y_pred": metrics["y_pred"],
                "decision_threshold": chosen_threshold,
            }
        )
        pred_df.to_csv(output_dir / f"predictions_{target_col}.csv", index=False, encoding="utf-8-sig")

    return {
        "target_col": target_col,
        "n_total_non_missing": int(len(y_bin)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_pos_train": int(np.sum(y_train)),
        "n_val": int(len(y_val)),
        "n_pos_val": int(np.sum(y_val)) if len(y_val) > 0 else 0,
        "n_pos_test": int(np.sum(y_test)),
        "feature_count": int(len(feature_cols)),
        "train_loss": float(train_loss),
        "chosen_threshold": float(chosen_threshold),
        "threshold_selection_mode": threshold_selection_mode,
        "val_metrics_base": val_base,
        "val_metrics_selected": val_selected,
        "metrics": metrics,
        "cm": cm,
        "cm_file": cm_file,
        "cm_plotted": plotted,
        "acc_curve_file": acc_curve_file,
        "acc_curve_plotted": acc_curve_plotted,
    }


def print_result(result: dict):
    m = result["metrics"]
    print("=" * 82)
    print(f"Target: {result['target_col']}")
    print("=" * 82)
    print(f"Samples (non-missing)    : {result['n_total_non_missing']}")
    print(f"Train / Val / Test       : {result['n_train']} / {result['n_val']} / {result['n_test']}")
    print(f"Pos in Train/Val/Test    : {result['n_pos_train']} / {result['n_pos_val']} / {result['n_pos_test']}")
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
    print("Confusion Matrix (rows=true [0,1], cols=pred [0,1])")
    print(f"[[{m['tn']}, {m['fp']}],")
    print(f" [{m['fn']}, {m['tp']}]]")
    if result["cm_plotted"]:
        print(f"Confusion matrix figure  : {result['cm_file']}")
    else:
        print("Confusion matrix figure  : skipped (matplotlib unavailable)")
    if result["acc_curve_plotted"]:
        print(f"Train/Test acc curve     : {result['acc_curve_file']}")
    else:
        print("Train/Test acc curve     : skipped (matplotlib unavailable)")
    print("=" * 82)


def main():
    args = parse_args()
    df = read_csv_with_fallback(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.day == "all":
        selected_days = OUTCOME_DAYS
    else:
        if args.day not in OUTCOME_DAYS:
            raise ValueError(f"--day must be one of {OUTCOME_DAYS + ['all']}, got: {args.day}")
        selected_days = [args.day]

    targets = []
    for day in selected_days:
        if args.pain_type in ("rest", "both"):
            targets.append(f"{day}_{PAIN_TYPES['rest']}")
        if args.pain_type in ("movement", "both"):
            targets.append(f"{day}_{PAIN_TYPES['movement']}")

    available_targets = [t for t in targets if t in df.columns]
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        print(f"Skipped missing targets  : {missing_targets}")
    targets = available_targets
    if not targets:
        raise ValueError("No valid targets found in input file for current --day/--pain-type settings.")

    print(f"Input file               : {args.input}")
    print(f"Pain threshold rule      : 0-3 -> class 0, >= {args.pain_threshold} -> class 1")
    print("Missing outcome handling : drop missing rows before train/test split and batch training")
    print(f"Positive weight mode     : {args.positive_weight_mode}")
    print(f"Threshold strategy       : {args.threshold_strategy}")
    print(f"Feature mode             : {args.feature_mode}")
    if args.feature_mode == "all":
        print("Note                     : Using other outcomes as features may inflate metrics.")
    if args.feature_mode == "temporal":
        print("Note                     : Temporal mode uses only earlier-day outcomes to avoid leakage.")
    print(f"Selected days            : {selected_days}")
    print(f"Targets                  : {targets}")
    print(f"Output dir               : {args.output_dir}")

    all_rows = []
    for target_col in targets:
        result = run_one_target(args, df, target_col=target_col, output_dir=args.output_dir)
        print_result(result)

        m = result["metrics"]
        all_rows.append(
            {
                "target_col": target_col,
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
            }
        )

    summary_df = pd.DataFrame(all_rows)
    summary_file = args.output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Saved summary metrics    : {summary_file}")


if __name__ == "__main__":
    main()
