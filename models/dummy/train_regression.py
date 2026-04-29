"""Train naive single-target regression baselines.

The default mean predictor is the floor that every optimized regression model
should beat. Use the median variant when the target distribution is skewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = next((parent for parent in Path(__file__).resolve().parents if (parent / "pixi.toml").exists()), None)
if PROJECT_DIR is not None and str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from splits.split_utils import get_target_tier, split_positions_from_reference

from config import (
    ARTIFACT_DIR,
    DATA_PATH,
    ENABLE_TEMPORAL_FILTER,
    MANUAL_FEATURE_COLUMNS,
    OUTPUT_DIR,
    RANDOM_STATE,
    STRICT_PAST_ONLY,
    TARGET_COLUMN,
    TEST_SIZE,
)
from models.random_forest.temporal_feature_filter import apply_temporal_feature_filter

META_PREFIX = "__meta_"
META_IS_GENERATED_COL = "__meta_is_generated"
META_SPLIT_COL = "__meta_dataset_split"
TEST_SPLIT_VALUE = "test"
VALIDATION_SPLIT_VALUE = "validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a naive regression baseline.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["mean", "median"], default="mean")
    return parser.parse_args()


def is_metadata_column(col: str) -> bool:
    return str(col).startswith(META_PREFIX)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("数据为空，无法训练模型。")
    return df


def build_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"目标列不存在: {TARGET_COLUMN}")

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if ENABLE_TEMPORAL_FILTER:
        feature_cols, dropped_cols = apply_temporal_feature_filter(
            all_columns=df.columns.tolist(),
            target_column=TARGET_COLUMN,
            manual_feature_columns=MANUAL_FEATURE_COLUMNS,
            strict_past_only=STRICT_PAST_ONLY,
        )
        if dropped_cols:
            print(f"时间过滤已启用：剔除 {len(dropped_cols)} 个目标时点及之后的时间列。")
    elif MANUAL_FEATURE_COLUMNS:
        missing = [col for col in MANUAL_FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"手动指定的特征列不存在: {missing}")
        feature_cols = [col for col in MANUAL_FEATURE_COLUMNS if col != TARGET_COLUMN]
    else:
        feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

    feature_cols = [col for col in feature_cols if not is_metadata_column(col)]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    if META_IS_GENERATED_COL in df.columns:
        is_generated = pd.to_numeric(df.loc[valid_mask, META_IS_GENERATED_COL], errors="coerce").fillna(0).astype(int)
    else:
        is_generated = pd.Series(0, index=X.index, dtype=int)
    if META_SPLIT_COL in df.columns:
        split_labels = df.loc[valid_mask, META_SPLIT_COL].astype(str)
    else:
        split_labels = pd.Series("", index=X.index, dtype=str)
    if len(X) < 10:
        raise ValueError("可用样本过少（<10），建议检查数据或目标列。")
    return X, y, is_generated, split_labels


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    is_generated: pd.Series,
    split_labels: pd.Series,
    split_file: Path | None,
    split_seed: int,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    val_idx = pd.Index([])
    explicit_test_mask = split_labels.eq(TEST_SPLIT_VALUE)
    explicit_val_mask = split_labels.eq(VALIDATION_SPLIT_VALUE)
    if explicit_test_mask.any():
        train_idx = X.index[~(explicit_test_mask | explicit_val_mask)]
        val_idx = X.index[explicit_val_mask]
        test_idx = X.index[explicit_test_mask]
    elif split_file is not None:
        train_pos, val_pos, test_pos = split_positions_from_reference(
            split_file, target_col=TARGET_COLUMN, split_seed=split_seed, row_ids=X.index.to_numpy(dtype=int)
        )
        train_idx = X.index[train_pos]
        val_idx = X.index[val_pos]
        test_idx = X.index[test_pos]
    else:
        original_idx = X.index[is_generated.eq(0)]
        test_size = max(1, int(round(len(original_idx) * TEST_SIZE)))
        test_size = min(test_size, len(original_idx) - 1)
        train_idx, test_idx = train_test_split(original_idx, test_size=test_size, random_state=RANDOM_STATE)
        train_idx = pd.Index(train_idx)
        test_idx = pd.Index(test_idx)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"切分无效: train={len(train_idx)}, test={len(test_idx)}")
    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


def safe_mean(values: np.ndarray):
    return None if len(values) == 0 else float(np.mean(values))


def safe_metric(num: float, den: float):
    return None if den <= 0 else float(num / den)


def compute_high_pain_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true_arr - y_pred_arr)
    metrics = {
        "mae_true_0": safe_mean(abs_err[y_true_arr == 0]),
        "mae_true_1_3": safe_mean(abs_err[(y_true_arr >= 1) & (y_true_arr < 4)]),
        "mae_true_ge_4": safe_mean(abs_err[y_true_arr >= 4]),
        "bias_true_ge_4": safe_mean((y_pred_arr - y_true_arr)[y_true_arr >= 4]),
    }
    for threshold in (4.0, 5.0):
        true_high = y_true_arr >= threshold
        pred_high = y_pred_arr >= threshold
        tp = int(np.sum(true_high & pred_high))
        fp = int(np.sum(~true_high & pred_high))
        fn = int(np.sum(true_high & ~pred_high))
        precision = safe_metric(tp, tp + fp)
        recall = safe_metric(tp, tp + fn)
        f1 = None if precision is None or recall is None or precision + recall == 0 else float(2 * precision * recall / (precision + recall))
        key = str(int(threshold))
        metrics.update(
            {
                f"threshold_{key}_actual_positive_count": int(np.sum(true_high)),
                f"threshold_{key}_predicted_positive_count": int(np.sum(pred_high)),
                f"threshold_{key}_tp": tp,
                f"threshold_{key}_fp": fp,
                f"threshold_{key}_fn": fn,
                f"threshold_{key}_recall": recall,
                f"threshold_{key}_precision": precision,
                f"threshold_{key}_f1": f1,
                f"threshold_{key}_subset_mae": safe_mean(abs_err[true_high]),
                f"threshold_{key}_subset_bias": safe_mean((y_pred_arr - y_true_arr)[true_high]),
            }
        )
    return metrics


def refresh_model_registry_safely() -> None:
    if PROJECT_DIR is None:
        return
    try:
        from registry.model_output_registry import refresh_registry

        out_path = refresh_registry(project_dir=PROJECT_DIR)
        print(f"模型结果总表已刷新: {out_path}")
    except Exception as exc:
        print(f"警告：刷新模型结果总表失败: {exc}")


def main() -> None:
    args = parse_args()
    df = load_data(args.data_path)
    X, y, is_generated, split_labels = build_feature_target(df)
    train_idx, val_idx, test_idx = make_split(X, y, is_generated, split_labels, args.split_file, args.split_seed)

    model = DummyRegressor(strategy=args.strategy)
    model.fit(X.loc[train_idx], y.loc[train_idx])
    y_pred = model.predict(X.loc[test_idx])

    metrics = {
        "model_name": f"dummy_{args.strategy}_regressor",
        "target_column": TARGET_COLUMN,
        "target_tier": get_target_tier(TARGET_COLUMN),
        "source_data_path": str(args.data_path),
        "output_dir": str(args.output_dir),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_original_size": int(is_generated.loc[train_idx].eq(0).sum()),
        "train_generated_size": int(is_generated.loc[train_idx].eq(1).sum()),
        "val_original_size": int(is_generated.loc[val_idx].eq(0).sum()) if len(val_idx) else 0,
        "val_generated_size": int(is_generated.loc[val_idx].eq(1).sum()) if len(val_idx) else 0,
        "test_original_size": int(is_generated.loc[test_idx].eq(0).sum()),
        "test_generated_size": int(is_generated.loc[test_idx].eq(1).sum()),
        "baseline_strategy": args.strategy,
        "constant_prediction": float(model.constant_.ravel()[0]),
        "mae": float(mean_absolute_error(y.loc[test_idx], y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y.loc[test_idx], y_pred))),
        "r2": float(r2_score(y.loc[test_idx], y_pred)),
    }
    metrics.update(compute_high_pain_regression_metrics(y.loc[test_idx], y_pred))

    pred_df = pd.DataFrame(
        {
            "y_true": y.loc[test_idx].values,
            "y_pred": y_pred,
            "abs_error": np.abs(y.loc[test_idx].values - y_pred),
            "y_true_high_4": (y.loc[test_idx].values >= 4).astype(int),
            "y_pred_high_4": (y_pred >= 4).astype(int),
        }
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifact_dir / f"dummy_{args.strategy}_regressor.joblib"
    joblib.dump(model, model_path)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print("评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"模型文件: {model_path}")
    print(f"评估指标: {args.output_dir / 'metrics.json'}")
    print(f"预测明细: {args.output_dir / 'test_predictions.csv'}")
    refresh_model_registry_safely()


if __name__ == "__main__":
    main()
