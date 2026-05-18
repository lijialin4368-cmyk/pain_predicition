"""使用随机森林完成回归任务（训练 + 评估 + 保存模型）。

运行方式：
    cd pain_prediction
    pixi run rf-train-raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

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
    RF_PARAMS,
    RF_SEARCH_CV,
    RF_SEARCH_N_ITER,
    RF_SEARCH_PARAM_DISTRIBUTIONS,
    RF_SEARCH_SCORING,
    STRICT_PAST_ONLY,
    TARGET_COLUMN,
    TEST_SIZE,
    USE_HYPERPARAM_SEARCH,
)
from temporal_feature_filter import apply_temporal_feature_filter

META_PREFIX = "__meta_"
META_IS_GENERATED_COL = "__meta_is_generated"
META_SPLIT_COL = "__meta_dataset_split"
META_SOURCE_ROW_ID_COL = "__meta_source_row_id"
VALIDATION_SPLIT_VALUE = "validation"
TEST_SPLIT_VALUE = "test"


def refresh_model_registry_safely() -> None:
    current_path = Path(__file__).resolve()
    project_dir = next((parent for parent in current_path.parents if (parent / "pixi.toml").exists()), None)
    if project_dir is None:
        return
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    try:
        from registry.model_output_registry import refresh_registry

        out_path = refresh_registry(project_dir=project_dir)
        print(f"模型结果总表已刷新: {out_path}")
    except Exception as exc:
        print(f"警告：刷新模型结果总表失败: {exc}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，允许临时覆盖数据与输出路径。"""
    parser = argparse.ArgumentParser(description="Train and evaluate a random forest regressor.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Input CSV path. Defaults to config.DATA_PATH; pass augmented data explicitly for augmentation experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for metrics/predictions/feature importance.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Directory for saved model artifact.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional reference split CSV. When provided, it overrides random train/test splitting for raw data.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Seed key to read from --split-file.")
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "high_pain", "focal_residual"],
        default="none",
        help="Tail-aware regression weighting. focal_residual does a second RF fit with residual-focused weights.",
    )
    parser.add_argument("--high-pain-threshold", type=float, default=4.0)
    parser.add_argument("--high-pain-weight", type=float, default=2.5)
    parser.add_argument("--severe-pain-threshold", type=float, default=7.0)
    parser.add_argument("--severe-pain-weight", type=float, default=4.0)
    parser.add_argument("--focal-alpha", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-final-sample-weight", type=float, default=10.0)
    parser.add_argument(
        "--enable-quantile-policy",
        action="store_true",
        help="Use validation data to choose a continuous RF tree-quantile prediction policy.",
    )
    parser.add_argument("--quantile-grid", type=str, default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument(
        "--prediction-policy-metric",
        choices=["mae", "high_pain_mae", "combined"],
        default="combined",
    )
    parser.add_argument("--tail-mae-weight", type=float, default=0.25)
    parser.add_argument("--underprediction-weight", type=float, default=0.25)
    parser.add_argument(
        "--enable-residual-calibration",
        action="store_true",
        help="Fit an isotonic residual calibration layer on validation predictions when it improves validation score.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    """读取数据并做最基础的校验。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    # encoding='utf-8-sig' 能更稳妥处理包含 BOM 的 CSV 头。
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("数据为空，无法训练模型。")
    return df


def is_metadata_column(col: str) -> bool:
    return str(col).startswith(META_PREFIX)


def select_balanced_augmented_train_indices(
    candidate_idx: pd.Index,
    source_row_ids: pd.Series,
    is_generated: pd.Series,
    random_state: int,
) -> tuple[np.ndarray, dict]:
    candidate_idx = pd.Index(candidate_idx)
    info = {
        "sampling_applied": False,
        "unaugmented_original_available": 0,
        "augmented_original_available": 0,
        "generated_available": 0,
        "augmented_pool_available": 0,
        "selected_unaugmented_original": 0,
        "selected_augmented_original": 0,
        "selected_generated": 0,
        "selected_augmented_pool": 0,
        "selected_total": int(len(candidate_idx)),
        "reason": "metadata_unavailable",
    }
    if len(candidate_idx) == 0:
        info["reason"] = "empty_train_pool"
        return candidate_idx.to_numpy(dtype=int), info

    train_source_ids = pd.to_numeric(source_row_ids.loc[candidate_idx], errors="coerce").fillna(-1).astype(int)
    train_is_generated = pd.to_numeric(is_generated.loc[candidate_idx], errors="coerce").fillna(0).astype(int).eq(1)
    generated_parent_ids = pd.Index(train_source_ids.loc[train_is_generated].unique())
    if len(generated_parent_ids) == 0:
        info["reason"] = "no_generated_samples"
        return candidate_idx.to_numpy(dtype=int), info

    original_mask = ~train_is_generated
    augmented_original_mask = original_mask & train_source_ids.isin(generated_parent_ids)
    unaugmented_original_mask = original_mask & ~augmented_original_mask
    augmented_pool_mask = augmented_original_mask | train_is_generated

    unaugmented_original_idx = candidate_idx[unaugmented_original_mask.to_numpy()]
    augmented_pool_idx = candidate_idx[augmented_pool_mask.to_numpy()]

    info.update(
        {
            "unaugmented_original_available": int(len(unaugmented_original_idx)),
            "augmented_original_available": int(augmented_original_mask.sum()),
            "generated_available": int(train_is_generated.sum()),
            "augmented_pool_available": int(len(augmented_pool_idx)),
            "reason": "ok",
        }
    )

    if len(unaugmented_original_idx) == 0 or len(augmented_pool_idx) == 0:
        info["reason"] = "one_side_empty"
        return candidate_idx.to_numpy(dtype=int), info

    rng = np.random.default_rng(random_state)
    sample_n = min(len(unaugmented_original_idx), len(augmented_pool_idx))
    sampled_augmented_idx = pd.Index(rng.choice(augmented_pool_idx.to_numpy(), size=sample_n, replace=False))
    sampled_augmented_generated = (
        pd.to_numeric(is_generated.loc[sampled_augmented_idx], errors="coerce").fillna(0).astype(int).eq(1)
    )

    selected_idx = np.concatenate([unaugmented_original_idx.to_numpy(), sampled_augmented_idx.to_numpy()]).astype(int)
    selected_idx = rng.permutation(selected_idx)

    info.update(
        {
            "sampling_applied": True,
            "selected_unaugmented_original": int(len(unaugmented_original_idx)),
            "selected_augmented_original": int((~sampled_augmented_generated).sum()),
            "selected_generated": int(sampled_augmented_generated.sum()),
            "selected_augmented_pool": int(sample_n),
            "selected_total": int(len(selected_idx)),
        }
    )
    return selected_idx.astype(int), info


def build_feature_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series, pd.Series, pd.Series]:
    """构建特征 X 与目标 y。"""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"目标列不存在: {TARGET_COLUMN}")

    # 目标列必须是数值（回归任务）。
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    # 用户可在 config.py 手动指定要用的特征列；也可启用时间防泄漏过滤。
    if ENABLE_TEMPORAL_FILTER:
        feature_cols, dropped_cols = apply_temporal_feature_filter(
            all_columns=df.columns.tolist(),
            target_column=TARGET_COLUMN,
            manual_feature_columns=MANUAL_FEATURE_COLUMNS,
            strict_past_only=STRICT_PAST_ONLY,
        )
        if dropped_cols:
            print(f"时间过滤已启用：剔除 {len(dropped_cols)} 个目标时点及之后的时间列。")
    else:
        if MANUAL_FEATURE_COLUMNS:
            missing = [c for c in MANUAL_FEATURE_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"手动指定的特征列不存在: {missing}")
            feature_cols = [c for c in MANUAL_FEATURE_COLUMNS if c != TARGET_COLUMN]
        else:
            feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    feature_cols = [c for c in feature_cols if not is_metadata_column(c)]

    X = df[feature_cols].copy()

    # 将所有特征转成数值；无法解析的值转 NaN，后续统一填充。
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 目标列的缺失行不能用于监督学习，直接过滤。
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
    if META_SOURCE_ROW_ID_COL in df.columns:
        source_row_ids = pd.to_numeric(df.loc[valid_mask, META_SOURCE_ROW_ID_COL], errors="coerce")
        fallback_ids = pd.Series(df.index, index=df.index, dtype="Int64").loc[valid_mask]
        source_row_ids = source_row_ids.fillna(fallback_ids).astype(int)
    else:
        source_row_ids = pd.Series(X.index, index=X.index, dtype=int)

    if len(X) < 10:
        raise ValueError("可用样本过少（<10），建议检查数据或目标列。")

    return X, y, feature_cols, is_generated, split_labels, source_row_ids


def build_rf_pipeline(rf_params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, **rf_params)),
        ]
    )


def fit_rf_pipeline(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray | None) -> Pipeline:
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, model__sample_weight=np.asarray(sample_weight, dtype=float))
    return model


def to_jsonable_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    search_rf_params = dict(RF_PARAMS)
    search_rf_params["n_jobs"] = -1
    pipeline = build_rf_pipeline(search_rf_params)
    cv = KFold(n_splits=RF_SEARCH_CV, shuffle=True, random_state=RANDOM_STATE)
    param_distributions = {f"model__{k}": v for k, v in RF_SEARCH_PARAM_DISTRIBUTIONS.items()}
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=RF_SEARCH_N_ITER,
        scoring=RF_SEARCH_SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=0,
        return_train_score=True,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = np.asarray(sample_weight, dtype=float)
    search.fit(X_train, y_train, **fit_kwargs)

    best_params = {
        key.replace("model__", ""): to_jsonable_scalar(value)
        for key, value in search.best_params_.items()
    }
    search_summary = {
        "enabled": True,
        "n_iter": int(RF_SEARCH_N_ITER),
        "cv_splits": int(RF_SEARCH_CV),
        "scoring": RF_SEARCH_SCORING,
        "best_score": float(search.best_score_),
        "best_params": best_params,
    }
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    return search.best_estimator_, search_summary, cv_results


def build_tail_sample_weight(y_train: pd.Series, args: argparse.Namespace) -> tuple[np.ndarray | None, dict]:
    mode = args.sample_weight_mode
    info = {
        "mode": mode,
        "high_pain_threshold": float(args.high_pain_threshold),
        "high_pain_weight": float(args.high_pain_weight),
        "severe_pain_threshold": float(args.severe_pain_threshold),
        "severe_pain_weight": float(args.severe_pain_weight),
        "high_pain_count": int((y_train >= args.high_pain_threshold).sum()),
        "severe_pain_count": int((y_train >= args.severe_pain_threshold).sum()),
        "mean_weight": 1.0,
        "max_weight": 1.0,
    }
    if mode == "none":
        return None, info

    y_arr = np.asarray(y_train, dtype=float)
    weights = np.ones(len(y_arr), dtype=float)
    high_weight = max(1.0, float(args.high_pain_weight))
    severe_weight = max(high_weight, float(args.severe_pain_weight))
    weights[y_arr >= float(args.high_pain_threshold)] = high_weight
    weights[y_arr >= float(args.severe_pain_threshold)] = severe_weight
    info["mean_weight"] = float(np.mean(weights))
    info["max_weight"] = float(np.max(weights))
    return weights, info


def build_residual_focus_weight(
    y_true: pd.Series,
    y_pred: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    abs_error = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    positive_scale = abs_error[abs_error > 0]
    scale = float(np.quantile(positive_scale, 0.75)) if len(positive_scale) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    normalized_error = np.clip(abs_error / scale, 0.0, 5.0)
    alpha = max(0.0, float(args.focal_alpha))
    gamma = max(0.0, float(args.focal_gamma))
    focus_weight = 1.0 + alpha * np.power(normalized_error, gamma)
    focus_weight = np.clip(focus_weight, 1.0, 10.0)
    return focus_weight, {
        "scale": scale,
        "alpha": alpha,
        "gamma": gamma,
        "mean_weight": float(np.mean(focus_weight)),
        "max_weight": float(np.max(focus_weight)),
    }


def parse_quantile_grid(quantile_grid: str) -> list[float]:
    quantiles: list[float] = []
    for item in str(quantile_grid).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value > 1.0:
            value = value / 100.0
        if 0.0 < value < 1.0:
            quantiles.append(value)
    return sorted(set(quantiles))


def tree_prediction_matrix(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    imputer = model.named_steps["imputer"]
    rf_model = model.named_steps["model"]
    X_imputed = imputer.transform(X)
    return np.column_stack([estimator.predict(X_imputed) for estimator in rf_model.estimators_])


def build_prediction_candidates(model: Pipeline, X: pd.DataFrame, quantiles: list[float]) -> dict[str, np.ndarray]:
    candidates = {"mean": np.asarray(model.predict(X), dtype=float)}
    if quantiles:
        tree_preds = tree_prediction_matrix(model, X)
        for quantile in quantiles:
            key = f"q{int(round(quantile * 100)):02d}"
            candidates[key] = np.quantile(tree_preds, quantile, axis=1)
    return candidates


def score_prediction_policy(y_true: pd.Series, y_pred: np.ndarray, args: argparse.Namespace) -> dict:
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_arr, pred_arr))
    high_mask = y_arr >= float(args.high_pain_threshold)
    if high_mask.any():
        high_mae = float(mean_absolute_error(y_arr[high_mask], pred_arr[high_mask]))
        high_bias = float(np.mean(pred_arr[high_mask] - y_arr[high_mask]))
        underprediction_penalty = max(0.0, -high_bias)
    else:
        high_mae = None
        high_bias = None
        underprediction_penalty = 0.0

    if args.prediction_policy_metric == "mae" or high_mae is None:
        selection_score = mae
    elif args.prediction_policy_metric == "high_pain_mae":
        selection_score = high_mae
    else:
        selection_score = mae + float(args.tail_mae_weight) * high_mae + float(args.underprediction_weight) * underprediction_penalty

    return {
        "mae": mae,
        "high_pain_mae": high_mae,
        "high_pain_bias": high_bias,
        "selection_score": float(selection_score),
    }


def choose_prediction_policy(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict, IsotonicRegression | None, pd.DataFrame | None]:
    base_policy = {
        "enabled": bool(args.enable_quantile_policy),
        "name": "mean",
        "metric": args.prediction_policy_metric,
        "calibration": "none",
        "validation_available": int(len(X_val)),
    }
    if len(X_val) == 0:
        return base_policy, None, None

    quantiles = parse_quantile_grid(args.quantile_grid) if args.enable_quantile_policy else []
    candidates = build_prediction_candidates(model, X_val, quantiles)
    rows = []
    best_name = "mean"
    best_score = None
    for name, pred in candidates.items():
        score = score_prediction_policy(y_val, pred, args)
        row = {"policy": name, "calibration": "none", **score}
        rows.append(row)
        if best_score is None or score["selection_score"] < best_score:
            best_score = score["selection_score"]
            best_name = name

    chosen_pred = candidates[best_name]
    calibrator = None
    calibration_score = None
    calibration_improved = False
    if args.enable_residual_calibration and len(X_val) >= 20 and len(np.unique(chosen_pred)) >= 2:
        candidate_calibrator = IsotonicRegression(out_of_bounds="clip")
        candidate_calibrator.fit(chosen_pred, np.asarray(y_val, dtype=float))
        calibrated_pred = candidate_calibrator.predict(chosen_pred)
        calibration_score = score_prediction_policy(y_val, calibrated_pred, args)
        rows.append({"policy": best_name, "calibration": "isotonic", **calibration_score})
        if best_score is None or calibration_score["selection_score"] <= best_score:
            calibrator = candidate_calibrator
            best_score = calibration_score["selection_score"]
            calibration_improved = True

    policy = {
        **base_policy,
        "name": best_name,
        "quantile_grid": quantiles,
        "calibration": "isotonic" if calibrator is not None else "none",
        "calibration_improved": calibration_improved,
        "validation_score": float(best_score) if best_score is not None else None,
    }
    if calibration_score is not None:
        policy["isotonic_validation_score"] = calibration_score
    return policy, calibrator, pd.DataFrame(rows)


def predict_with_policy(
    model: Pipeline,
    X: pd.DataFrame,
    policy: dict,
    calibrator: IsotonicRegression | None,
) -> np.ndarray:
    name = str(policy.get("name") or "mean")
    if name == "mean":
        pred = np.asarray(model.predict(X), dtype=float)
    elif name.startswith("q"):
        quantile = float(name.removeprefix("q")) / 100.0
        pred = np.quantile(tree_prediction_matrix(model, X), quantile, axis=1)
    else:
        pred = np.asarray(model.predict(X), dtype=float)

    if calibrator is not None:
        pred = np.asarray(calibrator.predict(pred), dtype=float)
    return pred


def safe_mean(values: np.ndarray):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def safe_metric(num: float, den: float):
    if den <= 0:
        return None
    return float(num / den)


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
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = float(2 * precision * recall / (precision + recall))
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


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    is_generated: pd.Series,
    split_labels: pd.Series,
    source_row_ids: pd.Series,
    split_file: Path | None,
    split_seed: int,
    args: argparse.Namespace,
) -> tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict, IsotonicRegression | None]:
    """拆分数据、训练模型并计算评估指标。"""
    val_idx = pd.Index([])
    explicit_test_mask = split_labels.eq(TEST_SPLIT_VALUE)
    explicit_val_mask = split_labels.eq(VALIDATION_SPLIT_VALUE)
    if explicit_test_mask.any():
        test_idx = X.index[explicit_test_mask]
        val_idx = X.index[explicit_val_mask]
        train_candidate_idx = X.index[~(explicit_test_mask | explicit_val_mask)]
        if len(train_candidate_idx) == 0:
            raise ValueError("显式切分后训练集为空，请检查 __meta_dataset_split 标记。")
    elif split_file is not None:
        train_pos, val_pos, test_pos = split_positions_from_reference(
            split_file,
            target_col=TARGET_COLUMN,
            split_seed=split_seed,
            row_ids=X.index.to_numpy(dtype=int),
        )
        train_candidate_idx = X.index[train_pos]
        val_idx = X.index[val_pos]
        test_idx = X.index[test_pos]
    else:
        original_mask = is_generated.eq(0)
        original_indices = X.index[original_mask]
        if len(original_indices) < 2:
            raise ValueError("原始样本不足，无法构建仅包含原始数据的测试集。")

        test_size_from_original = max(1, int(round(len(original_indices) * TEST_SIZE)))
        test_size_from_original = min(test_size_from_original, len(original_indices) - 1)
        train_candidate_idx, test_idx = train_test_split(
            original_indices,
            test_size=test_size_from_original,
            random_state=RANDOM_STATE,
        )

    train_idx, train_sampling_info = select_balanced_augmented_train_indices(
        candidate_idx=train_candidate_idx,
        source_row_ids=source_row_ids,
        is_generated=is_generated,
        random_state=RANDOM_STATE + 101,
    )
    train_idx = pd.Index(train_idx)

    X_train = X.loc[train_idx]
    X_val = X.loc[val_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]
    y_test = y.loc[test_idx]
    train_sample_weight, sample_weight_info = build_tail_sample_weight(y_train, args)
    if USE_HYPERPARAM_SEARCH:
        model, search_summary, cv_results_df = run_hyperparameter_search(X_train, y_train, train_sample_weight)
    else:
        model = build_rf_pipeline(RF_PARAMS)
        fit_rf_pipeline(model, X_train, y_train, train_sample_weight)
        search_summary = {
            "enabled": False,
            "best_params": {k: to_jsonable_scalar(v) for k, v in RF_PARAMS.items()},
        }
        cv_results_df = None

    residual_focus_info = None
    if args.sample_weight_mode == "focal_residual":
        initial_train_pred = model.predict(X_train)
        residual_focus_weight, residual_focus_info = build_residual_focus_weight(y_train, initial_train_pred, args)
        if train_sample_weight is None:
            final_sample_weight = residual_focus_weight
        else:
            final_sample_weight = np.asarray(train_sample_weight, dtype=float) * residual_focus_weight
        final_sample_weight = np.clip(final_sample_weight, 1.0, max(1.0, float(args.max_final_sample_weight)))
        final_params = dict(search_summary.get("best_params") or RF_PARAMS)
        final_params["n_jobs"] = -1
        model = build_rf_pipeline(final_params)
        fit_rf_pipeline(model, X_train, y_train, final_sample_weight)
        sample_weight_info["residual_focus"] = residual_focus_info
        sample_weight_info["final_mean_weight"] = float(np.mean(final_sample_weight))
        sample_weight_info["final_max_weight"] = float(np.max(final_sample_weight))
        search_summary["residual_focus_refit"] = True
    else:
        search_summary["residual_focus_refit"] = False

    policy_info, calibrator, policy_candidates_df = choose_prediction_policy(model, X_val, y_val, args)
    y_pred_mean = model.predict(X_test)
    y_pred = predict_with_policy(model, X_test, policy_info, calibrator)

    # 评估指标：MAE / RMSE / R^2
    metrics = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(X_test)),
        "train_original_size": int(is_generated.loc[train_idx].eq(0).sum()),
        "train_generated_size": int(is_generated.loc[train_idx].eq(1).sum()),
        "val_original_size": int(is_generated.loc[val_idx].eq(0).sum()) if len(val_idx) else 0,
        "val_generated_size": int(is_generated.loc[val_idx].eq(1).sum()) if len(val_idx) else 0,
        "test_original_size": int(is_generated.loc[test_idx].eq(0).sum()),
        "test_generated_size": int(is_generated.loc[test_idx].eq(1).sum()),
        "hyperparameter_search": search_summary,
        "target_tier": get_target_tier(TARGET_COLUMN),
        "sample_weighting": sample_weight_info,
        "prediction_policy": policy_info,
        "train_sampling_applied": bool(train_sampling_info["sampling_applied"]),
        "train_sampling_reason": train_sampling_info["reason"],
        "train_unaugmented_original_available": int(train_sampling_info["unaugmented_original_available"]),
        "train_augmented_original_available": int(train_sampling_info["augmented_original_available"]),
        "train_generated_available": int(train_sampling_info["generated_available"]),
        "train_augmented_pool_available": int(train_sampling_info["augmented_pool_available"]),
        "train_unaugmented_original_selected": int(train_sampling_info["selected_unaugmented_original"]),
        "train_augmented_original_selected": int(train_sampling_info["selected_augmented_original"]),
        "train_generated_selected": int(train_sampling_info["selected_generated"]),
        "train_augmented_pool_selected": int(train_sampling_info["selected_augmented_pool"]),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    metrics.update(compute_high_pain_regression_metrics(y_test, y_pred))

    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_pred_mean": y_pred_mean,
            "abs_error": np.abs(y_test.values - y_pred),
            "y_true_high_4": (y_test.values >= 4).astype(int),
            "y_pred_high_4": (y_pred >= 4).astype(int),
        }
    )

    return model, metrics, pred_df, cv_results_df, policy_candidates_df, policy_info, calibrator


def save_outputs(
    model: Pipeline,
    metrics: dict,
    pred_df: pd.DataFrame,
    X: pd.DataFrame,
    artifact_dir: Path,
    output_dir: Path,
    cv_results_df: pd.DataFrame | None,
    policy_candidates_df: pd.DataFrame | None,
    policy_info: dict,
    calibrator: IsotonicRegression | None,
) -> None:
    """保存模型、评估结果和特征重要性。"""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存训练后的模型
    model_path = artifact_dir / "rf_regressor.joblib"
    joblib.dump(model, model_path)
    bundle_path = artifact_dir / "rf_tail_aware_bundle.joblib"
    joblib.dump(
        {
            "model": model,
            "prediction_policy": policy_info,
            "calibrator": calibrator,
            "feature_columns": list(X.columns),
            "target_column": TARGET_COLUMN,
        },
        bundle_path,
    )

    # 2) 保存评估指标
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 3) 保存测试集预测明细
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 4) 保存特征重要性（用于解释哪些变量影响更大）
    rf_model = model.named_steps["model"]
    fi_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    fi_path = output_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    if cv_results_df is not None:
        cv_results_path = output_dir / "rf_search_results.csv"
        keep_cols = [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "param_model__n_estimators",
            "param_model__max_depth",
            "param_model__min_samples_split",
            "param_model__min_samples_leaf",
            "param_model__max_features",
            "param_model__max_samples",
        ]
        existing_cols = [col for col in keep_cols if col in cv_results_df.columns]
        cv_results_df.loc[:, existing_cols].to_csv(cv_results_path, index=False, encoding="utf-8-sig")

    policy_path = output_dir / "prediction_policy.json"
    with policy_path.open("w", encoding="utf-8") as f:
        json.dump(policy_info, f, ensure_ascii=False, indent=2)
    if policy_candidates_df is not None:
        policy_candidates_df.to_csv(output_dir / "prediction_policy_candidates.csv", index=False, encoding="utf-8-sig")

    print("\n=== 训练完成，文件已保存 ===")
    print(f"模型文件: {model_path}")
    print(f"增强推理包: {bundle_path}")
    print(f"评估指标: {metrics_path}")
    print(f"预测明细: {pred_path}")
    print(f"特征重要性: {fi_path}")
    if cv_results_df is not None:
        print(f"调参结果: {output_dir / 'rf_search_results.csv'}")
    print(f"预测策略: {policy_path}")


def main() -> None:
    args = parse_args()

    # Step 1) 读取数据
    df = load_data(args.data_path)
    print(f"读取数据成功: {args.data_path}")
    print(f"数据形状: {df.shape}")

    # Step 2) 构建训练输入
    X, y, feature_cols, is_generated, split_labels, source_row_ids = build_feature_target(df)
    print(f"目标列: {TARGET_COLUMN}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"原始样本数: {int(is_generated.eq(0).sum())}")
    print(f"生成样本数: {int(is_generated.eq(1).sum())}")
    if split_labels.eq(TEST_SPLIT_VALUE).any():
        print(f"固定测试集原始样本数: {int(split_labels.eq(TEST_SPLIT_VALUE).sum())}")
    elif args.split_file is not None:
        print(f"参考切分文件: {args.split_file} (seed={args.split_seed})")
    print("增强训练采样策略: 未增强原始样本全保留；从增强池随机抽取等量样本。")

    # Step 3) 训练并评估
    model, metrics, pred_df, cv_results_df, policy_candidates_df, policy_info, calibrator = train_and_evaluate(
        X,
        y,
        is_generated,
        split_labels,
        source_row_ids,
        split_file=args.split_file,
        split_seed=args.split_seed,
        args=args,
    )
    tail_aware_enabled = (
        args.sample_weight_mode != "none"
        or bool(args.enable_quantile_policy)
        or bool(args.enable_residual_calibration)
    )
    metrics["model_name"] = "randomforest_tail_aware_regressor" if tail_aware_enabled else "randomforest_regressor"
    metrics["target_column"] = TARGET_COLUMN
    metrics["source_data_path"] = str(args.data_path)
    metrics["output_dir"] = str(args.output_dir)
    print("评估结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Step 4) 保存输出
    save_outputs(
        model,
        metrics,
        pred_df,
        X,
        args.artifact_dir,
        args.output_dir,
        cv_results_df,
        policy_candidates_df,
        policy_info,
        calibrator,
    )
    refresh_model_registry_safely()


if __name__ == "__main__":
    main()
