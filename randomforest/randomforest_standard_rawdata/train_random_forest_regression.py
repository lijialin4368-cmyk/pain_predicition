"""使用标准随机森林完成原始数据回归任务（训练 + 评估 + 保存模型）。

运行方式：
    cd pain_prediction
    pixi run std-rf-raw-train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

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
TEST_SPLIT_VALUE = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a raw-data random forest regressor.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Input CSV path. Defaults to standard_rf_rawdata/config.py::DATA_PATH.",
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
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

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


def to_jsonable_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_hyperparameter_search(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict, pd.DataFrame]:
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
    search.fit(X_train, y_train)

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


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    is_generated: pd.Series,
    split_labels: pd.Series,
    source_row_ids: pd.Series,
) -> tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame | None]:
    explicit_test_mask = split_labels.eq(TEST_SPLIT_VALUE)
    if explicit_test_mask.any():
        test_idx = X.index[explicit_test_mask]
        train_candidate_idx = X.index[~explicit_test_mask]
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
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    if USE_HYPERPARAM_SEARCH:
        model, search_summary, cv_results_df = run_hyperparameter_search(X_train, y_train)
    else:
        model = build_rf_pipeline(RF_PARAMS)
        model.fit(X_train, y_train)
        search_summary = {
            "enabled": False,
            "best_params": {k: to_jsonable_scalar(v) for k, v in RF_PARAMS.items()},
        }
        cv_results_df = None

    y_pred = model.predict(X_test)

    metrics = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_original_size": int(is_generated.loc[train_idx].eq(0).sum()),
        "train_generated_size": int(is_generated.loc[train_idx].eq(1).sum()),
        "test_original_size": int(is_generated.loc[test_idx].eq(0).sum()),
        "test_generated_size": int(is_generated.loc[test_idx].eq(1).sum()),
        "hyperparameter_search": search_summary,
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

    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "abs_error": np.abs(y_test.values - y_pred),
        }
    )

    return model, metrics, pred_df, cv_results_df


def save_outputs(
    model: Pipeline,
    metrics: dict,
    pred_df: pd.DataFrame,
    X: pd.DataFrame,
    artifact_dir: Path,
    output_dir: Path,
    cv_results_df: pd.DataFrame | None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "rf_regressor.joblib"
    joblib.dump(model, model_path)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

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

    print("\n=== 训练完成，文件已保存 ===")
    print(f"模型文件: {model_path}")
    print(f"评估指标: {metrics_path}")
    print(f"预测明细: {pred_path}")
    print(f"特征重要性: {fi_path}")
    if cv_results_df is not None:
        print(f"调参结果: {output_dir / 'rf_search_results.csv'}")


def main() -> None:
    args = parse_args()

    df = load_data(args.data_path)
    print(f"读取数据成功: {args.data_path}")
    print(f"数据形状: {df.shape}")

    X, y, feature_cols, is_generated, split_labels, source_row_ids = build_feature_target(df)
    print(f"目标列: {TARGET_COLUMN}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"原始样本数: {int(is_generated.eq(0).sum())}")
    print(f"生成样本数: {int(is_generated.eq(1).sum())}")
    if split_labels.eq(TEST_SPLIT_VALUE).any():
        print(f"固定测试集原始样本数: {int(split_labels.eq(TEST_SPLIT_VALUE).sum())}")
    else:
        print(f"原始数据随机切分测试比例: {TEST_SIZE}")

    model, metrics, pred_df, cv_results_df = train_and_evaluate(X, y, is_generated, split_labels, source_row_ids)
    print("评估结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_outputs(model, metrics, pred_df, X, args.artifact_dir, args.output_dir, cv_results_df)


if __name__ == "__main__":
    main()
