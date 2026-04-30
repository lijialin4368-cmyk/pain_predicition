"""Build a train-only BorderlineSMOTE dataset for one binary clinical target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

PROJECT_DIR = next((parent for parent in Path(__file__).resolve().parents if (parent / "pixi.toml").exists()), None)
if PROJECT_DIR is not None and str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from imblearn.over_sampling import BorderlineSMOTE
except ImportError as exc:
    raise ImportError("缺少 imbalanced-learn 依赖；请先运行 `pixi install` 或使用 pixi 环境。") from exc

from splits.split_utils import OUTCOME_DAYS, OUTCOME_METRICS, split_positions_from_reference

from config import (
    AUGMENTED_DATASET_FILENAME,
    DATA_PATH,
    GENERATED_ONLY_FILENAME,
    K_NEIGHBORS,
    KIND,
    M_NEIGHBORS,
    META_AUGMENTATION_METHOD_COL,
    META_BINARY_TARGET_COL,
    META_IS_GENERATED_COL,
    META_PREFIX,
    META_SOURCE_ROW_ID_COL,
    META_SPLIT_COL,
    META_TARGET_COLUMN_COL,
    META_TARGET_THRESHOLD_COL,
    OUTPUT_DIR,
    RANDOM_STATE,
    SAMPLING_STRATEGY,
    SPLIT_FILE,
    SPLIT_SEED,
    SUMMARY_FILENAME,
    TARGET_COLUMN,
    TARGET_THRESHOLD,
    TEST_ORIGINAL_FILENAME,
    TEST_SPLIT_VALUE,
    TRAIN_ORIGINAL_FILENAME,
    TRAIN_SPLIT_VALUE,
    VALIDATION_ORIGINAL_FILENAME,
    VALIDATION_SPLIT_VALUE,
)

METHOD_NAME = "borderline_smote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a train-only BorderlineSMOTE dataset.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--target-column", type=str, default=TARGET_COLUMN)
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    parser.add_argument("--m-neighbors", type=int, default=M_NEIGHBORS)
    parser.add_argument("--sampling-strategy", default=SAMPLING_STRATEGY)
    parser.add_argument("--kind", choices=["borderline-1", "borderline-2"], default=KIND)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("数据为空，无法做 BorderlineSMOTE。")
    return df


def outcome_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for day in OUTCOME_DAYS:
        for metric in OUTCOME_METRICS:
            col = f"{day}_{metric}"
            if col in df.columns:
                cols.append(col)
    return cols


def feature_columns_for_smote(df: pd.DataFrame, target_column: str) -> list[str]:
    outcomes = set(outcome_columns(df))
    features = [
        col
        for col in df.columns
        if col != target_column and col not in outcomes and not str(col).startswith(META_PREFIX)
    ]
    if not features:
        raise ValueError("没有可用于 BorderlineSMOTE 的非结局特征列。")
    return features


def attach_original_metadata(
    df: pd.DataFrame,
    split_label: str,
    *,
    target_column: str,
    target_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out[target_column], errors="coerce")
    out[META_IS_GENERATED_COL] = 0
    out[META_SPLIT_COL] = split_label
    out[META_SOURCE_ROW_ID_COL] = out.index.astype(int)
    out[META_AUGMENTATION_METHOD_COL] = "original"
    out[META_TARGET_COLUMN_COL] = target_column
    out[META_TARGET_THRESHOLD_COL] = float(target_threshold)
    out[META_BINARY_TARGET_COL] = (score >= float(target_threshold)).astype("Int64")
    return out


def build_generated_rows(
    source_columns: list[str],
    feature_cols: list[str],
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    *,
    target_column: str,
    target_threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    outcome_set = set(source_columns) - set(feature_cols)
    for i, (x_row, y_value) in enumerate(zip(generated_x, generated_y, strict=True)):
        row = {col: np.nan for col in source_columns}
        for col, value in zip(feature_cols, x_row, strict=True):
            row[col] = value
        # The synthetic row is intended for this one binary target. Keep other
        # outcomes missing so it is not accidentally reused as another target.
        for col in outcome_set:
            if str(col).startswith(META_PREFIX):
                continue
            if col != target_column:
                row[col] = np.nan
        row[target_column] = float(target_threshold) if int(y_value) == 1 else max(float(target_threshold) - 1.0, 0.0)
        row[META_IS_GENERATED_COL] = 1
        row[META_SPLIT_COL] = TRAIN_SPLIT_VALUE
        row[META_SOURCE_ROW_ID_COL] = -1
        row[META_AUGMENTATION_METHOD_COL] = METHOD_NAME
        row[META_TARGET_COLUMN_COL] = target_column
        row[META_TARGET_THRESHOLD_COL] = float(target_threshold)
        row[META_BINARY_TARGET_COL] = int(y_value)
        row["__meta_generated_row_id"] = i
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = load_data(args.data_path)
    if args.target_column not in df.columns:
        raise ValueError(f"目标列不存在: {args.target_column}")

    y_score_all = pd.to_numeric(df[args.target_column], errors="coerce")
    valid_idx = df.index[y_score_all.notna()]
    if len(valid_idx) < 10:
        raise ValueError("目标非缺失样本过少，无法做 BorderlineSMOTE。")

    train_pos, val_pos, test_pos = split_positions_from_reference(
        args.split_file,
        target_col=args.target_column,
        split_seed=args.split_seed,
        row_ids=valid_idx.to_numpy(dtype=int),
    )
    train_idx = valid_idx[train_pos]
    val_idx = valid_idx[val_pos]
    test_idx = valid_idx[test_pos]

    train_original = attach_original_metadata(
        df.loc[train_idx], TRAIN_SPLIT_VALUE, target_column=args.target_column, target_threshold=args.target_threshold
    )
    validation_original = attach_original_metadata(
        df.loc[val_idx], VALIDATION_SPLIT_VALUE, target_column=args.target_column, target_threshold=args.target_threshold
    )
    test_original = attach_original_metadata(
        df.loc[test_idx], TEST_SPLIT_VALUE, target_column=args.target_column, target_threshold=args.target_threshold
    )

    feature_cols = feature_columns_for_smote(df, args.target_column)
    x_train_df = train_original[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = (pd.to_numeric(train_original[args.target_column], errors="coerce") >= args.target_threshold).astype(int)
    class_counts = y_train.value_counts().sort_index().to_dict()
    if len(class_counts) < 2:
        raise ValueError(f"训练集中只有一个类别，无法做 BorderlineSMOTE: {class_counts}")

    min_class_count = int(min(class_counts.values()))
    k_neighbors = min(int(args.k_neighbors), max(1, min_class_count - 1))
    m_neighbors = min(int(args.m_neighbors), max(1, len(y_train) - 1))
    if k_neighbors < 1:
        raise ValueError(f"少数类样本过少，无法做 BorderlineSMOTE: {class_counts}")

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_df)
    smote = BorderlineSMOTE(
        sampling_strategy=args.sampling_strategy,
        random_state=args.random_state,
        k_neighbors=k_neighbors,
        m_neighbors=m_neighbors,
        kind=args.kind,
    )
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train.to_numpy(dtype=int))
    n_original_train = len(x_train)
    generated_x = x_resampled[n_original_train:]
    generated_y = y_resampled[n_original_train:]

    source_columns = list(df.columns)
    for meta_col in [
        META_IS_GENERATED_COL,
        META_SPLIT_COL,
        META_SOURCE_ROW_ID_COL,
        META_AUGMENTATION_METHOD_COL,
        META_TARGET_COLUMN_COL,
        META_TARGET_THRESHOLD_COL,
        META_BINARY_TARGET_COL,
        "__meta_generated_row_id",
    ]:
        if meta_col not in source_columns:
            source_columns.append(meta_col)

    generated_only = build_generated_rows(
        source_columns,
        feature_cols,
        generated_x,
        generated_y,
        target_column=args.target_column,
        target_threshold=args.target_threshold,
    )
    train_original = train_original.reindex(columns=source_columns)
    validation_original = validation_original.reindex(columns=source_columns)
    test_original = test_original.reindex(columns=source_columns)
    augmented_dataset = pd.concat(
        [train_original, generated_only, validation_original, test_original],
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_original.to_csv(args.output_dir / TRAIN_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    validation_original.to_csv(args.output_dir / VALIDATION_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    test_original.to_csv(args.output_dir / TEST_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    generated_only.to_csv(args.output_dir / GENERATED_ONLY_FILENAME, index=False, encoding="utf-8-sig")
    augmented_dataset.to_csv(args.output_dir / AUGMENTED_DATASET_FILENAME, index=False, encoding="utf-8-sig")

    summary = {
        "method": METHOD_NAME,
        "target_column": args.target_column,
        "target_threshold": float(args.target_threshold),
        "source_data_path": str(args.data_path),
        "split_file": str(args.split_file),
        "split_seed": int(args.split_seed),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "train_original_size": int(len(train_original)),
        "validation_original_size": int(len(validation_original)),
        "test_original_size": int(len(test_original)),
        "generated_size": int(len(generated_only)),
        "augmented_dataset_size": int(len(augmented_dataset)),
        "train_class_counts_before": {str(k): int(v) for k, v in class_counts.items()},
        "train_class_counts_after": {str(k): int(v) for k, v in pd.Series(y_resampled).value_counts().sort_index().to_dict().items()},
        "k_neighbors": int(k_neighbors),
        "m_neighbors": int(m_neighbors),
        "sampling_strategy": args.sampling_strategy,
        "kind": args.kind,
    }
    with (args.output_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("BorderlineSMOTE 数据集已生成:")
    print(f"  augmented_dataset: {args.output_dir / AUGMENTED_DATASET_FILENAME}")
    print(f"  generated_only    : {args.output_dir / GENERATED_ONLY_FILENAME}")
    print(f"  summary           : {args.output_dir / SUMMARY_FILENAME}")
    print(f"  train before      : {summary['train_class_counts_before']}")
    print(f"  train after       : {summary['train_class_counts_after']}")


if __name__ == "__main__":
    main()
