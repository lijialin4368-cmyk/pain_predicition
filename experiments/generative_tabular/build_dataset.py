"""Build CTGAN/TVAE train-only synthetic datasets for classification and regression."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = next((parent for parent in Path(__file__).resolve().parents if (parent / "pixi.toml").exists()), None)
if PROJECT_DIR is not None and str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from sdv.metadata import Metadata
except ImportError:
    Metadata = None

try:
    from sdv.metadata import SingleTableMetadata
except ImportError:
    SingleTableMetadata = None

try:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
except ImportError as exc:
    raise ImportError("缺少 sdv 依赖；请先运行 `pixi install` 或使用 pixi 环境。") from exc

try:
    from sdv.utils import load_synthesizer
except Exception:
    load_synthesizer = None

from splits.split_utils import split_positions_from_reference

from config import (
    AUGMENTED_DATASET_FILENAME,
    CLASSIFICATION_POSITIVE_MULTIPLIER,
    CTGAN_BATCH_SIZE,
    CTGAN_EPOCHS,
    DATA_PATH,
    DEFAULT_METHOD,
    DEFAULT_TASK,
    ENABLE_GPU,
    ENFORCE_MIN_MAX_VALUES,
    ENFORCE_ROUNDING,
    GENERATED_ONLY_FILENAME,
    MAX_SAMPLE_ATTEMPTS,
    METADATA_FILENAME,
    META_AUGMENTATION_METHOD_COL,
    META_BINARY_TARGET_COL,
    META_IS_GENERATED_COL,
    META_PREFIX,
    META_SOURCE_ROW_ID_COL,
    META_SPLIT_COL,
    META_TARGET_COLUMN_COL,
    META_TARGET_THRESHOLD_COL,
    META_TASK_COL,
    OUTPUT_DIR,
    RANDOM_STATE,
    REGRESSION_GENERATED_MULTIPLIER,
    SAMPLE_BATCH_SIZE,
    SPLIT_FILE,
    SPLIT_SEED,
    SUMMARY_FILENAME,
    SYNTHESIZER_FILENAME,
    TARGET_COLUMN,
    TARGET_THRESHOLD,
    TEST_ORIGINAL_FILENAME,
    TEST_SPLIT_VALUE,
    TRAIN_ORIGINAL_FILENAME,
    TRAIN_SPLIT_VALUE,
    TVAE_BATCH_SIZE,
    TVAE_EPOCHS,
    VALIDATION_ORIGINAL_FILENAME,
    VALIDATION_SPLIT_VALUE,
    VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-only CTGAN/TVAE dataset.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--method", choices=["ctgan", "tvae"], default=DEFAULT_METHOD)
    parser.add_argument("--task", choices=["classification", "regression"], default=DEFAULT_TASK)
    parser.add_argument("--target-column", type=str, default=TARGET_COLUMN)
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD)
    parser.add_argument("--positive-multiplier", type=float, default=CLASSIFICATION_POSITIVE_MULTIPLIER)
    parser.add_argument("--generated-multiplier", type=float, default=REGRESSION_GENERATED_MULTIPLIER)
    parser.add_argument("--sample-batch-size", type=int, default=SAMPLE_BATCH_SIZE)
    parser.add_argument("--max-sample-attempts", type=int, default=MAX_SAMPLE_ATTEMPTS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--enable-gpu", action="store_true", default=ENABLE_GPU)
    parser.add_argument("--verbose", action="store_true", default=VERBOSE)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("数据为空，无法做生成式增强。")
    return df


def non_meta_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not str(col).startswith(META_PREFIX)]


def infer_binary_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    binary_cols: list[str] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        values = set(series.dropna().unique().tolist())
        if values and values.issubset({0, 1}):
            binary_cols.append(col)
    return binary_cols


def detect_metadata(df: pd.DataFrame):
    if Metadata is not None:
        return Metadata.detect_from_dataframe(data=df, table_name="pain_prediction")
    if SingleTableMetadata is not None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        return metadata
    raise ImportError("无法导入 SDV metadata API。")


def save_metadata(metadata, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    if hasattr(metadata, "save_to_json"):
        metadata.save_to_json(str(output_path))
    else:
        output_path.write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def save_synthesizer(synthesizer, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    synthesizer.save(filepath=str(output_path))


def build_synthesizer(method: str, metadata, args: argparse.Namespace):
    epochs = args.epochs if args.epochs is not None else (CTGAN_EPOCHS if method == "ctgan" else TVAE_EPOCHS)
    batch_size = args.batch_size if args.batch_size is not None else (CTGAN_BATCH_SIZE if method == "ctgan" else TVAE_BATCH_SIZE)
    common_kwargs = {
        "enforce_min_max_values": ENFORCE_MIN_MAX_VALUES,
        "enforce_rounding": ENFORCE_ROUNDING,
        "epochs": epochs,
        "verbose": bool(args.verbose),
        "cuda": bool(args.enable_gpu),
        "batch_size": batch_size,
    }
    if method == "ctgan":
        return CTGANSynthesizer(metadata, **common_kwargs)
    return TVAESynthesizer(metadata, **common_kwargs)


def attach_original_metadata(
    df: pd.DataFrame,
    split_label: str,
    *,
    task: str,
    target_column: str,
    target_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    y_score = pd.to_numeric(out[target_column], errors="coerce")
    out[META_IS_GENERATED_COL] = 0
    out[META_SPLIT_COL] = split_label
    out[META_SOURCE_ROW_ID_COL] = out.index.astype(int)
    out[META_AUGMENTATION_METHOD_COL] = "original"
    out[META_TASK_COL] = task
    out[META_TARGET_COLUMN_COL] = target_column
    out[META_TARGET_THRESHOLD_COL] = float(target_threshold)
    out[META_BINARY_TARGET_COL] = (y_score >= float(target_threshold)).astype("Int64")
    return out


def round_binary_columns(df: pd.DataFrame, binary_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in binary_cols:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = np.where(values >= 0.5, 1, 0)
    return out


def filter_valid_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    y = pd.to_numeric(df[target_column], errors="coerce")
    return df.loc[y.notna()].copy()


def build_generated_classification_rows(
    synthesizer,
    train_df: pd.DataFrame,
    model_columns: list[str],
    binary_cols: list[str],
    *,
    target_column: str,
    target_threshold: float,
    positive_multiplier: float,
    sample_batch_size: int,
    max_attempts: int,
    task: str,
) -> pd.DataFrame:
    y_train = pd.to_numeric(train_df[target_column], errors="coerce")
    positive_count = int((y_train >= target_threshold).sum())
    target_generated_positive = int(math.ceil(max(0.0, positive_multiplier) * positive_count))
    if target_generated_positive <= 0:
        return pd.DataFrame(columns=train_df.columns)

    collected: list[pd.DataFrame] = []
    fallback_candidates: list[pd.DataFrame] = []
    generated_positive = 0
    for _ in range(max_attempts):
        sampled = synthesizer.sample(num_rows=sample_batch_size)
        sampled = round_binary_columns(sampled, binary_cols)
        sampled = filter_valid_target(sampled, target_column)
        if sampled.empty:
            continue
        fallback_candidates.append(sampled.copy())
        y_score = pd.to_numeric(sampled[target_column], errors="coerce")
        sampled = sampled.loc[y_score >= target_threshold].copy()
        if sampled.empty:
            continue
        collected.append(sampled)
        generated_positive += len(sampled)
        if generated_positive >= target_generated_positive:
            break

    if collected:
        generated = pd.concat(collected, ignore_index=True).head(target_generated_positive).copy()
    elif fallback_candidates:
        generated = pd.concat(fallback_candidates, ignore_index=True).copy()
        target_scores = pd.to_numeric(generated[target_column], errors="coerce")
        generated = generated.assign(__fallback_target_score=target_scores).sort_values(
            "__fallback_target_score", ascending=False
        ).drop(columns="__fallback_target_score").head(target_generated_positive).copy()
    else:
        return pd.DataFrame(columns=train_df.columns)

    generated[META_IS_GENERATED_COL] = 1
    generated[META_SPLIT_COL] = TRAIN_SPLIT_VALUE
    generated[META_SOURCE_ROW_ID_COL] = -1
    generated[META_AUGMENTATION_METHOD_COL] = synthesizer.__class__.__name__.replace("Synthesizer", "").lower()
    generated[META_TASK_COL] = task
    generated[META_TARGET_COLUMN_COL] = target_column
    generated[META_TARGET_THRESHOLD_COL] = float(target_threshold)
    y_generated = pd.to_numeric(generated[target_column], errors="coerce")
    generated[META_BINARY_TARGET_COL] = (y_generated >= float(target_threshold)).astype("Int64")
    return generated


def build_generated_regression_rows(
    synthesizer,
    train_df: pd.DataFrame,
    model_columns: list[str],
    binary_cols: list[str],
    *,
    target_column: str,
    target_threshold: float,
    generated_multiplier: float,
    task: str,
) -> pd.DataFrame:
    target_generated = int(math.ceil(max(0.0, generated_multiplier) * len(train_df)))
    if target_generated <= 0:
        return pd.DataFrame(columns=train_df.columns)
    generated = synthesizer.sample(num_rows=target_generated)
    generated = round_binary_columns(generated, binary_cols)
    generated = filter_valid_target(generated, target_column).head(target_generated).copy()
    generated[META_IS_GENERATED_COL] = 1
    generated[META_SPLIT_COL] = TRAIN_SPLIT_VALUE
    generated[META_SOURCE_ROW_ID_COL] = -1
    generated[META_AUGMENTATION_METHOD_COL] = synthesizer.__class__.__name__.replace("Synthesizer", "").lower()
    generated[META_TASK_COL] = task
    generated[META_TARGET_COLUMN_COL] = target_column
    generated[META_TARGET_THRESHOLD_COL] = float(target_threshold)
    generated[META_BINARY_TARGET_COL] = pd.Series([pd.NA] * len(generated), dtype="Int64")
    return generated


def loss_summary(synthesizer) -> dict:
    try:
        values = synthesizer.get_loss_values()
    except Exception:
        return {}
    if values is None or len(values) == 0:
        return {}
    tail = values.tail(1).to_dict(orient="records")[0]
    return {str(key): float(value) if isinstance(value, (int, float, np.generic)) else value for key, value in tail.items()}


def build_summary(
    args: argparse.Namespace,
    *,
    method_name: str,
    task: str,
    target_column: str,
    train_original: pd.DataFrame,
    validation_original: pd.DataFrame,
    test_original: pd.DataFrame,
    generated_only: pd.DataFrame,
    synthesizer_train_size: int,
    feature_columns: list[str],
    loss_info: dict,
) -> dict:
    summary = {
        "method": method_name,
        "task": task,
        "target_column": target_column,
        "target_threshold": float(args.target_threshold),
        "source_data_path": str(args.data_path),
        "split_file": str(args.split_file),
        "split_seed": int(args.split_seed),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "train_original_size": int(len(train_original)),
        "validation_original_size": int(len(validation_original)),
        "test_original_size": int(len(test_original)),
        "synthesizer_train_size": int(synthesizer_train_size),
        "generated_size": int(len(generated_only)),
        "augmented_dataset_size": int(len(train_original) + len(validation_original) + len(test_original) + len(generated_only)),
        "loss_summary": loss_info,
    }
    train_target = pd.to_numeric(train_original[target_column], errors="coerce")
    summary["train_target_mean"] = float(train_target.mean())
    summary["train_target_std"] = float(train_target.std()) if len(train_target) > 1 else 0.0
    if task == "classification":
        summary["positive_multiplier"] = float(args.positive_multiplier)
        summary["train_positive_before"] = int((train_target >= args.target_threshold).sum())
        summary["train_positive_after"] = int((pd.to_numeric(generated_only[target_column], errors="coerce") >= args.target_threshold).sum())
    else:
        summary["generated_multiplier"] = float(args.generated_multiplier)
    return summary


def main() -> None:
    args = parse_args()
    df = load_data(args.data_path)
    if args.target_column not in df.columns:
        raise ValueError(f"目标列不存在: {args.target_column}")

    y_score_all = pd.to_numeric(df[args.target_column], errors="coerce")
    valid_idx = df.index[y_score_all.notna()]
    if len(valid_idx) < 10:
        raise ValueError("目标非缺失样本过少，无法做 CTGAN/TVAE。")

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
        df.loc[train_idx], TRAIN_SPLIT_VALUE, task=args.task, target_column=args.target_column, target_threshold=args.target_threshold
    )
    validation_original = attach_original_metadata(
        df.loc[val_idx], VALIDATION_SPLIT_VALUE, task=args.task, target_column=args.target_column, target_threshold=args.target_threshold
    )
    test_original = attach_original_metadata(
        df.loc[test_idx], TEST_SPLIT_VALUE, task=args.task, target_column=args.target_column, target_threshold=args.target_threshold
    )

    model_columns = non_meta_columns(train_original)
    train_model_df = train_original[model_columns].copy()
    binary_cols = infer_binary_columns(train_model_df, model_columns)
    synth_train_df = train_model_df
    if args.task == "classification":
        y_train_score = pd.to_numeric(train_original[args.target_column], errors="coerce")
        positive_mask = y_train_score >= float(args.target_threshold)
        positive_train_df = train_model_df.loc[positive_mask].copy()
        if len(positive_train_df) >= 10:
            synth_train_df = positive_train_df
    metadata = detect_metadata(synth_train_df)
    synthesizer = build_synthesizer(args.method, metadata, args)
    synthesizer.fit(synth_train_df)

    if args.task == "classification":
        generated_only = build_generated_classification_rows(
            synthesizer,
            train_original,
            model_columns,
            binary_cols,
            target_column=args.target_column,
            target_threshold=args.target_threshold,
            positive_multiplier=args.positive_multiplier,
            sample_batch_size=args.sample_batch_size,
            max_attempts=args.max_sample_attempts,
            task=args.task,
        )
    else:
        generated_only = build_generated_regression_rows(
            synthesizer,
            train_original,
            model_columns,
            binary_cols,
            target_column=args.target_column,
            target_threshold=args.target_threshold,
            generated_multiplier=args.generated_multiplier,
            task=args.task,
        )

    all_columns = list(train_original.columns)
    generated_only = generated_only.reindex(columns=all_columns)
    train_original = train_original.reindex(columns=all_columns)
    validation_original = validation_original.reindex(columns=all_columns)
    test_original = test_original.reindex(columns=all_columns)
    augmented_dataset = pd.concat([train_original, generated_only, validation_original, test_original], ignore_index=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_original.to_csv(output_dir / TRAIN_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    validation_original.to_csv(output_dir / VALIDATION_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    test_original.to_csv(output_dir / TEST_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    generated_only.to_csv(output_dir / GENERATED_ONLY_FILENAME, index=False, encoding="utf-8-sig")
    augmented_dataset.to_csv(output_dir / AUGMENTED_DATASET_FILENAME, index=False, encoding="utf-8-sig")
    save_metadata(metadata, output_dir / METADATA_FILENAME)
    if hasattr(synthesizer, "save"):
        save_synthesizer(synthesizer, output_dir / SYNTHESIZER_FILENAME)

    summary = build_summary(
        args,
        method_name=args.method,
        task=args.task,
        target_column=args.target_column,
        train_original=train_original,
        validation_original=validation_original,
        test_original=test_original,
        generated_only=generated_only,
        synthesizer_train_size=len(synth_train_df),
        feature_columns=model_columns,
        loss_info=loss_summary(synthesizer),
    )
    with (output_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("生成式增强数据集已生成:")
    print(f"  method/task       : {args.method} / {args.task}")
    print(f"  augmented_dataset : {output_dir / AUGMENTED_DATASET_FILENAME}")
    print(f"  generated_only    : {output_dir / GENERATED_ONLY_FILENAME}")
    print(f"  summary           : {output_dir / SUMMARY_FILENAME}")
    print(f"  train/val/test    : {len(train_original)} / {len(validation_original)} / {len(test_original)}")
    print(f"  generated         : {len(generated_only)}")


if __name__ == "__main__":
    main()
