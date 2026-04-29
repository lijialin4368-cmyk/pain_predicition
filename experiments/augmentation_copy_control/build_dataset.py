"""基于现有增强结果构造“直接复制样本”的对比组数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    AUGMENTED_DATASET_FILENAME,
    DUPLICATION_COUNTS_FILENAME,
    DUPLICATION_RULE_VALUE,
    DUPLICATION_TRIGGER_VALUE,
    DUPLICATION_VARIANT_PREFIX,
    GENERATED_ONLY_FILENAME,
    META_AGE_DELTA_COL,
    META_IS_GENERATED_COL,
    META_PAIN_SHIFT_COL,
    META_RULE_COL,
    META_SOURCE_ROW_ID_COL,
    META_SPLIT_COL,
    META_TRIGGER_COL,
    META_VARIANT_NAME_COL,
    META_WEIGHT_DELTA_COL,
    OUTPUT_DIR,
    REFERENCE_GENERATED_PATH,
    REFERENCE_SUMMARY_PATH,
    SUMMARY_FILENAME,
    TEST_ORIGINAL_FILENAME,
    TEST_ORIGINAL_PATH,
    TRAIN_ORIGINAL_FILENAME,
    TRAIN_ORIGINAL_PATH,
    TRAIN_SPLIT_VALUE,
    VALIDATION_ORIGINAL_FILENAME,
    VALIDATION_ORIGINAL_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a direct-copy comparison dataset matching the reference augmentation volume."
    )
    parser.add_argument(
        "--train-original-path",
        type=Path,
        default=TRAIN_ORIGINAL_PATH,
        help="Path to the reference train_original.csv.",
    )
    parser.add_argument(
        "--validation-original-path",
        type=Path,
        default=VALIDATION_ORIGINAL_PATH,
        help="Path to the reference validation_original.csv.",
    )
    parser.add_argument(
        "--test-original-path",
        type=Path,
        default=TEST_ORIGINAL_PATH,
        help="Path to the reference test_original.csv.",
    )
    parser.add_argument(
        "--reference-generated-path",
        type=Path,
        default=REFERENCE_GENERATED_PATH,
        help="Path to the reference generated_only.csv.",
    )
    parser.add_argument(
        "--reference-summary-path",
        type=Path,
        default=REFERENCE_SUMMARY_PATH,
        help="Path to the reference augmentation_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the comparison dataset outputs.",
    )
    return parser.parse_args()


def read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少输入文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"输入文件为空: {csv_path}")
    return df


def read_reference_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_source_ids(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"缺少关键列: {column}")
    out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    return out


def build_duplication_counts(reference_generated_df: pd.DataFrame) -> pd.DataFrame:
    counts_df = (
        reference_generated_df.groupby(META_SOURCE_ROW_ID_COL)
        .size()
        .rename("duplicate_count")
        .reset_index()
        .sort_values(META_SOURCE_ROW_ID_COL)
        .reset_index(drop=True)
    )
    return counts_df


def build_duplicated_rows(train_original_df: pd.DataFrame, counts_df: pd.DataFrame) -> pd.DataFrame:
    train_lookup = (
        train_original_df.drop_duplicates(subset=[META_SOURCE_ROW_ID_COL], keep="first")
        .set_index(META_SOURCE_ROW_ID_COL, drop=False)
    )

    missing_source_ids = sorted(set(counts_df[META_SOURCE_ROW_ID_COL]) - set(train_lookup.index))
    if missing_source_ids:
        preview = missing_source_ids[:10]
        raise ValueError(f"以下 source_row_id 未在 train_original 中找到: {preview}")

    generated_rows: list[dict] = []
    for row in counts_df.to_dict(orient="records"):
        source_row_id = int(row[META_SOURCE_ROW_ID_COL])
        duplicate_count = int(row["duplicate_count"])
        base_row = train_lookup.loc[source_row_id].copy()
        for duplicate_idx in range(1, duplicate_count + 1):
            duplicated_row = base_row.copy()
            duplicated_row[META_IS_GENERATED_COL] = 1
            duplicated_row[META_SPLIT_COL] = TRAIN_SPLIT_VALUE
            duplicated_row[META_RULE_COL] = DUPLICATION_RULE_VALUE
            duplicated_row[META_TRIGGER_COL] = DUPLICATION_TRIGGER_VALUE
            duplicated_row[META_PAIN_SHIFT_COL] = 0
            duplicated_row[META_AGE_DELTA_COL] = 0
            duplicated_row[META_WEIGHT_DELTA_COL] = 0
            duplicated_row[META_VARIANT_NAME_COL] = f"{DUPLICATION_VARIANT_PREFIX}_{duplicate_idx:04d}"
            generated_rows.append(duplicated_row.to_dict())

    if not generated_rows:
        return pd.DataFrame(columns=train_original_df.columns.tolist())

    return pd.DataFrame(generated_rows, columns=train_original_df.columns.tolist())


def build_summary(
    train_original_df: pd.DataFrame,
    validation_original_df: pd.DataFrame,
    test_original_df: pd.DataFrame,
    duplicated_generated_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    reference_generated_df: pd.DataFrame,
    reference_summary: dict,
    reference_summary_path: Path,
) -> dict:
    train_original_rows = int(len(train_original_df))
    generated_rows = int(len(duplicated_generated_df))
    total_train_rows = train_original_rows + generated_rows
    return {
        "comparison_strategy": "direct_duplicate",
        "reference_summary_path": str(reference_summary_path),
        "reference_generated_rows": int(len(reference_generated_df)),
        "reference_generated_rule_counts": (
            reference_generated_df[META_RULE_COL].value_counts().sort_index().to_dict()
            if META_RULE_COL in reference_generated_df.columns
            else {}
        ),
        "reference_augmented_parent_rows": int(len(counts_df)),
        "train_original_rows": train_original_rows,
        "validation_original_rows": int(len(validation_original_df)),
        "test_original_rows": int(len(test_original_df)),
        "generated_rows": generated_rows,
        "final_rows": int(train_original_rows + len(validation_original_df) + len(test_original_df) + generated_rows),
        "generated_to_original_ratio": (generated_rows / train_original_rows) if train_original_rows else None,
        "generated_fraction_in_training": (generated_rows / total_train_rows) if total_train_rows else None,
        "duplicate_count_per_parent_min": int(counts_df["duplicate_count"].min()) if not counts_df.empty else 0,
        "duplicate_count_per_parent_max": int(counts_df["duplicate_count"].max()) if not counts_df.empty else 0,
        "duplicate_count_per_parent_mean": float(counts_df["duplicate_count"].mean()) if not counts_df.empty else 0.0,
        "reference_summary_excerpt": {
            key: reference_summary[key]
            for key in [
                "source_rows",
                "train_original_rows",
                "validation_original_rows",
                "test_original_rows",
                "generated_rows",
                "final_rows",
            ]
            if key in reference_summary
        },
    }


def save_outputs(
    output_dir: Path,
    train_original_df: pd.DataFrame,
    validation_original_df: pd.DataFrame,
    test_original_df: pd.DataFrame,
    duplicated_generated_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_original_df.to_csv(output_dir / TRAIN_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    validation_original_df.to_csv(output_dir / VALIDATION_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    test_original_df.to_csv(output_dir / TEST_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    duplicated_generated_df.to_csv(output_dir / GENERATED_ONLY_FILENAME, index=False, encoding="utf-8-sig")
    combined_df.to_csv(output_dir / AUGMENTED_DATASET_FILENAME, index=False, encoding="utf-8-sig")
    counts_df.to_csv(output_dir / DUPLICATION_COUNTS_FILENAME, index=False, encoding="utf-8-sig")

    with (output_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    train_original_df = normalize_source_ids(read_csv(args.train_original_path), META_SOURCE_ROW_ID_COL)
    validation_original_df = normalize_source_ids(read_csv(args.validation_original_path), META_SOURCE_ROW_ID_COL)
    test_original_df = normalize_source_ids(read_csv(args.test_original_path), META_SOURCE_ROW_ID_COL)
    reference_generated_df = normalize_source_ids(read_csv(args.reference_generated_path), META_SOURCE_ROW_ID_COL)
    reference_summary = read_reference_summary(args.reference_summary_path)

    counts_df = build_duplication_counts(reference_generated_df)
    duplicated_generated_df = build_duplicated_rows(train_original_df, counts_df)
    combined_df = pd.concat(
        [train_original_df, duplicated_generated_df, validation_original_df, test_original_df],
        ignore_index=True,
    )

    summary = build_summary(
        train_original_df=train_original_df,
        validation_original_df=validation_original_df,
        test_original_df=test_original_df,
        duplicated_generated_df=duplicated_generated_df,
        counts_df=counts_df,
        reference_generated_df=reference_generated_df,
        reference_summary=reference_summary,
        reference_summary_path=args.reference_summary_path,
    )
    save_outputs(
        output_dir=args.output_dir,
        train_original_df=train_original_df,
        validation_original_df=validation_original_df,
        test_original_df=test_original_df,
        duplicated_generated_df=duplicated_generated_df,
        combined_df=combined_df,
        counts_df=counts_df,
        summary=summary,
    )

    print("对比组数据生成完成。")
    print(f"训练原始样本: {len(train_original_df)}")
    print(f"验证原始样本: {len(validation_original_df)}")
    print(f"固定测试集原始样本: {len(test_original_df)}")
    print(f"复制新增样本: {len(duplicated_generated_df)}")
    print(f"训练集新增样本占比: {summary['generated_fraction_in_training']:.6f}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
