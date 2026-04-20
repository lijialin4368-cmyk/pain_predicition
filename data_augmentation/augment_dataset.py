"""为 pain_prediction 生成固定原始保留集 + 规则增强后的训练数据。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    AGE_COL,
    AGE_HIGH_RISK_THRESHOLD,
    AUGMENTED_DATASET_FILENAME,
    ALL_PAIN_COLS,
    DATA_PATH,
    DEDUPLICATE_GENERATED_ROWS,
    DEMOGRAPHIC_DELTAS,
    GENERATED_KEEP_FRACTION,
    GENERATED_ONLY_FILENAME,
    GENDER_COL,
    HIGH_PAIN_KEEP_COUNT,
    MALE_CODE,
    META_AGE_DELTA_COL,
    META_IS_GENERATED_COL,
    META_PAIN_SHIFT_COL,
    META_RULE_COL,
    META_SOURCE_ROW_ID_COL,
    META_SPLIT_COL,
    META_TRIGGER_COL,
    META_VARIANT_NAME_COL,
    META_WEIGHT_DELTA_COL,
    MOVEMENT_PAIN_COLS,
    OUTPUT_DIR,
    PAIN_MAX_VALUE,
    PAIN_MIN_VALUE,
    PAIN_UPSHIFT_DELTA,
    RANDOM_STATE,
    REST_PAIN_COLS,
    RULE_COUNTS_FILENAME,
    SAME_DAY_HIGH_PAIN_THRESHOLD,
    SAME_DAY_TRIGGER_COLS,
    SUMMARY_FILENAME,
    TEST_ORIGINAL_FILENAME,
    TEST_SIZE,
    TEST_SPLIT_VALUE,
    TRAIN_ORIGINAL_FILENAME,
    TRAIN_SPLIT_VALUE,
    WAVE_DELTA,
    WAVE_TRIGGER_MIN_COUNT,
    WAVE_TRIGGER_THRESHOLD,
    WEIGHT_COL,
)

ORIGINAL_RULE_VALUE = "original"
HIGH_PAIN_ORIGINAL_RULE = "high_pain_original_jitter"
HIGH_PAIN_UPSHIFT_RULE = "high_pain_upshift_jitter"
REST_WAVE_UP_RULE = "rest_wave_up"
REST_WAVE_DOWN_RULE = "rest_wave_down"
MOVE_WAVE_UP_RULE = "movement_wave_up"
MOVE_WAVE_DOWN_RULE = "movement_wave_down"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an augmented dataset for pain_prediction.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Input CSV path. Defaults to data_augmentation/config.py::DATA_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of original rows reserved as held-out originals (for validation/test) before augmentation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used for train/test split and candidate sampling.",
    )
    parser.add_argument(
        "--generated-keep-fraction",
        type=float,
        default=GENERATED_KEEP_FRACTION,
        help="Keep fraction of generated rows after deduplication. Originals are always fully kept.",
    )
    return parser.parse_args()


def load_source_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("原始数据为空，无法做增强。")

    required_cols = {
        GENDER_COL,
        AGE_COL,
        WEIGHT_COL,
        *ALL_PAIN_COLS,
    }
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少增强所需字段: {missing_cols}")

    numeric_cols = [GENDER_COL, AGE_COL, WEIGHT_COL, *ALL_PAIN_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[META_SOURCE_ROW_ID_COL] = df.index.astype(int)
    return df


def attach_original_metadata(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    out = df.copy()
    out[META_IS_GENERATED_COL] = 0
    out[META_SPLIT_COL] = split_label
    out[META_RULE_COL] = ORIGINAL_RULE_VALUE
    out[META_TRIGGER_COL] = ORIGINAL_RULE_VALUE
    out[META_PAIN_SHIFT_COL] = 0
    out[META_AGE_DELTA_COL] = 0
    out[META_WEIGHT_DELTA_COL] = 0
    out[META_VARIANT_NAME_COL] = ORIGINAL_RULE_VALUE
    return out


def shift_pain_value(value: float, delta: float) -> float:
    if pd.isna(value):
        return value
    new_value = float(value) + delta
    if PAIN_MIN_VALUE is not None:
        new_value = max(PAIN_MIN_VALUE, new_value)
    if PAIN_MAX_VALUE is not None:
        new_value = min(PAIN_MAX_VALUE, new_value)
    return new_value


def shift_pain_columns(row: pd.Series, columns: list[str], delta: float) -> pd.Series:
    out = row.copy()
    for col in columns:
        out[col] = shift_pain_value(out[col], delta)
    return out


def make_generated_row(
    base_row: pd.Series,
    *,
    rule: str,
    trigger_reason: str,
    pain_shift: int,
    age_delta: int,
    weight_delta: int,
    variant_name: str,
) -> dict:
    row = base_row.copy()
    if pd.notna(row[AGE_COL]):
        row[AGE_COL] = float(row[AGE_COL]) + age_delta
    if pd.notna(row[WEIGHT_COL]):
        row[WEIGHT_COL] = float(row[WEIGHT_COL]) + weight_delta

    row[META_IS_GENERATED_COL] = 1
    row[META_SPLIT_COL] = TRAIN_SPLIT_VALUE
    row[META_RULE_COL] = rule
    row[META_TRIGGER_COL] = trigger_reason
    row[META_PAIN_SHIFT_COL] = pain_shift
    row[META_AGE_DELTA_COL] = age_delta
    row[META_WEIGHT_DELTA_COL] = weight_delta
    row[META_VARIANT_NAME_COL] = variant_name
    return row.to_dict()


def get_high_pain_trigger_reason(row: pd.Series) -> str | None:
    reason_tags: list[str] = []
    if pd.notna(row[GENDER_COL]) and float(row[GENDER_COL]) == float(MALE_CODE):
        reason_tags.append("male")
    if pd.notna(row[AGE_COL]) and float(row[AGE_COL]) > AGE_HIGH_RISK_THRESHOLD:
        reason_tags.append(f"age_gt_{AGE_HIGH_RISK_THRESHOLD}")
    if not reason_tags:
        return None

    same_day_values = [
        float(row[col])
        for col in SAME_DAY_TRIGGER_COLS
        if pd.notna(row[col])
    ]
    if not same_day_values or max(same_day_values) < SAME_DAY_HIGH_PAIN_THRESHOLD:
        return None

    reason_tags.append(f"same_day_pain_gte_{SAME_DAY_HIGH_PAIN_THRESHOLD}")
    return "|".join(reason_tags)


def generate_high_pain_rows(row: pd.Series, rng: np.random.Generator) -> list[dict]:
    trigger_reason = get_high_pain_trigger_reason(row)
    if trigger_reason is None:
        return []

    upshifted_row = shift_pain_columns(row, ALL_PAIN_COLS, PAIN_UPSHIFT_DELTA)
    candidates: list[dict] = []

    for age_delta in DEMOGRAPHIC_DELTAS:
        for weight_delta in DEMOGRAPHIC_DELTAS:
            if not (age_delta == 0 and weight_delta == 0):
                candidates.append(
                    make_generated_row(
                        row,
                        rule=HIGH_PAIN_ORIGINAL_RULE,
                        trigger_reason=trigger_reason,
                        pain_shift=0,
                        age_delta=age_delta,
                        weight_delta=weight_delta,
                        variant_name=f"orig_age_{age_delta:+d}_weight_{weight_delta:+d}",
                    )
                )
            candidates.append(
                make_generated_row(
                    upshifted_row,
                    rule=HIGH_PAIN_UPSHIFT_RULE,
                    trigger_reason=trigger_reason,
                    pain_shift=PAIN_UPSHIFT_DELTA,
                    age_delta=age_delta,
                    weight_delta=weight_delta,
                    variant_name=f"upshift_age_{age_delta:+d}_weight_{weight_delta:+d}",
                )
            )

    keep_count = min(HIGH_PAIN_KEEP_COUNT, len(candidates))
    keep_indices = np.sort(rng.choice(len(candidates), size=keep_count, replace=False))
    return [candidates[int(i)] for i in keep_indices]


def generate_wave_rows(row: pd.Series) -> list[dict]:
    generated_rows: list[dict] = []
    wave_specs = [
        ("rest", REST_PAIN_COLS, REST_WAVE_UP_RULE, REST_WAVE_DOWN_RULE),
        ("movement", MOVEMENT_PAIN_COLS, MOVE_WAVE_UP_RULE, MOVE_WAVE_DOWN_RULE),
    ]

    for label, cols, up_rule, down_rule in wave_specs:
        qualifying_count = sum(
            int(float(row[col]) > WAVE_TRIGGER_THRESHOLD)
            for col in cols
            if pd.notna(row[col])
        )
        if qualifying_count < WAVE_TRIGGER_MIN_COUNT:
            continue

        trigger_reason = f"{label}_pain_{WAVE_TRIGGER_MIN_COUNT}of4_gt_{WAVE_TRIGGER_THRESHOLD}"
        up_row = shift_pain_columns(row, cols, WAVE_DELTA)
        down_row = shift_pain_columns(row, cols, -WAVE_DELTA)

        generated_rows.append(
            make_generated_row(
                up_row,
                rule=up_rule,
                trigger_reason=trigger_reason,
                pain_shift=WAVE_DELTA,
                age_delta=0,
                weight_delta=0,
                variant_name=f"{label}_wave_up",
            )
        )
        generated_rows.append(
            make_generated_row(
                down_row,
                rule=down_rule,
                trigger_reason=trigger_reason,
                pain_shift=-WAVE_DELTA,
                age_delta=0,
                weight_delta=0,
                variant_name=f"{label}_wave_down",
            )
        )

    return generated_rows


def build_generated_dataset(train_source_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    generated_rows: list[dict] = []

    for _, row in train_source_df.iterrows():
        generated_rows.extend(generate_high_pain_rows(row, rng))
        generated_rows.extend(generate_wave_rows(row))

    if not generated_rows:
        return pd.DataFrame(columns=train_source_df.columns.tolist())

    return pd.DataFrame(generated_rows)


def postprocess_generated_dataset(
    generated_df: pd.DataFrame,
    *,
    keep_fraction: float,
    deduplicate_generated: bool,
    random_state: int,
    non_meta_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    stats = {
        "raw_generated_rows": int(len(generated_df)),
        "deduplicated_generated_rows": int(len(generated_df)),
        "generated_keep_fraction": float(keep_fraction),
        "final_generated_rows": int(len(generated_df)),
        "deduplicate_generated_rows": bool(deduplicate_generated),
    }
    if generated_df.empty:
        return generated_df.copy(), stats

    out = generated_df.copy()
    if deduplicate_generated:
        out = out.drop_duplicates(subset=non_meta_columns, keep="first").reset_index(drop=True)
        stats["deduplicated_generated_rows"] = int(len(out))

    if not (0 < keep_fraction <= 1):
        raise ValueError("--generated-keep-fraction 必须在 (0, 1] 范围内。")

    if keep_fraction < 1.0 and len(out) > 0:
        keep_n = max(1, int(round(len(out) * keep_fraction)))
        keep_n = min(keep_n, len(out))
        out = out.sample(n=keep_n, random_state=random_state, replace=False).reset_index(drop=True)

    stats["final_generated_rows"] = int(len(out))
    return out, stats


def build_summary(
    source_df: pd.DataFrame,
    train_original_df: pd.DataFrame,
    test_original_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    generated_postprocess_stats: dict,
) -> dict:
    generated_rule_counts = Counter()
    generated_parent_counts = Counter()
    if not generated_df.empty:
        generated_rule_counts.update(generated_df[META_RULE_COL].value_counts().to_dict())
        generated_parent_counts.update(
            generated_df.groupby(META_RULE_COL)[META_SOURCE_ROW_ID_COL].nunique().to_dict()
        )

    return {
        "source_rows": int(len(source_df)),
        "heldout_size_fraction": float(test_size),
        "test_size_fraction": float(test_size),
        "random_state": int(random_state),
        "train_original_rows": int(len(train_original_df)),
        "heldout_original_rows": int(len(test_original_df)),
        "test_original_rows": int(len(test_original_df)),
        "generated_rows": int(len(generated_df)),
        "final_rows": int(len(train_original_df) + len(test_original_df) + len(generated_df)),
        "male_code": int(MALE_CODE),
        "same_day_trigger_columns": SAME_DAY_TRIGGER_COLS,
        "high_pain_total_candidates_per_parent": int((len(DEMOGRAPHIC_DELTAS) ** 2) * 2 - 1),
        "high_pain_kept_per_parent": int(HIGH_PAIN_KEEP_COUNT),
        "generated_postprocess": generated_postprocess_stats,
        "generated_rule_counts": dict(sorted(generated_rule_counts.items())),
        "generated_parent_counts": dict(sorted(generated_parent_counts.items())),
    }


def save_outputs(
    output_dir: Path,
    train_original_df: pd.DataFrame,
    test_original_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_original_df.to_csv(output_dir / TRAIN_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    test_original_df.to_csv(output_dir / TEST_ORIGINAL_FILENAME, index=False, encoding="utf-8-sig")
    generated_df.to_csv(output_dir / GENERATED_ONLY_FILENAME, index=False, encoding="utf-8-sig")
    combined_df.to_csv(output_dir / AUGMENTED_DATASET_FILENAME, index=False, encoding="utf-8-sig")

    rule_counts_df = pd.DataFrame(
        [
            {"rule": key, "count": value}
            for key, value in summary["generated_rule_counts"].items()
        ]
    )
    rule_counts_df.to_csv(output_dir / RULE_COUNTS_FILENAME, index=False, encoding="utf-8-sig")

    with (output_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if not (0 < args.test_size < 1):
        raise ValueError("--test-size 必须在 0 和 1 之间。")

    source_df = load_source_data(args.data_path)
    train_idx, test_idx = train_test_split(
        source_df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    train_source_df = source_df.loc[train_idx].copy()
    test_source_df = source_df.loc[test_idx].copy()
    meta_columns = [
        META_SOURCE_ROW_ID_COL,
        META_IS_GENERATED_COL,
        META_SPLIT_COL,
        META_RULE_COL,
        META_TRIGGER_COL,
        META_PAIN_SHIFT_COL,
        META_AGE_DELTA_COL,
        META_WEIGHT_DELTA_COL,
        META_VARIANT_NAME_COL,
    ]
    non_meta_columns = [col for col in source_df.columns if col not in meta_columns]

    train_original_df = attach_original_metadata(train_source_df, TRAIN_SPLIT_VALUE)
    test_original_df = attach_original_metadata(test_source_df, TEST_SPLIT_VALUE)
    generated_df = build_generated_dataset(train_source_df, args.random_state)
    generated_df, generated_postprocess_stats = postprocess_generated_dataset(
        generated_df,
        keep_fraction=args.generated_keep_fraction,
        deduplicate_generated=DEDUPLICATE_GENERATED_ROWS,
        random_state=args.random_state,
        non_meta_columns=non_meta_columns,
    )
    combined_columns = [col for col in source_df.columns if col not in meta_columns] + meta_columns

    if generated_df.empty:
        generated_df = pd.DataFrame(columns=combined_columns)
    else:
        generated_df = generated_df.reindex(columns=combined_columns)

    train_original_df = train_original_df.reindex(columns=combined_columns)
    test_original_df = test_original_df.reindex(columns=combined_columns)
    combined_df = pd.concat(
        [train_original_df, generated_df, test_original_df],
        ignore_index=True,
    )

    summary = build_summary(
        source_df=source_df,
        train_original_df=train_original_df,
        test_original_df=test_original_df,
        generated_df=generated_df,
        test_size=args.test_size,
        random_state=args.random_state,
        generated_postprocess_stats=generated_postprocess_stats,
    )
    save_outputs(
        output_dir=args.output_dir,
        train_original_df=train_original_df,
        test_original_df=test_original_df,
        generated_df=generated_df,
        combined_df=combined_df,
        summary=summary,
    )

    print("增强数据生成完成。")
    print(f"原始样本: {len(source_df)}")
    print(f"训练原始样本: {len(train_original_df)}")
    print(f"预留原始样本(validation/test 池): {len(test_original_df)}")
    print(f"生成样本: {len(generated_df)}")
    print(f"生成样本后处理: {generated_postprocess_stats}")
    print(f"合并后总样本: {len(combined_df)}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
