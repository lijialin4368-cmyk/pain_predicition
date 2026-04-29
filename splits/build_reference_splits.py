from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from split_utils import (
    OUTCOME_DAYS,
    OUTCOME_METRICS,
    SHARED_BACKBONE_SPLIT_TARGET,
    get_positive_min,
    get_target_tier,
    make_binary_target_values,
    target_to_english_name,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build stratified 80/10/10 reference splits for outcome targets.")
    parser.add_argument("--input", type=Path, default=project_dir / "data" / "processed" / "data_vectorized.csv")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "splits")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated split seeds.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--pain-threshold", type=float, default=4.0)
    parser.add_argument("--max-tries", type=int, default=5000)
    return parser.parse_args()


def parse_seeds(seed_text: str) -> list[int]:
    return [int(s.strip()) for s in str(seed_text).split(",") if s.strip()]


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test 比例之和必须为 1，当前为 {total}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train/val/test 比例都必须大于 0。")


def allocate_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n < 3:
        n_test = 1 if n >= 2 else 0
        n_val = 0
        n_train = n - n_test
        return n_train, n_val, n_test

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_test = max(1, n_test)
    n_val = max(1, n_val)
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def stratified_split_binary(y: np.ndarray, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    rng = np.random.default_rng(seed)
    train_pos: list[int] = []
    val_pos: list[int] = []
    test_pos: list[int] = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train, n_val, n_test = allocate_counts(len(idx), train_ratio, val_ratio, test_ratio)
        test_pos.extend(idx[:n_test].tolist())
        val_pos.extend(idx[n_test : n_test + n_val].tolist())
        train_pos.extend(idx[n_test + n_val : n_test + n_val + n_train].tolist())

    rng.shuffle(train_pos)
    rng.shuffle(val_pos)
    rng.shuffle(test_pos)
    return np.array(train_pos, dtype=int), np.array(val_pos, dtype=int), np.array(test_pos, dtype=int)


def build_shared_split(y_mat: np.ndarray, seed: int, train_ratio: float, val_ratio: float, test_ratio: float, max_tries: int):
    n = y_mat.shape[0]
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_test + n_val >= n:
        raise ValueError("共享 split 的样本数不足。")

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        perm = rng.permutation(n)
        test_pos = perm[:n_test]
        val_pos = perm[n_test : n_test + n_val]
        train_pos = perm[n_test + n_val :]

        ok = True
        for h in range(y_mat.shape[1]):
            y_h = y_mat[:, h]
            if np.nansum(y_h == 1.0) < 3:
                continue
            for idx in (train_pos, val_pos, test_pos):
                y_split = y_h[idx]
                if np.nansum(y_split == 1.0) < 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return train_pos.astype(int), val_pos.astype(int), test_pos.astype(int)

    raise ValueError(f"无法在 {max_tries} 次尝试内生成共享 80/10/10 split，请检查极端少数类目标。")


def rows_for_target(df: pd.DataFrame, target_col: str, seed: int, args: argparse.Namespace) -> tuple[list[dict], dict]:
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = y_raw.notna().to_numpy()
    row_ids = df.index.to_numpy(dtype=int)[valid_mask]
    y_values = y_raw.to_numpy(dtype=float)[valid_mask]
    y_bin = make_binary_target_values(target_col, y_values, pain_threshold=args.pain_threshold).astype(int)

    train_pos, val_pos, test_pos = stratified_split_binary(
        y_bin,
        seed=seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    split_values = np.empty(len(y_bin), dtype=object)
    split_values[train_pos] = "train"
    split_values[val_pos] = "validation"
    split_values[test_pos] = "test"

    target_tier = get_target_tier(target_col)
    target_en = target_to_english_name(target_col)
    positive_min = get_positive_min(target_col, pain_threshold=args.pain_threshold)

    rows = [
        {
            "row_id": int(row_id),
            "target": target_col,
            "target_en": target_en,
            "target_tier": target_tier,
            "split_seed": int(seed),
            "split": str(split_values[i]),
            "y_value": float(y_values[i]),
            "y_binary": int(y_bin[i]),
            "positive_min": float(positive_min),
        }
        for i, row_id in enumerate(row_ids)
    ]
    summary = {
        "target": target_col,
        "target_en": target_en,
        "target_tier": target_tier,
        "split_seed": int(seed),
        "n_total": int(len(y_bin)),
        "n_positive": int(np.sum(y_bin == 1)),
        "positive_rate": float(np.mean(y_bin == 1)),
        "train_size": int(len(train_pos)),
        "validation_size": int(len(val_pos)),
        "test_size": int(len(test_pos)),
        "train_positive": int(np.sum(y_bin[train_pos] == 1)),
        "validation_positive": int(np.sum(y_bin[val_pos] == 1)),
        "test_positive": int(np.sum(y_bin[test_pos] == 1)),
    }
    return rows, summary


def rows_for_shared_split(df: pd.DataFrame, target_cols: list[str], seed: int, args: argparse.Namespace):
    y_cols = []
    for target_col in target_cols:
        y_raw = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        y_cols.append(make_binary_target_values(target_col, y_raw, pain_threshold=args.pain_threshold))
    y_mat_all = np.stack(y_cols, axis=1)
    any_label_mask = np.any(~np.isnan(y_mat_all), axis=1)
    row_ids = df.index.to_numpy(dtype=int)[any_label_mask]
    y_mat = y_mat_all[any_label_mask]

    train_pos, val_pos, test_pos = build_shared_split(
        y_mat,
        seed=seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_tries=args.max_tries,
    )
    split_values = np.empty(len(row_ids), dtype=object)
    split_values[train_pos] = "train"
    split_values[val_pos] = "validation"
    split_values[test_pos] = "test"
    rows = [
        {
            "row_id": int(row_id),
            "target": SHARED_BACKBONE_SPLIT_TARGET,
            "target_en": SHARED_BACKBONE_SPLIT_TARGET,
            "target_tier": "shared_multitask",
            "split_seed": int(seed),
            "split": str(split_values[i]),
            "y_value": "",
            "y_binary": "",
            "positive_min": "",
        }
        for i, row_id in enumerate(row_ids)
    ]
    summary = {
        "target": SHARED_BACKBONE_SPLIT_TARGET,
        "target_en": SHARED_BACKBONE_SPLIT_TARGET,
        "target_tier": "shared_multitask",
        "split_seed": int(seed),
        "n_total": int(len(row_ids)),
        "n_positive": "",
        "positive_rate": "",
        "train_size": int(len(train_pos)),
        "validation_size": int(len(val_pos)),
        "test_size": int(len(test_pos)),
        "train_positive": "",
        "validation_positive": "",
        "test_positive": "",
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    target_cols = [f"{day}_{metric}" for day in OUTCOME_DAYS for metric in OUTCOME_METRICS if f"{day}_{metric}" in df.columns]
    if not target_cols:
        raise ValueError("输入数据中未找到支持的结局列。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    all_summary_rows: list[dict] = []
    for seed in seeds:
        all_rows: list[dict] = []
        for target_col in target_cols:
            rows, summary = rows_for_target(df, target_col, seed, args)
            all_rows.extend(rows)
            all_summary_rows.append(summary)

        shared_rows, shared_summary = rows_for_shared_split(df, target_cols, seed, args)
        all_rows.extend(shared_rows)
        all_summary_rows.append(shared_summary)

        out_file = args.output_dir / f"reference_splits_seed_{seed}.csv"
        pd.DataFrame(all_rows).to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Saved split file: {out_file}")

    summary_file = args.output_dir / "reference_splits_summary.csv"
    pd.DataFrame(all_summary_rows).to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Saved split summary: {summary_file}")


if __name__ == "__main__":
    main()
