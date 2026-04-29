from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTCOME_DAYS = ["手术当天", "术后第一天", "术后第二天", "术后第三天"]
OUTCOME_METRICS = ["静息痛", "活动痛", "镇静评分", "活动状态", "恶心呕吐"]
SHARED_BACKBONE_SPLIT_TARGET = "__shared_backbone__"

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

METRIC_POSITIVE_MIN = {
    "静息痛": 4.0,
    "活动痛": 4.0,
    "镇静评分": 3.0,
    "活动状态": 3.0,
    "恶心呕吐": 2.0,
}

MAIN_TARGETS = {
    "手术当天_静息痛",
    "手术当天_活动痛",
    "手术当天_恶心呕吐",
    "术后第一天_静息痛",
    "术后第一天_活动痛",
    "术后第一天_恶心呕吐",
    "术后第二天_活动痛",
}

EXPLORATORY_TARGETS = {
    "术后第二天_静息痛",
    "术后第二天_恶心呕吐",
    "术后第三天_活动痛",
}


def split_day_metric(target_col: str) -> tuple[str, str]:
    for day in OUTCOME_DAYS:
        prefix = f"{day}_"
        if target_col.startswith(prefix):
            return day, target_col[len(prefix) :]
    return "", target_col


def target_to_english_name(target_col: str) -> str:
    day, metric = split_day_metric(target_col)
    if day:
        return f"{DAY_EN.get(day, day)}_{METRIC_EN.get(metric, metric)}"
    return target_col


def get_target_tier(target_col: str) -> str:
    day, metric = split_day_metric(target_col)
    if target_col in MAIN_TARGETS:
        return "main"
    if target_col in EXPLORATORY_TARGETS:
        return "exploratory"
    if metric == "活动状态":
        return "secondary_activity_status"
    if metric == "镇静评分":
        return "low_event_sedation"
    if day == "术后第三天" and metric in {"静息痛", "恶心呕吐"}:
        return "low_event"
    return "secondary"


def get_positive_min(target_col: str, pain_threshold: float = 4.0) -> float:
    _, metric = split_day_metric(target_col)
    if metric in {"静息痛", "活动痛"}:
        return float(pain_threshold)
    return float(METRIC_POSITIVE_MIN.get(metric, pain_threshold))


def make_binary_target_values(target_col: str, values, pain_threshold: float = 4.0) -> np.ndarray:
    scores = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    positive_min = get_positive_min(target_col, pain_threshold=pain_threshold)
    y = np.where(np.isnan(scores), np.nan, (scores >= positive_min).astype(float))
    return y


def split_positions_from_reference(
    split_file: str | Path,
    target_col: str,
    split_seed: int,
    row_ids,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path = Path(split_file).expanduser()
    split_df = pd.read_csv(split_path, encoding="utf-8-sig")
    required = {"row_id", "target", "split_seed", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"参考切分文件缺少列: {sorted(missing)}")

    target_rows = split_df[
        (split_df["target"].astype(str) == str(target_col))
        & (pd.to_numeric(split_df["split_seed"], errors="coerce").astype("Int64") == int(split_seed))
    ].copy()
    if target_rows.empty:
        raise ValueError(f"参考切分文件中找不到 target={target_col!r}, split_seed={split_seed} 的记录。")

    target_rows["row_id"] = pd.to_numeric(target_rows["row_id"], errors="raise").astype(int)
    split_by_row = dict(zip(target_rows["row_id"].tolist(), target_rows["split"].astype(str).tolist()))

    train_pos: list[int] = []
    val_pos: list[int] = []
    test_pos: list[int] = []
    missing_rows: list[int] = []
    for pos, row_id in enumerate(pd.Index(row_ids).astype(int).tolist()):
        split = split_by_row.get(int(row_id))
        if split == "train":
            train_pos.append(pos)
        elif split == "validation":
            val_pos.append(pos)
        elif split == "test":
            test_pos.append(pos)
        else:
            missing_rows.append(int(row_id))

    if missing_rows:
        preview = missing_rows[:10]
        raise ValueError(f"参考切分文件缺少 {len(missing_rows)} 个当前有效样本 row_id，示例: {preview}")
    if not train_pos or not test_pos:
        raise ValueError(f"参考切分无效: train={len(train_pos)}, validation={len(val_pos)}, test={len(test_pos)}")

    return (
        np.array(train_pos, dtype=int),
        np.array(val_pos, dtype=int),
        np.array(test_pos, dtype=int),
    )

