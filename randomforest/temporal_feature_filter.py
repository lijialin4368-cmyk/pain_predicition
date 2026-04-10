"""时间顺序特征过滤工具。

核心目标：避免用“目标时点及之后”的变量去预测当前目标，减少信息泄漏。
"""

from __future__ import annotations

from typing import Iterable


# 按时间先后定义可识别的前缀。
DAY_PREFIX_TO_INDEX = {
    "手术当天_": 0,
    "术后第一天_": 1,
    "术后第二天_": 2,
    "术后第三天_": 3,
}


def _get_day_index(col_name: str) -> int | None:
    """如果列名是时间相关列，返回其天数索引；否则返回 None。"""
    for prefix, idx in DAY_PREFIX_TO_INDEX.items():
        if col_name.startswith(prefix):
            return idx
    return None


def apply_temporal_feature_filter(
    *,
    all_columns: Iterable[str],
    target_column: str,
    manual_feature_columns: list[str] | None = None,
    strict_past_only: bool = True,
) -> tuple[list[str], list[str]]:
    """根据目标列的时间点筛选可用特征列。

    返回：
    - selected_features: 允许使用的特征列
    - dropped_features: 因时间规则被剔除的特征列

    规则（strict_past_only=True）：
    - 对于时间相关列，只允许使用“早于目标时点”的变量。
    - 非时间列（如年龄、ASA等）默认保留。
    """
    cols = list(all_columns)
    if target_column not in cols:
        raise ValueError(f"目标列不存在: {target_column}")

    if manual_feature_columns:
        missing = [c for c in manual_feature_columns if c not in cols]
        if missing:
            raise ValueError(f"手动指定的特征列不存在: {missing}")
        candidate_features = [c for c in manual_feature_columns if c != target_column]
    else:
        candidate_features = [c for c in cols if c != target_column]

    target_day_idx = _get_day_index(target_column)
    # 目标列不是时间相关字段时，不做时间过滤。
    if target_day_idx is None:
        return candidate_features, []

    selected_features: list[str] = []
    dropped_features: list[str] = []

    for col in candidate_features:
        col_day_idx = _get_day_index(col)
        if col_day_idx is None:
            # 非时间列默认保留。
            selected_features.append(col)
            continue

        if strict_past_only:
            keep = col_day_idx < target_day_idx
        else:
            keep = col_day_idx <= target_day_idx

        if keep:
            selected_features.append(col)
        else:
            dropped_features.append(col)

    return selected_features, dropped_features
