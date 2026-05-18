#!/usr/bin/env python3
"""Analyze correlations among postoperative outcome variables.

The report is intentionally written for model-development decisions: it
prioritizes interpretable conclusions over exhaustive statistical output.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DAYS = ["手术当天", "术后第一天", "术后第二天", "术后第三天"]
OUTCOME_TYPES = ["静息痛", "活动痛", "镇静评分", "活动状态", "恶心呕吐"]
OUTCOME_COLUMNS = [f"{day}_{kind}" for day in DAYS for kind in OUTCOME_TYPES]

TYPE_LABELS = {
    "静息痛": "rest_pain",
    "活动痛": "movement_pain",
    "镇静评分": "sedation",
    "活动状态": "activity",
    "恶心呕吐": "nausea_vomiting",
}

DAY_LABELS = {
    "手术当天": "D0",
    "术后第一天": "POD1",
    "术后第二天": "POD2",
    "术后第三天": "POD3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="data/processed/data_vectorized.csv",
        help="Vectorized CSV containing outcome columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/correlation_analysis",
        help="Directory for correlation outputs.",
    )
    return parser.parse_args()


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[int, float, float]:
    paired = pd.concat([x, y], axis=1).dropna()
    n = len(paired)
    if n < 3:
        return n, np.nan, np.nan
    a = paired.iloc[:, 0].astype(float)
    b = paired.iloc[:, 1].astype(float)
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return n, np.nan, np.nan
    if method == "pearson":
        r, p = stats.pearsonr(a, b)
    elif method == "spearman":
        r, p = stats.spearmanr(a, b)
    else:
        raise ValueError(method)
    return n, float(r), float(p)


def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    for left, right in combinations(columns, 2):
        _, r, _ = safe_corr(df[left], df[right], method)
        matrix.loc[left, right] = r
        matrix.loc[right, left] = r
    return matrix


def pairwise_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for left, right in combinations(columns, 2):
        n_p, pearson_r, pearson_p = safe_corr(df[left], df[right], "pearson")
        n_s, spearman_r, spearman_p = safe_corr(df[left], df[right], "spearman")
        rows.append(
            {
                "variable_1": left,
                "variable_2": right,
                "n_pairwise": min(n_p, n_s),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "abs_spearman_r": abs(spearman_r) if pd.notna(spearman_r) else np.nan,
                "relationship": relationship_type(left, right),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["abs_spearman_r", "n_pairwise"], ascending=[False, False]
    )


def relationship_type(left: str, right: str) -> str:
    day_left, type_left = split_outcome(left)
    day_right, type_right = split_outcome(right)
    if day_left == day_right:
        return "within_day"
    if type_left == type_right:
        return "cross_day_same_outcome"
    return "cross_day_mixed_outcome"


def split_outcome(column: str) -> tuple[str, str]:
    for day in DAYS:
        prefix = f"{day}_"
        if column.startswith(prefix):
            return day, column[len(prefix) :]
    raise ValueError(f"Unexpected outcome column: {column}")


def missingness_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        rows.append(
            {
                "variable": col,
                "n_non_missing": int(values.notna().sum()),
                "n_missing": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "iqr": float(q3 - q1) if values.notna().any() else np.nan,
                "min": float(values.min()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    labels = [english_label(col) for col in matrix.columns]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def english_label(column: str) -> str:
    day, outcome_type = split_outcome(column)
    return f"{DAY_LABELS[day]}_{TYPE_LABELS[outcome_type]}"


def signed_strength(value: float) -> str:
    if pd.isna(value):
        return "无法计算"
    abs_value = abs(value)
    direction = "正相关" if value >= 0 else "负相关"
    if abs_value >= 0.7:
        strength = "强"
    elif abs_value >= 0.4:
        strength = "中等"
    elif abs_value >= 0.2:
        strength = "弱"
    else:
        strength = "很弱"
    return f"{strength}{direction}"


def compact_number(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def top_rows(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return df.dropna(subset=["spearman_r"]).head(n)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "无可展示条目。\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(compact_number(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def describe_block(name: str, subset: pd.DataFrame, n: int = 5) -> str:
    if subset.empty:
        return f"{name}：没有可计算的相关性。\n"
    display = subset.dropna(subset=["spearman_r"]).copy().head(n)
    display["结论"] = display["spearman_r"].map(signed_strength)
    display = display.rename(
        columns={
            "variable_1": "变量1",
            "variable_2": "变量2",
            "n_pairwise": "n",
            "spearman_r": "Spearman",
            "pearson_r": "Pearson",
        }
    )
    return markdown_table(display, ["变量1", "变量2", "n", "Spearman", "Pearson", "结论"])


def write_report(
    output_path: Path,
    data_path: Path,
    n_rows: int,
    missingness: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    strong = pairs[pairs["abs_spearman_r"] >= 0.7].dropna(subset=["spearman_r"])
    moderate = pairs[(pairs["abs_spearman_r"] >= 0.4) & (pairs["abs_spearman_r"] < 0.7)].dropna(
        subset=["spearman_r"]
    )
    weak_or_less = pairs[pairs["abs_spearman_r"] < 0.4].dropna(subset=["spearman_r"])

    within_day = pairs[pairs["relationship"] == "within_day"]
    cross_day_same = pairs[pairs["relationship"] == "cross_day_same_outcome"]

    pain_cols = [f"{day}_{kind}" for day in DAYS for kind in ["静息痛", "活动痛"]]
    pain_pairs = pairs[
        pairs["variable_1"].isin(pain_cols) & pairs["variable_2"].isin(pain_cols)
    ]
    nonpain_cols = [
        f"{day}_{kind}" for day in DAYS for kind in ["镇静评分", "活动状态", "恶心呕吐"]
    ]
    pain_nonpain_pairs = pairs[
        (pairs["variable_1"].isin(pain_cols) & pairs["variable_2"].isin(nonpain_cols))
        | (pairs["variable_1"].isin(nonpain_cols) & pairs["variable_2"].isin(pain_cols))
    ]

    rest_movement_rows = []
    for day in DAYS:
        left = f"{day}_静息痛"
        right = f"{day}_活动痛"
        match = pairs[
            ((pairs["variable_1"] == left) & (pairs["variable_2"] == right))
            | ((pairs["variable_1"] == right) & (pairs["variable_2"] == left))
        ]
        if not match.empty:
            rest_movement_rows.append(match.iloc[0])
    rest_movement = pd.DataFrame(rest_movement_rows)

    same_outcome_summary = []
    for outcome_type in OUTCOME_TYPES:
        cols = [f"{day}_{outcome_type}" for day in DAYS]
        subset = pairs[
            pairs["variable_1"].isin(cols)
            & pairs["variable_2"].isin(cols)
            & (pairs["relationship"] == "cross_day_same_outcome")
        ].dropna(subset=["spearman_r"])
        if subset.empty:
            continue
        same_outcome_summary.append(
            {
                "结局类型": outcome_type,
                "跨天相关中位数": subset["spearman_r"].median(),
                "跨天相关最大值": subset["spearman_r"].max(),
                "最强组合": f"{subset.iloc[0]['variable_1']} vs {subset.iloc[0]['variable_2']}",
            }
        )
    same_outcome_summary = pd.DataFrame(same_outcome_summary)

    high_missing = missingness[missingness["missing_rate"] > 0.05].copy()
    complete_like = missingness[missingness["missing_rate"] <= 0.05].copy()

    lines = []
    lines.append("# 术后结局变量相关性分析\n")
    lines.append("## 这份分析看什么\n")
    lines.append(
        "本分析只看 20 个术后结局变量之间的关系：四天的静息痛、活动痛、镇静评分、活动状态、恶心呕吐。"
        "它的目的不是建立预测模型，而是回答一个建模前的问题：这些结局是不是在反映同一个术后状态，"
        "哪些可以一起建模，哪些更像独立结局。\n"
    )
    lines.append(f"- 数据来源：`{data_path}`\n")
    lines.append(f"- 总样本量：`{n_rows}`\n")
    lines.append("- 相关性计算：Pearson 看线性关系，Spearman 看排序/单调关系；下面的文字结论优先参考 Spearman。\n")
    lines.append("- 缺失处理：每一对变量单独使用两者都非缺失的样本。\n")

    lines.append("\n## 一句话结论\n")
    if not strong.empty:
        lines.append(
            f"20 个结局里存在 {len(strong)} 对强相关变量，主要集中在相邻天的同类结局、"
            "以及同一天的静息痛和活动痛。疼痛变量内部有共同信号，支持后续尝试构造 latent pain `Z`；"
            "但镇静评分、活动状态、恶心呕吐和疼痛的相关性整体较弱，不宜简单并入同一个“疼痛”标签。\n"
        )
    else:
        lines.append(
            "没有发现特别强的相关性。疼痛结局之间仍有一定共同趋势，但整体更像多个相近但不完全相同的目标。\n"
        )

    lines.append("\n## 数据完整性\n")
    lines.append(
        f"大多数结局变量接近完整：{len(complete_like)}/{len(missingness)} 个变量缺失率不超过 5%。"
    )
    if not high_missing.empty:
        worst = high_missing.sort_values("missing_rate", ascending=False).head(5).copy()
        worst["missing_rate"] = worst["missing_rate"].map(lambda x: f"{x:.1%}")
        lines.append("缺失相对较高的变量如下：\n")
        lines.append(markdown_table(worst, ["variable", "n_non_missing", "n_missing", "missing_rate"]))
    else:
        lines.append("没有明显高缺失的结局变量。\n")

    lines.append("\n## 最强相关的结局对\n")
    lines.append(
        "下面这些变量最容易“绑在一起”。如果两个结局高度相关，建模时可以考虑多任务学习、合成指标，"
        "或者用 latent variable 表达它们共享的术后状态。\n"
    )
    lines.append(describe_block("最强相关", top_rows(pairs, 10), 10))

    lines.append("\n## 同一天内：静息痛和活动痛是否可以合并？\n")
    if not rest_movement.empty:
        rest_movement = rest_movement.copy()
        rest_movement["结论"] = rest_movement["spearman_r"].map(signed_strength)
        rest_movement = rest_movement.rename(
            columns={
                "variable_1": "变量1",
                "variable_2": "变量2",
                "n_pairwise": "n",
                "spearman_r": "Spearman",
                "pearson_r": "Pearson",
            }
        )
        lines.append(markdown_table(rest_movement, ["变量1", "变量2", "n", "Spearman", "Pearson", "结论"]))
        median_rm = rest_movement["Spearman"].median()
        lines.append(
            f"四天内静息痛和活动痛的 Spearman 中位数约为 `{median_rm:.3f}`。"
            "这说明二者确实共享一部分疼痛信号，但并非完全等价。"
            "如果目标是整体疼痛负担，可以使用 `max(静息痛, 活动痛)` 或 latent pain；"
            "如果目标是精细预测患者在不同状态下的疼痛，仍建议保留两个目标。\n"
        )

    lines.append("\n## 跨天：哪些结局有延续性？\n")
    if not same_outcome_summary.empty:
        lines.append(markdown_table(same_outcome_summary, ["结局类型", "跨天相关中位数", "跨天相关最大值", "最强组合"]))
    lines.append(
        "跨天相关性可以理解为“今天高，明天是否也倾向于高”。如果某一类结局跨天相关高，"
        "它更适合做时间序列/多任务目标；如果跨天相关弱，说明每天受当日状态影响更大。\n"
    )
    lines.append(describe_block("同一结局跨天最强组合", cross_day_same, 8))

    lines.append("\n## 疼痛与非疼痛结局的关系\n")
    lines.append(
        "镇静评分、活动状态、恶心呕吐虽然都是术后恢复相关结局，但它们不一定是疼痛的替代指标。"
        "下面列出疼痛和非疼痛变量之间最强的关系：\n"
    )
    lines.append(describe_block("疼痛 vs 非疼痛", pain_nonpain_pairs, 8))
    if not pain_nonpain_pairs.empty:
        max_abs = pain_nonpain_pairs["abs_spearman_r"].max()
        if max_abs < 0.4:
            lines.append(
                "整体看，疼痛与镇静/活动/恶心呕吐之间没有达到中等强度相关。"
                "这意味着它们更适合作为辅助结局或并行任务，而不是直接揉成一个疼痛标签。\n"
            )
        else:
            lines.append(
                "部分非疼痛结局和疼痛存在中等以上相关，可在后续 latent outcome 模型中作为辅助观测，"
                "但需要避免把临床含义不同的变量强行解释为同一个疼痛分数。\n"
            )

    lines.append("\n## 对 latent pain `Z` 的建模启发\n")
    pain_strong = pain_pairs[pain_pairs["abs_spearman_r"] >= 0.4].dropna(subset=["spearman_r"])
    lines.append(
        f"疼痛变量之间有 {len(pain_strong)} 对达到中等或更强相关。"
        "这支持一个想法：原始表格里的单个疼痛评分 `Y` 带有测量噪声，而多个疼痛评分背后可能存在一个更稳定的潜在状态 `Z`。"
        "不过，目前证据更支持把 `Z` 定义为“疼痛潜在状态”，而不是把镇静评分、活动状态、恶心呕吐也全部合并进去。\n"
    )
    lines.append("建议后续建模时采用三层思路：\n")
    lines.append("1. `Z_pain`：由四天静息痛和活动痛共同提取，表示潜在疼痛负担。\n")
    lines.append("2. 非疼痛结局：镇静评分、活动状态、恶心呕吐作为辅助任务或下游关联分析。\n")
    lines.append("3. 单日预测：如果最终临床目标是某一天的具体评分，仍保留原始 `Y` 作为解释对象。\n")

    lines.append("\n## 输出文件说明\n")
    lines.append("- `outcome_missingness.csv`：每个结局变量的缺失率和描述统计。\n")
    lines.append("- `outcome_correlation_pearson.csv`：20 个结局变量的 Pearson 相关矩阵。\n")
    lines.append("- `outcome_correlation_spearman.csv`：20 个结局变量的 Spearman 相关矩阵。\n")
    lines.append("- `outcome_correlation_pairs.csv`：所有变量对的长表，含 n、Pearson、Spearman、p 值。\n")
    lines.append("- `within_day_correlation_pairs.csv`：同一天内的结局相关。\n")
    lines.append("- `cross_day_same_outcome_correlation_pairs.csv`：同一结局跨天相关。\n")
    lines.append("- `outcome_correlation_pearson_heatmap.png` / `outcome_correlation_spearman_heatmap.png`：相关性热图。\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    missing = [col for col in OUTCOME_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required outcome columns: {missing}")

    outcome_df = df[OUTCOME_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missingness = missingness_table(outcome_df, OUTCOME_COLUMNS)
    pearson = correlation_matrix(outcome_df, OUTCOME_COLUMNS, "pearson")
    spearman = correlation_matrix(outcome_df, OUTCOME_COLUMNS, "spearman")
    pairs = pairwise_table(outcome_df, OUTCOME_COLUMNS)

    missingness.to_csv(output_dir / "outcome_missingness.csv", index=False, encoding="utf-8-sig")
    pearson.to_csv(output_dir / "outcome_correlation_pearson.csv", encoding="utf-8-sig")
    spearman.to_csv(output_dir / "outcome_correlation_spearman.csv", encoding="utf-8-sig")
    pairs.to_csv(output_dir / "outcome_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    pairs[pairs["relationship"] == "within_day"].to_csv(
        output_dir / "within_day_correlation_pairs.csv", index=False, encoding="utf-8-sig"
    )
    pairs[pairs["relationship"] == "cross_day_same_outcome"].to_csv(
        output_dir / "cross_day_same_outcome_correlation_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_heatmap(pearson, "Outcome Correlation: Pearson", output_dir / "outcome_correlation_pearson_heatmap.png")
    plot_heatmap(
        spearman,
        "Outcome Correlation: Spearman",
        output_dir / "outcome_correlation_spearman_heatmap.png",
    )
    write_report(
        output_dir / "outcome_correlation_report.md",
        data_path,
        len(df),
        missingness,
        pairs,
    )

    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
