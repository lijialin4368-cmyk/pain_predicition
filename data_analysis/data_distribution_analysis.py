import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


DRUG_MAP = {
    "舒芬": "Sufentanil",
    "羟考酮": "Oxycodone",
    "曲马多": "Tramadol",
    "帕洛诺司琼": "Palonosetron",
    "布托啡诺": "Butorphanol",
    "纳布啡": "Nalbuphine",
    "地佐辛": "Dezocine",
    "罗哌卡因": "Ropivacaine",
    "布比卡因脂质体": "Liposomal Bupivacaine",
    "布比卡因": "Bupivacaine",
}

OUTCOME_DAY_PREFIXES = ["手术当天", "术后第一天", "术后第二天", "术后第三天"]
OUTCOME_METRICS = {
    "静息痛": "Rest Pain",
    "活动痛": "Movement Pain",
    "镇静评分": "Sedation Score",
    "活动状态": "Activity Status Score",
    "恶心呕吐": "Nausea/Vomiting",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate optimized grouped distribution plots and tables."
    )
    parser.add_argument(
        "--input",
        default="data_vectorized.csv",
        help="Input CSV file path. Default: data_vectorized.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="distribution_analysis",
        help="Output directory. Default: distribution_analysis",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=30,
        help="Max categories shown in regular categorical bar chart.",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=30,
        help="Max bins for numeric histograms.",
    )
    parser.add_argument(
        "--discrete-unique-threshold",
        type=int,
        default=20,
        help="If numeric unique values <= this threshold, plot as category bars.",
    )
    return parser.parse_args()


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read csv file: {path}")


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(name)).strip("_")
    return cleaned if cleaned else "column"


def to_english_col_name(col: str) -> str:
    exact = {
        "性别_num": "Sex",
        "ASA分级_num": "ASA Grade",
        "总量": "Pump Total Volume (mL)",
        "背景量": "Pump Basal Rate (mL/h)",
        "年份": "Year",
        "年龄": "Age",
        "体重": "Weight (kg)",
        "手术月份": "Surgery Month",
        "手术星期": "Surgery Weekday",
        "阻滞镇痛用药_总量": "Block Drug Total Dose",
        "阻滞镇痛用药_总体积_ml": "Block Drug Total Volume (mL)",
    }
    if col in exact:
        return exact[col]

    if col.startswith("镇痛泵配方_") and col.endswith("_mg"):
        drug = col.replace("镇痛泵配方_", "", 1).replace("_mg", "")
        return f"Pump Dose - {DRUG_MAP.get(drug, drug)} (mg)"

    if col.startswith("阻滞镇痛用药_") and col.endswith("_dose"):
        drug = col.replace("阻滞镇痛用药_", "", 1).replace("_dose", "")
        return f"Block Drug Dose - {DRUG_MAP.get(drug, drug)}"

    if col.startswith("阻滞镇痛用药_has_"):
        drug = col.replace("阻滞镇痛用药_has_", "", 1)
        return f"Block Drug Contains - {DRUG_MAP.get(drug, drug)}"

    return col


def format_value_text(x):
    if pd.isna(x):
        return "<NA>"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(x))
        return f"{float(x):.3f}".rstrip("0").rstrip(".")
    return str(x)


def build_count_table(counts: pd.Series, denominator: int, category_name: str = "category") -> pd.DataFrame:
    out = counts.rename_axis(category_name).reset_index(name="count")
    out[category_name] = out[category_name].map(format_value_text)
    if denominator <= 0:
        out["ratio"] = 0.0
    else:
        out["ratio"] = out["count"] / float(denominator)
    return out


def save_count_table(counts: pd.Series, denominator: int, out_file: Path, category_name: str = "category"):
    table = build_count_table(counts, denominator, category_name)
    table.to_csv(out_file, index=False, encoding="utf-8-sig")


def write_histogram_table_numeric(series: pd.Series, out_file: Path, max_bins: int):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        pd.DataFrame(columns=["bin_left", "bin_right", "count", "ratio"]).to_csv(
            out_file, index=False, encoding="utf-8-sig"
        )
        return

    bins = min(max_bins, max(10, int(np.sqrt(len(numeric)))))
    counts, edges = np.histogram(numeric, bins=bins)
    hist_df = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "ratio": counts / float(len(series)),
        }
    )
    hist_df.to_csv(out_file, index=False, encoding="utf-8-sig")


def annotate_vertical_bars(ax, bars, counts: np.ndarray, denominator: int):
    for bar, count in zip(bars, counts):
        ratio = 0.0 if denominator <= 0 else (count / float(denominator))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)} ({ratio * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def annotate_horizontal_bars(ax, bars, counts: np.ndarray, denominator: int):
    max_count = float(max(counts)) if len(counts) else 0.0
    offset = max(1.0, max_count * 0.01)
    for bar, count in zip(bars, counts):
        ratio = 0.0 if denominator <= 0 else (count / float(denominator))
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({ratio * 100:.1f}%)",
            ha="left",
            va="center",
            fontsize=8,
        )


def plot_bar_chart(
    counts: pd.Series,
    denominator: int,
    title: str,
    xlabel: str,
    out_file: Path,
    horizontal: bool = False,
):
    if not HAS_MATPLOTLIB:
        return

    labels = [format_value_text(x) for x in counts.index.tolist()]
    values = counts.to_numpy(dtype=float)

    if horizontal:
        fig_h = max(4.5, 0.45 * len(labels))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_ylabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        annotate_horizontal_bars(ax, bars, values, denominator)
    else:
        fig_w = max(8, 0.7 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 12 else 45, ha="right")
        ax.set_ylabel("Count")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        annotate_vertical_bars(ax, bars, values, denominator)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    out_file: Path,
    max_bins: int,
    annotate_bins: bool = False,
):
    if not HAS_MATPLOTLIB:
        return

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return

    bins = min(max_bins, max(10, int(np.sqrt(len(numeric)))))
    plt.figure(figsize=(9, 5))
    counts, _, patches = plt.hist(numeric, bins=bins, edgecolor="black", color="#4C78A8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    if annotate_bins:
        for count, patch in zip(counts, patches):
            if count <= 0:
                continue
            x_center = patch.get_x() + patch.get_width() / 2
            plt.text(
                x_center,
                count,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_outcome_day_trend(stats_df: pd.DataFrame, title: str, ylabel: str, out_file: Path):
    if not HAS_MATPLOTLIB:
        return
    if stats_df.empty:
        return

    x = np.arange(len(stats_df))
    y = stats_df["mean"].to_numpy(dtype=float)
    labels = stats_df["day_en"].tolist()
    valid_n = stats_df["non_missing_count"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, y, marker="o", color="#4C78A8", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    finite_y = y[np.isfinite(y)]
    if finite_y.size > 0:
        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))
        pad = max(0.3, (y_max - y_min) * 0.2 if y_max > y_min else 0.6)
        ax.set_ylim(y_min - pad, y_max + pad)

    for i, (v, n) in enumerate(zip(y, valid_n)):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i, 0, "NA", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close(fig)


def collect_onehot_counts(
    df: pd.DataFrame,
    prefix: str,
    label_map: dict[str, str] | None = None,
    ordered_suffixes: list[str] | None = None,
) -> pd.Series:
    label_map = label_map or {}
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(dtype="int64")

    suffix_to_col = {c[len(prefix):]: c for c in cols}

    if ordered_suffixes:
        suffixes = [s for s in ordered_suffixes if s in suffix_to_col]
        suffixes.extend([s for s in suffix_to_col if s not in suffixes])
    else:
        suffixes = sorted(suffix_to_col.keys())

    counts = {}
    for suffix in suffixes:
        col = suffix_to_col[suffix]
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        label = label_map.get(suffix, suffix)
        counts[label] = int((values > 0).sum())

    return pd.Series(counts, dtype="int64")


def maybe_numeric_sorted_counts(series: pd.Series) -> pd.Series:
    numeric_index = pd.to_numeric(series.index, errors="coerce")
    if numeric_index.notna().all():
        order = np.argsort(numeric_index.to_numpy())
        sorted_idx = [series.index[i] for i in order]
        return series.loc[sorted_idx]
    return series.sort_values(ascending=False)


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = (base_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    freq_dir = output_dir / "frequency_tables"
    plot_dir = output_dir / "plots"
    hist_table_dir = output_dir / "histogram_tables"

    output_dir.mkdir(parents=True, exist_ok=True)
    freq_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    hist_table_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_with_fallback(input_path)
    n_rows = len(df)

    summary_rows = []

    # 1) Grouped one-hot categories
    surgery_type_map = {
        "妇科": "Gynecology",
        "产科": "Obstetrics",
        "骨科-上肢": "Orthopedics - Upper Limb",
        "骨科-下肢": "Orthopedics - Lower Limb",
        "骨科-脊柱": "Orthopedics - Spine",
        "骨科-其他": "Orthopedics - Other",
        "腹外科-上腹": "Abdominal Surgery - Upper Abdomen",
        "腹外科-下腹": "Abdominal Surgery - Lower Abdomen",
        "腹外科-其他": "Abdominal Surgery - Other",
        "其他": "Other",
    }
    anesthesia_map = {
        "插管全麻": "General Anesthesia (Intubation)",
        "腰麻": "Spinal Anesthesia",
        "腰硬联合": "Combined Spinal-Epidural",
        "神经阻滞": "Nerve Block",
        "硬膜外": "Epidural",
        "其他": "Other",
    }
    analgesia_mode_map = {
        "PCIA": "PCIA",
        "PCEA": "PCEA",
        "PCNA": "PCNA",
        "PICA": "PICA",
        "其他": "Other",
    }
    block_site_map = {
        "下肢外周神经阻滞": "Peripheral Nerve Block - Lower Limb",
        "上肢/颈丛神经阻滞": "Peripheral Nerve Block - Upper Limb/Cervical Plexus",
        "腹壁平面阻滞": "Abdominal Wall Plane Block",
        "胸壁平面阻滞": "Chest Wall Plane Block",
        "椎管内镇痛": "Neuraxial Analgesia",
        "椎旁阻滞": "Paravertebral Block",
        "局部浸润": "Local Infiltration",
        "其他外周神经阻滞": "Other Peripheral Nerve Block",
        "未记录/未实施": "Not Documented / Not Performed",
        "其他": "Other",
    }

    group_specs = [
        {
            "key": "surgery_type",
            "prefix": "手术类型_",
            "label": "Surgery Type",
            "title": "Surgery Type Distribution",
            "label_map": surgery_type_map,
            "order": [
                "妇科",
                "产科",
                "骨科-上肢",
                "骨科-下肢",
                "骨科-脊柱",
                "骨科-其他",
                "腹外科-上腹",
                "腹外科-下腹",
                "腹外科-其他",
                "其他",
            ],
        },
        {
            "key": "anesthesia_method",
            "prefix": "麻醉方法_oh_",
            "label": "Anesthesia Method",
            "title": "Anesthesia Method Distribution",
            "label_map": anesthesia_map,
            "order": ["插管全麻", "腰麻", "腰硬联合", "神经阻滞", "硬膜外", "其他"],
        },
        {
            "key": "analgesia_mode",
            "prefix": "镇痛方式_oh_",
            "label": "Analgesia Mode",
            "title": "Analgesia Mode Distribution",
            "label_map": analgesia_mode_map,
            "order": ["PCIA", "PCEA", "PCNA", "PICA", "其他"],
        },
        {
            "key": "block_site",
            "prefix": "阻滞镇痛部位_",
            "label": "Block Site",
            "title": "Block Site Distribution",
            "label_map": block_site_map,
            "order": [
                "下肢外周神经阻滞",
                "上肢/颈丛神经阻滞",
                "腹壁平面阻滞",
                "胸壁平面阻滞",
                "椎管内镇痛",
                "椎旁阻滞",
                "局部浸润",
                "其他外周神经阻滞",
                "其他",
                "未记录/未实施",
            ],
        },
        {
            "key": "block_drug_presence",
            "prefix": "阻滞镇痛用药_has_",
            "label": "Block Drug",
            "title": "Block Drug Presence Distribution",
            "label_map": DRUG_MAP,
            "order": ["罗哌卡因", "布比卡因脂质体", "布比卡因", "利多卡因", "左旋布比卡因"],
        },
    ]

    for spec in group_specs:
        counts = collect_onehot_counts(
            df,
            prefix=spec["prefix"],
            label_map=spec.get("label_map"),
            ordered_suffixes=spec.get("order"),
        )
        if counts.empty:
            continue

        table_file = freq_dir / f"group_{spec['key']}.csv"
        plot_file = plot_dir / f"group_{spec['key']}.png"
        save_count_table(counts, n_rows, table_file, category_name=spec["label"])
        plot_bar_chart(
            counts,
            denominator=n_rows,
            title=spec["title"],
            xlabel=spec["label"],
            out_file=plot_file,
            horizontal=True,
        )

        summary_rows.append(
            {
                "name": spec["key"],
                "analysis_type": "grouped_one_hot",
                "frequency_table_file": str(table_file.relative_to(output_dir)),
                "plot_file": str(plot_file.relative_to(output_dir)) if HAS_MATPLOTLIB else "",
                "histogram_table_file": "",
            }
        )

    # 2) Month distribution: ordered 1-12, no surgery weekday plot
    if "手术月份" in df.columns:
        month = pd.to_numeric(df["手术月份"], errors="coerce")
        month = month[(month >= 1) & (month <= 12)].astype("Int64")
        month_counts = month.value_counts().reindex(range(1, 13), fill_value=0)

        table_file = freq_dir / "month_distribution.csv"
        plot_file = plot_dir / "month_distribution.png"
        save_count_table(month_counts, n_rows, table_file, category_name="Month")
        plot_bar_chart(
            month_counts,
            denominator=n_rows,
            title="Surgery Month Distribution (1-12)",
            xlabel="Month",
            out_file=plot_file,
            horizontal=False,
        )

        summary_rows.append(
            {
                "name": "month_distribution",
                "analysis_type": "ordered_month_bar",
                "frequency_table_file": str(table_file.relative_to(output_dir)),
                "plot_file": str(plot_file.relative_to(output_dir)) if HAS_MATPLOTLIB else "",
                "histogram_table_file": "",
            }
        )

    # 3) Pump formula dose detail table (different dose levels per drug)
    pump_dose_cols = [c for c in df.columns if c.startswith("镇痛泵配方_") and c.endswith("_mg")]
    pump_rows = []
    for col in pump_dose_cols:
        drug_cn = col.replace("镇痛泵配方_", "", 1).replace("_mg", "")
        drug_en = DRUG_MAP.get(drug_cn, drug_cn)
        doses = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        nonzero = doses[doses > 0]
        if nonzero.empty:
            continue
        dose_counts = nonzero.value_counts().sort_index()
        drug_total = int(len(nonzero))
        for dose, count in dose_counts.items():
            pump_rows.append(
                {
                    "drug_cn": drug_cn,
                    "drug_en": drug_en,
                    "dose_mg": float(dose),
                    "count": int(count),
                    "ratio_all_rows": float(count / n_rows),
                    "ratio_within_drug_nonzero": float(count / drug_total),
                }
            )

    if pump_rows:
        pump_table = pd.DataFrame(pump_rows).sort_values(["drug_en", "dose_mg"]).reset_index(drop=True)
        pump_file = freq_dir / "pump_formula_drug_dose_detail.csv"
        pump_table.to_csv(pump_file, index=False, encoding="utf-8-sig")

        # 宽表：每个药物一行，不同剂量作为不同列（值为样本数）
        pump_wide = (
            pump_table.pivot_table(
                index=["drug_cn", "drug_en"],
                columns="dose_mg",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index(axis=1)
            .reset_index()
        )
        rename_cols = {}
        for c in pump_wide.columns:
            if isinstance(c, (int, float, np.integer, np.floating)):
                dose_text = f"{float(c):.4f}".rstrip("0").rstrip(".")
                rename_cols[c] = f"dose_{dose_text}mg_count"
        pump_wide = pump_wide.rename(columns=rename_cols)
        pump_wide_file = freq_dir / "pump_formula_drug_dose_wide.csv"
        pump_wide.to_csv(pump_wide_file, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "name": "pump_formula_drug_dose_detail",
                "analysis_type": "custom_table",
                "frequency_table_file": str(pump_file.relative_to(output_dir)),
                "plot_file": "",
                "histogram_table_file": "",
            }
        )
        summary_rows.append(
            {
                "name": "pump_formula_drug_dose_wide",
                "analysis_type": "custom_table",
                "frequency_table_file": str(pump_wide_file.relative_to(output_dir)),
                "plot_file": "",
                "histogram_table_file": "",
            }
        )

    # 4) Outcome trends by day (one chart + one table per metric)
    outcome_all_cols = []
    day_order = [
        ("手术当天", "Surgery Day"),
        ("术后第一天", "POD1"),
        ("术后第二天", "POD2"),
        ("术后第三天", "POD3"),
    ]
    for metric_cn, metric_en in OUTCOME_METRICS.items():
        day_rows = []
        for day_cn, day_en in day_order:
            col = f"{day_cn}_{metric_cn}"
            if col not in df.columns:
                continue
            outcome_all_cols.append(col)
            values = pd.to_numeric(df[col], errors="coerce")
            non_missing_count = int(values.notna().sum())
            missing_count = int(values.isna().sum())
            day_rows.append(
                {
                    "day_cn": day_cn,
                    "day_en": day_en,
                    "column_name": col,
                    "non_missing_count": non_missing_count,
                    "missing_count": missing_count,
                    "missing_ratio": float(missing_count / n_rows),
                    "mean": float(values.mean()) if non_missing_count > 0 else np.nan,
                    "median": float(values.median()) if non_missing_count > 0 else np.nan,
                }
            )

        if not day_rows:
            continue

        stats_df = pd.DataFrame(day_rows)
        key = safe_name(f"outcome_trend_{metric_en.lower().replace('/', '_').replace(' ', '_')}")
        table_file = freq_dir / f"{key}.csv"
        plot_file = plot_dir / f"{key}.png"
        stats_df.to_csv(table_file, index=False, encoding="utf-8-sig")
        plot_outcome_day_trend(
            stats_df,
            title=f"{metric_en} Trend by Day",
            ylabel=f"Mean {metric_en}",
            out_file=plot_file,
        )

        summary_rows.append(
            {
                "name": key,
                "analysis_type": "outcome_day_trend",
                "frequency_table_file": str(table_file.relative_to(output_dir)),
                "plot_file": str(plot_file.relative_to(output_dir)) if HAS_MATPLOTLIB else "",
                "histogram_table_file": "",
            }
        )

    # 5) Regular columns (exclude grouped one-hot columns, outcome day columns, surgery weekday)
    excluded_prefixes = (
        "手术类型_",
        "麻醉方法_oh_",
        "镇痛方式_oh_",
        "阻滞镇痛部位_",
        "阻滞镇痛用药_has_",
        "镇痛泵配方_has_",  # explicitly excluded per requirement
    )
    excluded_columns = {"手术星期", "手术月份", *outcome_all_cols}

    for col in df.columns:
        if col in excluded_columns:
            continue
        if any(col.startswith(prefix) for prefix in excluded_prefixes):
            continue

        series = df[col]
        col_en = to_english_col_name(col)
        stem = f"column_{safe_name(col)}"

        freq_file = freq_dir / f"{stem}.csv"
        plot_file = plot_dir / f"{stem}.png"
        hist_table_file = hist_table_dir / f"{stem}.csv"

        cat = series.astype("string").replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        counts = cat.value_counts(dropna=False)

        if len(counts) > args.max_categories:
            top = counts.head(args.max_categories)
            remainder = int(counts.iloc[args.max_categories:].sum())
            if remainder > 0:
                top.loc["<OTHER>"] = remainder
            counts = top

        counts = maybe_numeric_sorted_counts(counts)
        save_count_table(counts, n_rows, freq_file, category_name="Value")

        numeric = pd.to_numeric(series, errors="coerce")
        valid_numeric = numeric.dropna()
        likely_numeric = pd.api.types.is_numeric_dtype(series) or (valid_numeric.size / max(1, len(series)) >= 0.95)

        if likely_numeric and valid_numeric.nunique() > args.discrete_unique_threshold:
            write_histogram_table_numeric(series, hist_table_file, args.max_bins)
            plot_histogram(
                series,
                title=f"{col_en} Histogram",
                xlabel=col_en,
                out_file=plot_file,
                max_bins=args.max_bins,
                annotate_bins=(col in {"年龄", "体重"}),
            )
            analysis_type = "regular_histogram"
        else:
            pd.DataFrame(columns=["bin_left", "bin_right", "count", "ratio"]).to_csv(
                hist_table_file, index=False, encoding="utf-8-sig"
            )
            plot_bar_chart(
                counts,
                denominator=n_rows,
                title=f"{col_en} Distribution",
                xlabel=col_en,
                out_file=plot_file,
                horizontal=(len(counts) > 10),
            )
            analysis_type = "regular_bar"

        summary_rows.append(
            {
                "name": col,
                "analysis_type": analysis_type,
                "frequency_table_file": str(freq_file.relative_to(output_dir)),
                "plot_file": str(plot_file.relative_to(output_dir)) if HAS_MATPLOTLIB else "",
                "histogram_table_file": str(hist_table_file.relative_to(output_dir)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "column_summary.csv", index=False, encoding="utf-8-sig")

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Distribution Analysis Output",
                "",
                f"- Input: `{input_path}`",
                f"- Total rows: {n_rows}",
                "- All chart titles/axes are in English.",
                "- All bar charts are annotated with count and percentage.",
                "- Age/Weight histograms are annotated with bin counts.",
                "- Grouped charts include: surgery type, anesthesia method, analgesia mode, block site, block drug presence.",
                "- Pump formula dose detail table: `frequency_tables/pump_formula_drug_dose_detail.csv`.",
                "- Pump formula wide table (dose as columns): `frequency_tables/pump_formula_drug_dose_wide.csv`.",
                "- Excluded from plotting: `镇痛泵配方_has_*`, `手术星期`.",
                "- Surgery month is plotted in fixed order 1..12.",
                "- Outcomes are split by day (Surgery Day/POD1/POD2/POD3) with trend charts per metric.",
                f"- Frequency tables: `{freq_dir}`",
                f"- Plots: `{plot_dir}`",
                f"- Histogram tables: `{hist_table_dir}`",
                f"- Matplotlib enabled: {HAS_MATPLOTLIB}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Analysis completed. Output directory: {output_dir}")
    print(f"Rows analyzed: {n_rows}")
    print(f"Artifacts: {len(summary_rows)}")
    print(f"Matplotlib enabled: {HAS_MATPLOTLIB}")


if __name__ == "__main__":
    main()
