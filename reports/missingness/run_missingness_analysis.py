import re
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_INPUT_CSV = PROJECT_DIR / "data" / "raw" / "data.csv"
VECTOR_INPUT_CSV = PROJECT_DIR / "data" / "processed" / "data_vectorized.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


OUTCOME_COLS = [
    "手术当天_静息痛",
    "手术当天_活动痛",
    "手术当天_镇静评分",
    "手术当天_活动状态",
    "手术当天_恶心呕吐",
    "术后第一天_静息痛",
    "术后第一天_活动痛",
    "术后第一天_镇静评分",
    "术后第一天_活动状态",
    "术后第一天_恶心呕吐",
    "术后第二天_静息痛",
    "术后第二天_活动痛",
    "术后第二天_镇静评分",
    "术后第二天_活动状态",
    "术后第二天_恶心呕吐",
    "术后第三天_静息痛",
    "术后第三天_活动痛",
    "术后第三天_镇静评分",
    "术后第三天_活动状态",
    "术后第三天_恶心呕吐",
]

COVARIATE_COLS_RAW = [
    "年份",
    "手术日期",
    "手术月份",
    "手术星期",
    "性别",
    "年龄",
    "体重",
    "ASA分级",
    "手术名称",
    "麻醉方法",
    "镇痛方式",
    "镇痛泵配方",
    "阻滞镇痛部位",
    "阻滞镇痛用药",
]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


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


def translate_outcome(col: str) -> str:
    day_map = {
        "手术当天": "Surgery Day",
        "术后第一天": "POD1",
        "术后第二天": "POD2",
        "术后第三天": "POD3",
    }
    metric_map = {
        "静息痛": "Rest Pain",
        "活动痛": "Movement Pain",
        "镇静评分": "Sedation Score",
        "活动状态": "Activity Status",
        "恶心呕吐": "Nausea/Vomiting",
    }
    day, metric = col.split("_", 1)
    return f"{day_map.get(day, day)} - {metric_map.get(metric, metric)}"


def translate_col(col: str) -> str:
    exact = {
        "性别": "Sex",
        "ASA分级": "ASA Grade",
        "总量": "Pump Total Volume",
        "背景量": "Pump Basal Rate",
        "手术日期": "Surgery Date",
        "手术名称": "Surgery Name",
        "麻醉方法": "Anesthesia Method",
        "镇痛方式": "Analgesia Mode",
        "镇痛泵配方": "Pump Formula",
        "阻滞镇痛部位": "Block Site",
        "阻滞镇痛用药": "Block Drugs",
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
    if col.startswith("手术类型_"):
        key = col.replace("手术类型_", "", 1)
        return f"Surgery Type - {surgery_type_map.get(key, key)}"

    anesthesia_map = {
        "插管全麻": "General Anesthesia (Intubation)",
        "腰麻": "Spinal Anesthesia",
        "腰硬联合": "Combined Spinal-Epidural",
        "神经阻滞": "Nerve Block",
        "硬膜外": "Epidural",
        "其他": "Other",
    }
    if col.startswith("麻醉方法_oh_"):
        key = col.replace("麻醉方法_oh_", "", 1)
        return f"Anesthesia - {anesthesia_map.get(key, key)}"

    if col.startswith("镇痛方式_oh_"):
        key = col.replace("镇痛方式_oh_", "", 1)
        return f"Analgesia Mode - {key}"

    block_site_map = {
        "神经阻滞（外周神经）": "Peripheral Nerve Block",
        "椎旁阻滞": "Paravertebral Block",
        "其他": "Other",
    }
    if col.startswith("阻滞镇痛部位_"):
        key = col.replace("阻滞镇痛部位_", "", 1)
        return f"Block Site - {block_site_map.get(key, key)}"

    if col.startswith("镇痛泵配方_has_"):
        key = col.replace("镇痛泵配方_has_", "", 1)
        return f"Pump Contains - {DRUG_MAP.get(key, key)}"

    if col.startswith("镇痛泵配方_") and col.endswith("_mg"):
        key = col.replace("镇痛泵配方_", "", 1).replace("_mg", "")
        return f"Pump Dose - {DRUG_MAP.get(key, key)} (mg)"

    if col.startswith("阻滞镇痛用药_has_"):
        key = col.replace("阻滞镇痛用药_has_", "", 1)
        return f"Block Drug Contains - {DRUG_MAP.get(key, key)}"

    if col.startswith("阻滞镇痛用药_") and col.endswith("_dose"):
        key = col.replace("阻滞镇痛用药_", "", 1).replace("_dose", "")
        return f"Block Drug Dose - {DRUG_MAP.get(key, key)}"

    return col


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read csv file: {path}")


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    if len(object_cols) == 0:
        return out
    out[object_cols] = out[object_cols].apply(lambda s: s.astype("string").str.strip())
    out[object_cols] = out[object_cols].replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NONE": pd.NA, "None": pd.NA, "null": pd.NA}
    )
    return out


def parse_surgery_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip().replace("日", "")
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return pd.NaT

    if re.fullmatch(r"\d+(\.0+)?", s):
        n = int(float(s))
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(n, unit="D")

    md = re.match(r"^\s*(\d{1,2})\.(\d{1,2})\s*$", s)
    if md:
        m = int(md.group(1))
        d = int(md.group(2))
        return pd.to_datetime(f"2023-{m:02d}-{d:02d}", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def build_missingness_table(df: pd.DataFrame, cols: list[str], is_outcome: bool) -> pd.DataFrame:
    sub = df[cols].copy()
    missing_count = sub.isna().sum()
    total = len(sub)
    table = pd.DataFrame(
        {
            "column_cn": cols,
            "column_en": [translate_outcome(c) if is_outcome else translate_col(c) for c in cols],
            "missing_count": [int(missing_count[c]) for c in cols],
            "missing_ratio": [float(missing_count[c] / total) for c in cols],
        }
    )
    table = table.sort_values(["missing_ratio", "missing_count"], ascending=False).reset_index(drop=True)
    return table


def plot_missingness(table: pd.DataFrame, title: str, out_file: Path):
    ratios = table["missing_ratio"].to_numpy()
    labels = table["column_en"].tolist()

    height = max(6, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(12, height))
    y = np.arange(len(labels))
    bars = ax.barh(y, ratios, color="#4C78A8")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.05, float(ratios.max()) * 1.20))
    ax.set_xlabel("Missing Ratio")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio*100:.1f}%",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = normalize_missing_values(read_csv_with_fallback(RAW_INPUT_CSV))
    _ = normalize_missing_values(read_csv_with_fallback(VECTOR_INPUT_CSV))

    if "手术日期" in raw_df.columns:
        date_series = raw_df["手术日期"].apply(parse_surgery_date)
        raw_df["手术日期"] = date_series
        raw_df["手术月份"] = date_series.dt.month
        raw_df["手术星期"] = date_series.dt.dayofweek

    outcome_cols = [c for c in OUTCOME_COLS if c in raw_df.columns]
    covariate_cols = [c for c in COVARIATE_COLS_RAW if c in raw_df.columns]

    cov_table = build_missingness_table(raw_df, covariate_cols, is_outcome=False)
    out_table = build_missingness_table(raw_df, outcome_cols, is_outcome=True)

    cov_table.to_csv(OUTPUT_DIR / "covariates_missingness.csv", index=False, encoding="utf-8-sig")
    out_table.to_csv(OUTPUT_DIR / "outcomes_missingness.csv", index=False, encoding="utf-8-sig")

    plot_missingness(
        cov_table,
        title="Covariates Missingness Ratio (Before Imputation)",
        out_file=OUTPUT_DIR / "covariates_missingness.png",
    )
    plot_missingness(
        out_table,
        title="Outcomes Missingness Ratio",
        out_file=OUTPUT_DIR / "outcomes_missingness.png",
    )

    readme = OUTPUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Missingness Report",
                "",
                "缺失率分析输入：",
                "",
                "```text",
                display_path(RAW_INPUT_CSV),
                display_path(VECTOR_INPUT_CSV),
                "```",
                "",
                "运行：",
                "",
                "```bash",
                "pixi run missingness-report",
                "```",
                "",
                "直接运行：",
                "",
                "```bash",
                "pixi run python reports/missingness/run_missingness_analysis.py",
                "```",
                "",
                "输出：",
                "",
                "```text",
                "covariates_missingness.png",
                "outcomes_missingness.png",
                "covariates_missingness.csv",
                "outcomes_missingness.csv",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    print("Missingness analysis generated:")
    print(f"- {OUTPUT_DIR / 'covariates_missingness.png'}")
    print(f"- {OUTPUT_DIR / 'outcomes_missingness.png'}")
    print(f"- {OUTPUT_DIR / 'covariates_missingness.csv'}")
    print(f"- {OUTPUT_DIR / 'outcomes_missingness.csv'}")


if __name__ == "__main__":
    main()
