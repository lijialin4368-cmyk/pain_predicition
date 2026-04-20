"""可视化随机森林回归输出结果。

运行方式：
    cd pain_prediction
    pixi run rf-plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


TOKEN_TRANSLATIONS = {
    "手术当天": "surgery_day",
    "术后第一天": "pod1",
    "术后第二天": "pod2",
    "术后第三天": "pod3",
    "静息痛": "rest_pain",
    "活动痛": "movement_pain",
    "镇静评分": "sedation_score",
    "活动状态": "activity_status",
    "恶心呕吐": "nausea_vomiting",
    "手术类型": "surgery_type",
    "麻醉方法": "anesthesia_method",
    "镇痛方式": "analgesia_mode",
    "阻滞镇痛部位": "block_site",
    "阻滞镇痛用药": "block_analgesic_drug",
    "镇痛泵配方": "pump_formula",
    "背景量": "basal_dose",
    "总量": "total_amount",
    "总体积": "total_volume",
    "年份": "year",
    "年龄": "age",
    "体重": "weight",
    "手术月份": "surgery_month",
    "手术星期": "surgery_weekday",
    "性别": "sex",
    "分级": "grade",
    "妇科": "gynecology",
    "产科": "obstetrics",
    "骨科": "orthopedics",
    "上肢": "upper_limb",
    "下肢": "lower_limb",
    "脊柱": "spine",
    "腹外科": "abdominal_surgery",
    "上腹": "upper_abdomen",
    "下腹": "lower_abdomen",
    "其他": "other",
    "插管全麻": "general_anesthesia_intubation",
    "腰麻": "spinal_anesthesia",
    "腰硬联合": "combined_spinal_epidural",
    "神经阻滞": "nerve_block",
    "硬膜外": "epidural",
    "罗哌卡因": "ropivacaine",
    "布比卡因脂质体": "liposomal_bupivacaine",
    "布比卡因": "bupivacaine",
    "舒芬": "sufentanil",
    "羟考酮": "oxycodone",
    "曲马多": "tramadol",
    "帕洛诺司琼": "palonosetron",
    "布托啡诺": "butorphanol",
    "纳布啡": "nalbuphine",
    "地佐辛": "dezocine",
    "下肢外周神经阻滞": "lower_limb_peripheral_nerve_block",
    "上肢/颈丛神经阻滞": "upper_limb_or_cervical_plexus_block",
    "腹壁平面阻滞": "abdominal_wall_plane_block",
    "胸壁平面阻滞": "chest_wall_plane_block",
    "椎旁阻滞": "paravertebral_block",
    "其他外周神经阻滞": "other_peripheral_nerve_block",
    "未记录/未实施": "not_recorded_or_not_performed",
}


def translate_feature_name(name: str) -> str:
    """将中文/混合列名转成英文蛇形命名，便于图表展示。"""
    out = name
    # 先替换长词，避免被短词抢先替换。
    for zh, en in sorted(TOKEN_TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        out = out.replace(zh, en)

    # 统一常见符号与字段前缀。
    out = out.replace("has_", "contains_")
    out = out.replace("_oh_", "_")
    out = out.replace("-", "_")
    out = out.replace("/", "_or_")
    out = out.replace("ml", "ml")

    # 清理多余字符，确保标签稳定可读。
    out = re.sub(r"[^A-Za-z0-9_]", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        return "feature"
    return out.lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random forest regression outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing metrics.json, test_predictions.csv, and feature_importance.csv.",
    )
    return parser.parse_args()


def load_files(output_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """读取可视化所需文件。"""
    metrics_path = output_dir / "metrics.json"
    test_pred_path = output_dir / "test_predictions.csv"
    feature_importance_path = output_dir / "feature_importance.csv"
    for p in [metrics_path, test_pred_path, feature_importance_path]:
        if not p.exists():
            raise FileNotFoundError(f"缺少文件: {p}，请先执行训练。")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    pred_df = pd.read_csv(test_pred_path, encoding="utf-8-sig")
    fi_df = pd.read_csv(feature_importance_path, encoding="utf-8-sig")
    return metrics, pred_df, fi_df


def plot_true_vs_pred(pred_df: pd.DataFrame, metrics: dict, plot_dir: Path) -> Path:
    """图1：真实值 vs 预测值散点图。"""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(pred_df["y_true"], pred_df["y_pred"], s=20, alpha=0.65)

    # 参考对角线：理想情况下点应尽量靠近这条线。
    min_v = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    max_v = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5)

    ax.set_title("True vs Predicted")
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.text(
        0.03,
        0.97,
        f"R2={metrics.get('r2', float('nan')):.3f}\nRMSE={metrics.get('rmse', float('nan')):.3f}\nMAE={metrics.get('mae', float('nan')):.3f}",
        transform=ax.transAxes,
        va="top",
    )

    out_path = plot_dir / "true_vs_pred.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_error_distribution(pred_df: pd.DataFrame, plot_dir: Path) -> Path:
    """图2：误差分布（y_pred - y_true）。"""
    err = pred_df["y_pred"] - pred_df["y_true"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(err, bins=35, alpha=0.8)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (y_pred - y_true)")
    ax.set_ylabel("Count")

    out_path = plot_dir / "error_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_abs_error_by_true(pred_df: pd.DataFrame, plot_dir: Path) -> Path:
    """图3：不同真实值水平上的平均绝对误差。"""
    group = (
        pred_df.groupby("y_true", as_index=False)["abs_error"]
        .mean()
        .sort_values("y_true")
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(group["y_true"], group["abs_error"], marker="o", linewidth=1.8)
    ax.set_title("Mean Absolute Error by True Value")
    ax.set_xlabel("True value")
    ax.set_ylabel("Mean absolute error")

    out_path = plot_dir / "mae_by_true_value.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_feature_importance(fi_df: pd.DataFrame, plot_dir: Path, top_n: int = 20) -> tuple[Path, Path]:
    """图4：Top-N 特征重要性条形图。"""
    top_df = fi_df.sort_values("importance", ascending=False).head(top_n)
    top_df = top_df.reset_index(drop=True)
    top_df["feature_en"] = top_df["feature"].astype(str).map(translate_feature_name)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(top_df["feature_en"][::-1], top_df["importance"][::-1])
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    out_path = plot_dir / "feature_importance_top20.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    mapping_path = plot_dir / "feature_importance_top20_mapping.csv"
    top_df[["feature_en", "feature", "importance"]].to_csv(
        mapping_path, index=False, encoding="utf-8-sig"
    )
    return out_path, mapping_path


def main() -> None:
    args = parse_args()
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics, pred_df, fi_df = load_files(args.output_dir)

    fi_plot_path, fi_map_path = plot_feature_importance(fi_df, plot_dir, top_n=20)
    paths = [
        plot_true_vs_pred(pred_df, metrics, plot_dir),
        plot_error_distribution(pred_df, plot_dir),
        plot_abs_error_by_true(pred_df, plot_dir),
        fi_plot_path,
        fi_map_path,
    ]

    print("可视化完成，生成文件：")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
