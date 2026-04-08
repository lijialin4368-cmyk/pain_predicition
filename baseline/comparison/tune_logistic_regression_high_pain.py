import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TOTAL_OUTPUT_FILES = [
    "prediction_overview_all_targets.csv",
    "confusion_matrix_prob_all_targets.csv",
    "confusion_matrix_prob_all_targets.png",
    "training_acc_all_targets.png",
]

MEAN_METRIC_COLUMNS = [
    "auc",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "fpr",
    "f1",
    "log_loss",
    "brier",
]


def decimal_range(start: float, stop: float, step: float):
    cur = Decimal(str(start))
    end = Decimal(str(stop))
    stride = Decimal(str(step))
    values = []
    while cur <= end + Decimal("1e-9"):
        values.append(float(cur))
        cur += stride
    return values


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search high pain enhancement parameters for train_logistic_regression.py, "
            "save 2 figures + 2 tables per run, and generate tuning records."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "tuning_records" / "logreg_high_pain_grid",
        help="Root directory to save all run folders and the tuning report.",
    )
    parser.add_argument("--oversample-start", type=float, default=2.5)
    parser.add_argument("--oversample-stop", type=float, default=5.0)
    parser.add_argument("--oversample-step", type=float, default=0.5)
    parser.add_argument("--loss-start", type=float, default=2.0)
    parser.add_argument("--loss-stop", type=float, default=10.0)
    parser.add_argument("--loss-step", type=float, default=0.5)
    parser.add_argument(
        "--objective",
        type=str,
        choices=["pain_f1", "mean_f1"],
        default="pain_f1",
        help="Criterion for selecting the best model.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to train_logistic_regression.py (must be placed after --extra-args).",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=None,
        help=(
            "Optional path to train_logistic_regression.py. "
            "If omitted, the script tries common project locations automatically."
        ),
    )
    return parser.parse_args()


def resolve_train_script(script_dir: Path, user_path: Optional[Path]) -> Path:
    if user_path is not None:
        train_script = user_path.expanduser().resolve()
        if not train_script.exists():
            raise FileNotFoundError(f"train script not found: {train_script}")
        return train_script

    candidates = [
        script_dir / "train_logistic_regression.py",
        script_dir.parent / "final_baseline_logistic_re" / "train_logistic_regression.py",
        script_dir.parent / "train_logistic_regression.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Cannot find train_logistic_regression.py. Tried:\n- "
        + "\n- ".join(str(p) for p in candidates)
    )


def summarize_prediction_overview(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    summary = {
        "n_targets": int(len(df)),
    }

    for col in MEAN_METRIC_COLUMNS:
        summary[f"mean_{col}"] = float(df[col].mean()) if col in df.columns else float("nan")

    pain_mask = df["target"].str.contains("RestPain|MovementPain", case=False, regex=True)
    pain_df = df[pain_mask].copy()
    summary["n_pain_targets"] = int(len(pain_df))
    for col in MEAN_METRIC_COLUMNS:
        key = f"pain_mean_{col}"
        summary[key] = float(pain_df[col].mean()) if len(pain_df) > 0 and col in pain_df.columns else float("nan")

    return summary


def build_train_command(train_script: Path, oversample_factor: float, loss_weight: float, output_dir: Path, extra_args):
    cmd = [
        sys.executable,
        str(train_script),
        "--high-pain-oversample-factor",
        f"{oversample_factor:.1f}",
        "--high-pain-loss-weight",
        f"{loss_weight:.1f}",
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def rank_key(row: dict, objective: str):
    if objective == "pain_f1":
        primary = row.get("pain_mean_f1", float("nan"))
        if np.isnan(primary):
            primary = row.get("mean_f1", float("nan"))
    else:
        primary = row.get("mean_f1", float("nan"))

    secondary = row.get("pain_mean_auc", float("nan"))
    tertiary = row.get("mean_auc", float("nan"))
    quaternary = row.get("mean_accuracy", float("nan"))

    vals = [primary, secondary, tertiary, quaternary]
    safe_vals = [(-1.0 if np.isnan(v) else float(v)) for v in vals]
    return tuple(safe_vals)


def write_markdown_report(md_path: Path, all_rows: pd.DataFrame, ok_rows: pd.DataFrame, best_row: dict, args, elapsed_sec: float):
    utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_total = int(len(all_rows))
    n_ok = int(len(ok_rows))
    n_fail = n_total - n_ok

    oversample_values = decimal_range(args.oversample_start, args.oversample_stop, args.oversample_step)
    loss_values = decimal_range(args.loss_start, args.loss_stop, args.loss_step)

    lines = []
    lines.append("# Logistic Regression 高痛样本增强参数调参记录")
    lines.append("")
    lines.append(f"- 生成时间: `{utc_time}`")
    lines.append(f"- 运行总耗时: `{elapsed_sec:.1f}s`")
    lines.append(f"- 搜索范围: `--high-pain-oversample-factor` = {oversample_values}")
    lines.append(f"- 搜索范围: `--high-pain-loss-weight` = {loss_values}")
    lines.append(f"- 总组合数: `{n_total}`，成功: `{n_ok}`，失败: `{n_fail}`")
    lines.append(f"- 最优模型选择准则: `{args.objective}`")
    lines.append("")

    if best_row:
        lines.append("## 最优参数")
        lines.append("")
        lines.append(f"- `--high-pain-oversample-factor = {best_row['high_pain_oversample_factor']:.1f}`")
        lines.append(f"- `--high-pain-loss-weight = {best_row['high_pain_loss_weight']:.1f}`")
        lines.append(f"- `pain_mean_f1 = {best_row.get('pain_mean_f1', float('nan')):.4f}`")
        lines.append(f"- `pain_mean_auc = {best_row.get('pain_mean_auc', float('nan')):.4f}`")
        lines.append(f"- `mean_f1 = {best_row.get('mean_f1', float('nan')):.4f}`")
        lines.append(f"- `mean_auc = {best_row.get('mean_auc', float('nan')):.4f}`")
        lines.append(f"- `mean_accuracy = {best_row.get('mean_accuracy', float('nan')):.4f}`")
        lines.append(f"- 输出目录（含2总图2总表）: `{best_row['output_dir']}`")
        lines.append("")

    lines.append("## 调参明细（每行对应一次参数变更）")
    lines.append("")
    lines.append(
        "| run_id | oversample | loss_weight | pain_mean_f1 | pain_mean_auc | mean_f1 | mean_auc | mean_acc | status | output_dir |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")

    for _, r in all_rows.iterrows():
        lines.append(
            "| {run_id} | {osf:.1f} | {lw:.1f} | {pmf1:.4f} | {pmauc:.4f} | {mf1:.4f} | {mauc:.4f} | {macc:.4f} | {status} | `{od}` |".format(
                run_id=int(r["run_id"]),
                osf=float(r["high_pain_oversample_factor"]),
                lw=float(r["high_pain_loss_weight"]),
                pmf1=float(r.get("pain_mean_f1", np.nan)),
                pmauc=float(r.get("pain_mean_auc", np.nan)),
                mf1=float(r.get("mean_f1", np.nan)),
                mauc=float(r.get("mean_auc", np.nan)),
                macc=float(r.get("mean_accuracy", np.nan)),
                status=str(r["status"]),
                od=str(r["output_dir"]),
            )
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    train_script = resolve_train_script(script_dir=script_dir, user_path=args.train_script)

    output_root = args.output_root.resolve()
    runs_root = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    oversample_values = decimal_range(args.oversample_start, args.oversample_stop, args.oversample_step)
    loss_values = decimal_range(args.loss_start, args.loss_stop, args.loss_step)
    total_runs = len(oversample_values) * len(loss_values)

    all_rows = []
    best_row = None
    start_all = time.time()
    run_id = 0

    print(f"[GridSearch] output_root={output_root}")
    print(f"[GridSearch] total_runs={total_runs}")

    for oversample in oversample_values:
        for loss_weight in loss_values:
            run_id += 1
            tag = f"os_{oversample:.1f}_lw_{loss_weight:.1f}"
            run_dir = runs_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = run_dir / "train.log"

            cmd = build_train_command(
                train_script=train_script,
                oversample_factor=oversample,
                loss_weight=loss_weight,
                output_dir=run_dir,
                extra_args=args.extra_args,
            )

            print(f"[{run_id:03d}/{total_runs:03d}] START {tag}")
            t0 = time.time()
            proc = subprocess.run(
                cmd,
                cwd=script_dir,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0
            log_file.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")

            row = {
                "run_id": run_id,
                "status": "ok" if proc.returncode == 0 else f"failed({proc.returncode})",
                "high_pain_oversample_factor": oversample,
                "high_pain_loss_weight": loss_weight,
                "elapsed_sec": elapsed,
                "output_dir": str(run_dir),
                "log_file": str(log_file),
            }

            if proc.returncode == 0:
                summary_csv = run_dir / "prediction_overview_all_targets.csv"
                if summary_csv.exists():
                    row.update(summarize_prediction_overview(summary_csv))
                else:
                    row["status"] = "failed(missing_summary_csv)"
            all_rows.append(row)

            if row["status"] == "ok":
                if best_row is None or rank_key(row, args.objective) > rank_key(best_row, args.objective):
                    best_row = row

            best_tag = "N/A"
            if best_row is not None:
                best_tag = (
                    f"os={best_row['high_pain_oversample_factor']:.1f}, "
                    f"lw={best_row['high_pain_loss_weight']:.1f}, "
                    f"pain_f1={best_row.get('pain_mean_f1', float('nan')):.4f}"
                )
            print(f"[{run_id:03d}/{total_runs:03d}] DONE {tag} elapsed={elapsed:.1f}s best={best_tag}")

    elapsed_all = time.time() - start_all
    all_df = pd.DataFrame(all_rows)
    ok_df = all_df[all_df["status"] == "ok"].copy()

    summary_csv = output_root / "grid_search_summary.csv"
    all_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    ranking_csv = output_root / "grid_search_ranking.csv"
    if len(ok_df) > 0:
        sort_cols = ["pain_mean_f1", "pain_mean_auc", "mean_f1", "mean_auc", "mean_accuracy"]
        for c in sort_cols:
            if c not in ok_df.columns:
                ok_df[c] = np.nan
        ranking_df = ok_df.sort_values(sort_cols, ascending=False)
    else:
        ranking_df = ok_df
    ranking_df.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    best_dir = output_root / "best_model_output_prediction_report_en"
    if best_row is not None:
        best_run_dir = Path(best_row["output_dir"])
        best_dir.mkdir(parents=True, exist_ok=True)
        for fname in TOTAL_OUTPUT_FILES:
            src = best_run_dir / fname
            if src.exists():
                shutil.copy2(src, best_dir / fname)

    md_report = output_root / "tuning_record.md"
    write_markdown_report(
        md_path=md_report,
        all_rows=all_df,
        ok_rows=ok_df,
        best_row=best_row,
        args=args,
        elapsed_sec=elapsed_all,
    )

    print("")
    print("[GridSearch] Finished.")
    print(f"[GridSearch] elapsed={elapsed_all:.1f}s")
    print(f"[GridSearch] summary_csv={summary_csv}")
    print(f"[GridSearch] ranking_csv={ranking_csv}")
    print(f"[GridSearch] report_md={md_report}")
    if best_row is None:
        print("[GridSearch] best_model=None (all runs failed)")
    else:
        print(
            "[GridSearch] best_model="
            f"os={best_row['high_pain_oversample_factor']:.1f}, "
            f"lw={best_row['high_pain_loss_weight']:.1f}, "
            f"pain_mean_f1={best_row.get('pain_mean_f1', float('nan')):.4f}"
        )
        print(f"[GridSearch] best_model_files={best_dir}")


if __name__ == "__main__":
    main()
