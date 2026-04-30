"""Build and refresh a unified registry of model outputs across the project.

This registry is designed to summarize both regression-style metrics.json files
and classification-style prediction_overview_all_targets.csv files.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_FILENAME = "registry/model_output_registry.csv"
SUMMARY_FILENAME = "registry/model_results_summary.csv"
SKIP_PATH_PARTS = {".git", ".pixi", "__pycache__", "tuning_records"}

REGISTRY_COLUMNS = [
    "record_origin",
    "task_type",
    "project_section",
    "run_name",
    "model_family",
    "model_name",
    "group_name",
    "strategy_name",
    "target",
    "target_column",
    "class_0_label",
    "class_1_label",
    "source_file",
    "source_mtime_utc",
    "output_dir",
    "source_data_path",
    "source_data_path_inferred",
    "n_samples",
    "n_features",
    "train_size",
    "val_size",
    "test_size",
    "train_original_size",
    "train_generated_size",
    "test_original_size",
    "test_generated_size",
    "auc",
    "auprc",
    "positive_rate",
    "auprc_lift",
    "acc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "log_loss",
    "brier",
    "r2",
    "rmse",
    "mae",
    "tn",
    "fp",
    "fn",
    "tp",
    "chosen_threshold",
    "train_loss",
    "hyperparam_search_enabled",
    "hyperparam_search_scoring",
    "hyperparam_search_best_score",
    "hyperparam_search_best_params",
    "high_pain_recall_at_4",
    "high_pain_precision_at_4",
    "high_pain_f1_at_4",
    "high_pain_subset_mae_at_4",
    "high_pain_recall_at_5",
    "high_pain_precision_at_5",
    "high_pain_f1_at_5",
    "high_pain_subset_mae_at_5",
]

SUMMARY_COLUMNS = [
    "model",
    "task",
    "target",
    "data_version",
    "mae",
    "rmse",
    "r2",
    "auroc",
    "auprc",
    "recall",
    "precision",
    "f1",
    "source_file",
    "updated_at",
]


def _find_project_dir(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if (candidate / "pixi.toml").exists():
            return candidate
    return Path(__file__).resolve().parent


def _blank_record() -> dict:
    return {col: "" for col in REGISTRY_COLUMNS}


def _normalize_scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return ""
    return value


def _record_path(path: Path, project_dir: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _should_skip_path(path: Path) -> bool:
    return any(part in SKIP_PATH_PARTS for part in path.parts)


def _read_target_from_config(config_path: Path) -> str:
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^TARGET_COLUMN\s*=\s*[\"'](.+?)[\"']\s*$", text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _infer_model_family(rel_parts: tuple[str, ...]) -> str:
    parts = set(rel_parts)
    if len(rel_parts) >= 2 and rel_parts[0] == "models":
        if rel_parts[1] == "random_forest":
            return "randomforest"
        if rel_parts[1] == "logistic":
            return "logistic_regression"
        return rel_parts[1]
    if "xgboost" in parts:
        return "xgboost"
    if "randomforest" in parts or "randomforest_standard_rawdata" in parts or "data_argmentation_A" in parts:
        return "randomforest"
    if "baseline_enhance_logistic" in parts or "baseline" in parts:
        return "logistic_regression"
    return rel_parts[0] if rel_parts else ""


def _infer_run_name(rel_parts: tuple[str, ...], group_name: str = "") -> str:
    if "randomforest_standard_rawdata" in rel_parts:
        return "random_forest_raw"
    if group_name:
        return group_name
    if not rel_parts:
        return ""
    if len(rel_parts) >= 2 and rel_parts[0] == "models":
        if "outputs" in rel_parts:
            out_idx = rel_parts.index("outputs")
            if out_idx + 1 < len(rel_parts):
                if rel_parts[out_idx + 1].endswith((".csv", ".json")):
                    return f"{rel_parts[1]}_raw"
                return f"{rel_parts[1]}_{rel_parts[out_idx + 1]}"
        return rel_parts[1]
    if rel_parts[0] == "randomforest":
        if "outputs_comparison" in rel_parts:
            return "outputs_comparison"
        if "outputs" in rel_parts:
            return "outputs"
    if rel_parts[0] == "xgboost":
        return "outputs"
    if rel_parts[0] == "baseline" and len(rel_parts) > 1:
        return rel_parts[1]
    if rel_parts[0] == "baseline_enhance_logistic" and len(rel_parts) > 1:
        return rel_parts[1]
    return rel_parts[0]


def _infer_source_data_path(project_dir: Path, rel_parts: tuple[str, ...], group_name: str = "") -> tuple[str, bool]:
    inferred = True
    if len(rel_parts) >= 4 and rel_parts[0] == "models" and rel_parts[2] == "outputs":
        if rel_parts[3] == "raw":
            return str((project_dir / "data" / "processed" / "data_vectorized.csv").resolve()), inferred
        if rel_parts[3] == "augmented":
            return str((project_dir / "experiments" / "augmentation" / "generated" / "augmented_dataset.csv").resolve()), inferred
        if rel_parts[3] == "copy_control":
            return str((project_dir / "experiments" / "augmentation_copy_control" / "generated" / "augmented_dataset.csv").resolve()), inferred
    if "randomforest_standard_rawdata" in rel_parts:
        return str((project_dir / "data" / "processed" / "data_vectorized.csv").resolve()), inferred
    if rel_parts[:2] == ("randomforest", "outputs_comparison"):
        return str((project_dir / "experiments" / "augmentation_copy_control" / "generated" / "augmented_dataset.csv").resolve()), inferred
    if rel_parts and rel_parts[0] == "randomforest":
        return str((project_dir / "experiments" / "augmentation" / "generated" / "augmented_dataset.csv").resolve()), inferred
    if rel_parts and rel_parts[0] == "xgboost":
        augmented_path = project_dir / "experiments" / "augmentation" / "generated" / "augmented_dataset.csv"
        source_path = augmented_path if augmented_path.exists() else project_dir / "data" / "processed" / "data_vectorized.csv"
        return str(source_path.resolve()), inferred
    if rel_parts and rel_parts[0] == "data_argmentation_A" and group_name:
        return str((project_dir / "data_argmentation_A" / "datasets" / group_name / "augmented_dataset.csv").resolve()), inferred
    return "", False


def _metrics_record_from_json(path: Path, project_dir: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rel_parts = path.relative_to(project_dir).parts
    dataset_summary = data.get("dataset_summary", {}) if isinstance(data.get("dataset_summary"), dict) else {}
    high_4 = data.get("high_pain_threshold_4", {}) if isinstance(data.get("high_pain_threshold_4"), dict) else {}
    high_5 = data.get("high_pain_threshold_5", {}) if isinstance(data.get("high_pain_threshold_5"), dict) else {}
    hyper = data.get("hyperparameter_search", {}) if isinstance(data.get("hyperparameter_search"), dict) else {}

    record = _blank_record()
    group_name = str(
        data.get("group_name")
        or dataset_summary.get("group_name")
        or data.get("strategy_name")
        or path.parent.name
    )
    record.update(
        {
            "record_origin": "metrics_json",
            "task_type": "regression",
            "project_section": rel_parts[0] if rel_parts else "",
            "run_name": _infer_run_name(rel_parts, group_name=group_name if rel_parts[:1] == ("data_argmentation_A",) else ""),
            "model_family": _infer_model_family(rel_parts),
            "model_name": data.get("model_name") or _infer_model_family(rel_parts) or path.parent.name,
            "group_name": group_name,
            "strategy_name": data.get("strategy_name") or "",
            "target": data.get("target_column") or dataset_summary.get("target_column") or "",
            "target_column": data.get("target_column") or dataset_summary.get("target_column") or "",
            "source_file": _record_path(path, project_dir),
            "source_mtime_utc": _mtime_utc(path),
            "output_dir": data.get("output_dir") or _record_path(path.parent, project_dir),
            "n_samples": data.get("n_samples"),
            "n_features": data.get("n_features"),
            "train_size": data.get("train_size"),
            "val_size": data.get("val_size"),
            "test_size": data.get("test_size"),
            "train_original_size": data.get("train_original_size"),
            "train_generated_size": data.get("train_generated_size"),
            "test_original_size": data.get("test_original_size"),
            "test_generated_size": data.get("test_generated_size"),
            "auc": data.get("auc"),
            "auprc": data.get("auprc"),
            "positive_rate": data.get("positive_rate"),
            "auprc_lift": data.get("auprc_lift"),
            "acc": data.get("acc") or data.get("accuracy"),
            "accuracy": data.get("accuracy"),
            "balanced_accuracy": data.get("balanced_accuracy"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "specificity": data.get("specificity"),
            "f1": data.get("f1"),
            "log_loss": data.get("log_loss"),
            "brier": data.get("brier"),
            "r2": data.get("r2"),
            "rmse": data.get("rmse"),
            "mae": data.get("mae"),
            "tn": data.get("tn"),
            "fp": data.get("fp"),
            "fn": data.get("fn"),
            "tp": data.get("tp"),
            "chosen_threshold": data.get("chosen_threshold"),
            "train_loss": data.get("train_loss"),
            "hyperparam_search_enabled": hyper.get("enabled"),
            "hyperparam_search_scoring": hyper.get("scoring"),
            "hyperparam_search_best_score": hyper.get("best_score"),
            "hyperparam_search_best_params": hyper.get("best_params"),
            "high_pain_recall_at_4": high_4.get("recall"),
            "high_pain_precision_at_4": high_4.get("precision"),
            "high_pain_f1_at_4": high_4.get("f1"),
            "high_pain_subset_mae_at_4": high_4.get("subset_mae"),
            "high_pain_recall_at_5": high_5.get("recall"),
            "high_pain_precision_at_5": high_5.get("precision"),
            "high_pain_f1_at_5": high_5.get("f1"),
            "high_pain_subset_mae_at_5": high_5.get("subset_mae"),
        }
    )

    if not record["target_column"]:
        config_map = {
            "randomforest": project_dir / "models" / "random_forest" / "config.py",
            "xgboost": project_dir / "models" / "xgboost" / "config.py",
            "models": project_dir / "models" / (rel_parts[1] if len(rel_parts) > 1 else "") / "config.py",
            "data_argmentation_A": project_dir / "data_argmentation_A" / "config.py",
        }
        config_path = config_map.get(rel_parts[0])
        if "randomforest_standard_rawdata" in rel_parts:
            config_path = project_dir / "models" / "random_forest" / "config.py"
            record["run_name"] = "random_forest_raw"
        if config_path is not None:
            target_column = _read_target_from_config(config_path)
            record["target_column"] = target_column
            record["target"] = target_column

    source_data_path = data.get("source_data_path")
    inferred_flag = False
    if not source_data_path:
        source_data_path, inferred_flag = _infer_source_data_path(project_dir, rel_parts, group_name=group_name)
    record["source_data_path"] = source_data_path or ""
    record["source_data_path_inferred"] = inferred_flag

    return {key: _normalize_scalar(value) for key, value in record.items()}


def _overview_records_from_csv(path: Path, project_dir: Path) -> list[dict]:
    rel_parts = path.relative_to(project_dir).parts
    run_name = _infer_run_name(rel_parts)
    group_name = run_name
    if len(rel_parts) >= 4 and rel_parts[0] == "models" and rel_parts[2] == "outputs":
        group_name = "raw" if rel_parts[3].endswith(".csv") else rel_parts[3]
    output_dir = path.parent
    records: list[dict] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = _blank_record()
            record.update(
                {
                    "record_origin": "prediction_overview_all_targets_csv",
                    "task_type": "classification",
                    "project_section": rel_parts[0] if rel_parts else "",
                    "run_name": run_name,
                    "model_family": _infer_model_family(rel_parts),
                    "model_name": _infer_model_family(rel_parts),
                    "group_name": group_name,
                    "target": row.get("target", ""),
                    "target_column": row.get("target_column") or row.get("target", ""),
                    "class_0_label": row.get("class_0_label", ""),
                    "class_1_label": row.get("class_1_label", ""),
                    "source_file": _record_path(path, project_dir),
                    "source_mtime_utc": _mtime_utc(path),
                    "output_dir": _record_path(output_dir, project_dir),
                    "n_samples": row.get("n_total_non_missing"),
                    "train_size": row.get("n_train"),
                    "val_size": row.get("n_val"),
                    "test_size": row.get("n_test"),
                    "auc": row.get("auc"),
                    "auprc": row.get("auprc"),
                    "positive_rate": row.get("positive_rate"),
                    "auprc_lift": row.get("auprc_lift"),
                    "acc": row.get("acc") or row.get("accuracy"),
                    "accuracy": row.get("accuracy"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "specificity": row.get("specificity"),
                    "f1": row.get("f1"),
                    "log_loss": row.get("log_loss"),
                    "brier": row.get("brier"),
                    "rmse": row.get("rmse"),
                    "mae": row.get("mae"),
                    "tn": row.get("tn"),
                    "fp": row.get("fp"),
                    "fn": row.get("fn"),
                    "tp": row.get("tp"),
                    "chosen_threshold": row.get("chosen_threshold"),
                    "train_loss": row.get("train_loss"),
                }
            )
            records.append({key: _normalize_scalar(value) for key, value in record.items()})

    return records


def collect_registry_rows(project_dir: Path | None = None) -> list[dict]:
    root = _find_project_dir(project_dir or Path(__file__).resolve())
    rows: list[dict] = []

    for metrics_path in sorted(root.rglob("metrics.json")):
        if _should_skip_path(metrics_path):
            continue
        rows.append(_metrics_record_from_json(metrics_path, root))

    for overview_path in sorted(root.rglob("prediction_overview_all_targets.csv")):
        if _should_skip_path(overview_path):
            continue
        rows.extend(_overview_records_from_csv(overview_path, root))

    rows.sort(
        key=lambda row: (
            str(row.get("project_section", "")),
            str(row.get("run_name", "")),
            str(row.get("model_name", "")),
            str(row.get("target", "")),
            str(row.get("source_file", "")),
        )
    )
    return rows


def _summary_record_from_registry_row(row: dict) -> dict:
    """Reduce the detailed registry row to the compact comparison schema."""
    return {
        "model": row.get("model_name") or row.get("model_family") or "",
        "task": row.get("task_type") or "",
        "target": row.get("target") or row.get("target_column") or "",
        "data_version": row.get("group_name") or row.get("run_name") or "",
        "mae": row.get("mae") or "",
        "rmse": row.get("rmse") or "",
        "r2": row.get("r2") or "",
        "auroc": row.get("auc") or "",
        "auprc": row.get("auprc") or "",
        "recall": row.get("recall") or "",
        "precision": row.get("precision") or "",
        "f1": row.get("f1") or "",
        "source_file": row.get("source_file") or "",
        "updated_at": row.get("source_mtime_utc") or "",
    }


def collect_summary_rows(project_dir: Path | None = None) -> list[dict]:
    rows = collect_registry_rows(project_dir)
    summary_rows = [_summary_record_from_registry_row(row) for row in rows]
    summary_rows.sort(
        key=lambda row: (
            str(row.get("task", "")),
            str(row.get("model", "")),
            str(row.get("target", "")),
            str(row.get("data_version", "")),
            str(row.get("source_file", "")),
        )
    )
    return summary_rows


def refresh_registry(project_dir: Path | None = None) -> Path:
    root = _find_project_dir(project_dir or Path(__file__).resolve())
    rows = collect_registry_rows(root)
    out_path = root / REGISTRY_FILENAME
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [_summary_record_from_registry_row(row) for row in rows]
    summary_rows.sort(
        key=lambda row: (
            str(row.get("task", "")),
            str(row.get("model", "")),
            str(row.get("target", "")),
            str(row.get("data_version", "")),
            str(row.get("source_file", "")),
        )
    )
    summary_path = root / SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return out_path


def main() -> None:
    out_path = refresh_registry()
    print(f"Model output registry refreshed: {out_path}")
    print(f"Model results summary refreshed: {out_path.parent / Path(SUMMARY_FILENAME).name}")


if __name__ == "__main__":
    main()
