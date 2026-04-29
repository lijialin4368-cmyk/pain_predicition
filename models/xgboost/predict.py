"""使用已训练好的 XGBoost 回归模型做推理。

运行方式（示例）：
    cd pain_prediction
    pixi run xgb-predict-raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import (
    ARTIFACT_DIR,
    DATA_PATH,
    ENABLE_TEMPORAL_FILTER,
    MANUAL_FEATURE_COLUMNS,
    STRICT_PAST_ONLY,
    TARGET_COLUMN,
)
from temporal_feature_filter import apply_temporal_feature_filter

META_PREFIX = "__meta_"


MODEL_PATH = ARTIFACT_DIR / "xgb_regressor.joblib"
PREDICT_OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "raw" / "inference_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference with a trained XGBoost regressor.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Input CSV path for inference. Defaults to models/xgboost/config.py::DATA_PATH.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to a trained model artifact.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PREDICT_OUTPUT_PATH,
        help="Path to save inference predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型文件不存在，请先训练: {args.model_path}")

    model = joblib.load(args.model_path)
    df = pd.read_csv(args.data_path, encoding="utf-8-sig")

    if ENABLE_TEMPORAL_FILTER:
        feature_cols, dropped_cols = apply_temporal_feature_filter(
            all_columns=df.columns.tolist(),
            target_column=TARGET_COLUMN,
            manual_feature_columns=MANUAL_FEATURE_COLUMNS,
            strict_past_only=STRICT_PAST_ONLY,
        )
        if dropped_cols:
            print(f"时间过滤已启用：推理阶段剔除 {len(dropped_cols)} 个目标时点及之后的时间列。")
    else:
        if MANUAL_FEATURE_COLUMNS:
            feature_cols = [c for c in MANUAL_FEATURE_COLUMNS if c != TARGET_COLUMN]
        else:
            feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    feature_cols = [c for c in feature_cols if not str(c).startswith(META_PREFIX)]

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    preds = model.predict(X)

    out_df = df.copy()
    out_df["xgb_prediction"] = preds

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    print("推理完成。")
    print(f"输入文件: {args.data_path}")
    print(f"模型文件: {args.model_path}")
    print(f"输出文件: {args.output_path}")


if __name__ == "__main__":
    main()
