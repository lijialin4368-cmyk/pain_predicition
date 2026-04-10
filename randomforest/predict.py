"""使用已训练好的随机森林回归模型做推理。

运行方式（示例）：
    cd pain_prediction/randomforest
    python3 predict.py
"""

from __future__ import annotations

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


# 这里默认读取训练时保存的模型。
MODEL_PATH = ARTIFACT_DIR / "rf_regressor.joblib"

# 推理结果输出文件。
PREDICT_OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "inference_predictions.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在，请先训练: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # 和训练时保持一致，避免训练-推理特征不一致。
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

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    preds = model.predict(X)

    out_df = df.copy()
    out_df["rf_prediction"] = preds

    PREDICT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("推理完成。")
    print(f"输出文件: {PREDICT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
