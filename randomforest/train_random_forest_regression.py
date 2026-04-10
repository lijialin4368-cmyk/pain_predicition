"""使用随机森林完成回归任务（训练 + 评估 + 保存模型）。

运行方式：
    cd pain_prediction/randomforest
    python3 train_random_forest_regression.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    ARTIFACT_DIR,
    DATA_PATH,
    ENABLE_TEMPORAL_FILTER,
    MANUAL_FEATURE_COLUMNS,
    OUTPUT_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    STRICT_PAST_ONLY,
    TARGET_COLUMN,
    TEST_SIZE,
)
from temporal_feature_filter import apply_temporal_feature_filter


def load_data(csv_path: Path) -> pd.DataFrame:
    """读取数据并做最基础的校验。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    # encoding='utf-8-sig' 能更稳妥处理包含 BOM 的 CSV 头。
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("数据为空，无法训练模型。")
    return df


def build_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """构建特征 X 与目标 y。"""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"目标列不存在: {TARGET_COLUMN}")

    # 目标列必须是数值（回归任务）。
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    # 用户可在 config.py 手动指定要用的特征列；也可启用时间防泄漏过滤。
    if ENABLE_TEMPORAL_FILTER:
        feature_cols, dropped_cols = apply_temporal_feature_filter(
            all_columns=df.columns.tolist(),
            target_column=TARGET_COLUMN,
            manual_feature_columns=MANUAL_FEATURE_COLUMNS,
            strict_past_only=STRICT_PAST_ONLY,
        )
        if dropped_cols:
            print(f"时间过滤已启用：剔除 {len(dropped_cols)} 个目标时点及之后的时间列。")
    else:
        if MANUAL_FEATURE_COLUMNS:
            missing = [c for c in MANUAL_FEATURE_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"手动指定的特征列不存在: {missing}")
            feature_cols = [c for c in MANUAL_FEATURE_COLUMNS if c != TARGET_COLUMN]
        else:
            feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    X = df[feature_cols].copy()

    # 将所有特征转成数值；无法解析的值转 NaN，后续统一填充。
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 简单缺失值策略：每列用中位数填补。
    # 你可以在这里改成更复杂方案（例如 KNNImputer、分组填补等）。
    X = X.fillna(X.median(numeric_only=True))

    # 目标列的缺失行不能用于监督学习，直接过滤。
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if len(X) < 10:
        raise ValueError("可用样本过少（<10），建议检查数据或目标列。")

    return X, y, feature_cols


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestRegressor, dict, pd.DataFrame]:
    """拆分数据、训练模型并计算评估指标。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = RandomForestRegressor(random_state=RANDOM_STATE, **RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 评估指标：MAE / RMSE / R^2
    metrics = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "abs_error": np.abs(y_test.values - y_pred),
        }
    )

    return model, metrics, pred_df


def save_outputs(model: RandomForestRegressor, metrics: dict, pred_df: pd.DataFrame, X: pd.DataFrame) -> None:
    """保存模型、评估结果和特征重要性。"""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 保存训练后的模型
    model_path = ARTIFACT_DIR / "rf_regressor.joblib"
    joblib.dump(model, model_path)

    # 2) 保存评估指标
    metrics_path = OUTPUT_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 3) 保存测试集预测明细
    pred_path = OUTPUT_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 4) 保存特征重要性（用于解释哪些变量影响更大）
    fi_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    fi_path = OUTPUT_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    print("\n=== 训练完成，文件已保存 ===")
    print(f"模型文件: {model_path}")
    print(f"评估指标: {metrics_path}")
    print(f"预测明细: {pred_path}")
    print(f"特征重要性: {fi_path}")


def main() -> None:
    # Step 1) 读取数据
    df = load_data(DATA_PATH)
    print(f"读取数据成功: {DATA_PATH}")
    print(f"数据形状: {df.shape}")

    # Step 2) 构建训练输入
    X, y, feature_cols = build_feature_target(df)
    print(f"目标列: {TARGET_COLUMN}")
    print(f"特征数量: {len(feature_cols)}")

    # Step 3) 训练并评估
    model, metrics, pred_df = train_and_evaluate(X, y)
    print("评估结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Step 4) 保存输出
    save_outputs(model, metrics, pred_df, X)


if __name__ == "__main__":
    main()
