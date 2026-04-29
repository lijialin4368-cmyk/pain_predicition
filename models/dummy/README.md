# Dummy Baseline

单目标 naive 回归 baseline，用于建立最低性能参照线。默认目标为 `config.py` 中的 `TARGET_COLUMN`。

运行：

```bash
pixi run dummy-train-mean
pixi run dummy-train-median
```

输出：

```text
artifacts/raw_mean/dummy_mean_regressor.joblib
artifacts/raw_median/dummy_median_regressor.joblib
outputs/raw_mean/metrics.json
outputs/raw_median/metrics.json
outputs/*/test_predictions.csv
```

解释：

- mean predictor：所有测试样本都预测训练集目标均值。
- median predictor：所有测试样本都预测训练集目标中位数。
- 任何 RandomForest、XGBoost、LightGBM、Transformer 回归模型都应至少优于该 baseline。
