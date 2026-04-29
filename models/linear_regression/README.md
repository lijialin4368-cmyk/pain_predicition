# Linear Regression Baseline

单目标线性回归 baseline，用于衡量线性可解释模型在连续术后结局预测上的表现。默认目标为 `config.py` 中的 `TARGET_COLUMN`。

运行：

```bash
pixi run linear-train-ridge
pixi run linear-train-elasticnet
```

输出：

```text
artifacts/raw_ridge/ridge_regressor.joblib
artifacts/raw_elasticnet/elasticnet_regressor.joblib
outputs/raw_ridge/metrics.json
outputs/raw_elasticnet/metrics.json
outputs/*/test_predictions.csv
outputs/*/coefficients.csv
outputs/*/linear_search_results.csv
```

解释：

- Ridge：推荐作为默认线性回归 baseline，稳定、可解释、对共线性更稳。
- ElasticNet：用于稀疏化特征和观察变量选择趋势。
- 如果树模型或 boosting 模型没有显著优于 Ridge，需要优先检查特征泄漏、目标定义和切分策略，而不是继续加复杂模型。
