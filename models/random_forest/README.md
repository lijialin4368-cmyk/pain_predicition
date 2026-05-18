# Random Forest

RandomForest 回归 baseline。默认预测目标在 `config.py` 的 `TARGET_COLUMN` 中配置，当前默认使用时间防泄漏过滤。

训练原始数据 baseline：

```bash
pixi run rf-train-raw
```

训练规则增强数据：

```bash
pixi run rf-train-aug
```

训练直接复制对照数据：

```bash
pixi run rf-train-copy
```

训练 tail-aware 回归增强版（输出到新目录，不覆盖 raw）：

```bash
pixi run rf-train-tail-aware-raw
pixi run rf-train-tail-aware-recall-raw
```

推理和绘图：

```bash
pixi run rf-predict-raw
pixi run rf-plot-raw
pixi run rf-plot-aug
pixi run rf-plot-copy
pixi run rf-plot-tail-aware-raw
pixi run rf-plot-tail-aware-recall-raw
```

主要脚本：

```text
train_regression.py       训练、调参、评估、保存模型
predict.py                批量推理
plot.py                   读取输出文件绘图
config.py                 数据、目标列、超参数配置
temporal_feature_filter.py 时间防泄漏特征过滤
```

目录布局：

```text
models/random_forest/
  train_regression.py
  predict.py
  plot.py
  config.py
  artifacts/
    raw/
    augmented/
    copy_control/
  outputs/
    raw/
    augmented/
    copy_control/
```

输出中包含总体回归指标，以及 `threshold_4_*`、`threshold_5_*` 高痛分层指标。

## Tail-Aware Regression

`rf-train-tail-aware-raw` 和 `rf-train-tail-aware-recall-raw` 保持连续评分回归任务不变，但增加了面向高痛尾部的训练和预测策略：

- `sample_weight_mode=focal_residual`：先用高痛分段权重训练一版 RF，再根据训练残差做第二轮 residual-focused reweighting。
- `high_pain_threshold=4`，`severe_pain_threshold=7`：高痛和重度高痛样本在训练中获得更高权重。
- `rf-train-tail-aware-raw`：温和版，`max_final_sample_weight=10`，更重视总体 MAE。
- `rf-train-tail-aware-recall-raw`：召回优先版，`max_final_sample_weight=25`，更重视高痛尾部召回。
- `enable_quantile_policy`：在 validation set 上比较 `mean/q50/q60/q70` 等连续预测策略，缓解 RF 均值预测对高痛的低估，同时避免过高分位数过度牺牲总体 MAE。
- `enable_residual_calibration`：如果 isotonic 校准在 validation 上改善目标分数，则自动启用校准。

新增输出目录：

```text
models/random_forest/
  artifacts/
    tail_aware_raw/
      rf_regressor.joblib
      rf_tail_aware_bundle.joblib
    tail_aware_recall_raw/
      rf_regressor.joblib
      rf_tail_aware_bundle.joblib
  outputs/
    tail_aware_raw/
      metrics.json
      test_predictions.csv
      feature_importance.csv
      rf_search_results.csv
      prediction_policy.json
      prediction_policy_candidates.csv
    tail_aware_recall_raw/
      metrics.json
      test_predictions.csv
      feature_importance.csv
      rf_search_results.csv
      prediction_policy.json
      prediction_policy_candidates.csv
```

目前的结论：
RF raw预测能力最强，在基本的randomforest模型中
