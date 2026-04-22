# Standard RF Rawdata Baseline

这个目录提供一个完全独立的标准随机森林基线：

- 直接读取原始 `data_vectorized.csv`
- 不依赖任何增强数据
- 训练、模型、图表、推理结果都只写到本目录
- 默认沿用你当前标准 RF 的同一套特征过滤、超参数搜索和随机种子

## 说明

- 由于原始数据没有增强流程里的 `__meta_dataset_split` 标记，这个基线默认会对原始有效样本执行随机 `80/20` 切分。
- 因此它和 `high_pain_scheme_a/original_only` 不同：`original_only` 用的是增强实验固定好的 `70/30` 参考切分，目的是给增强组做公平对照。

## 运行

在 `pain_prediction/` 目录下：

```bash
pixi run std-rf-raw-train
pixi run std-rf-raw-plot
pixi run std-rf-raw-predict
```

## 输出

- `outputs/metrics.json`
- `outputs/test_predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/rf_search_results.csv`
- `outputs/plots/*`
- `outputs/inference_predictions.csv`
- `artifacts/rf_regressor.joblib`
