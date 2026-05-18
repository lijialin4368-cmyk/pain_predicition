# Registry

统一汇总模型输出。脚本会扫描项目中的 `metrics.json` 和 `prediction_overview_all_targets.csv`。

当前生成两个表：

```text
model_results_summary.csv  核心结果表，用于模型横向比较
model_output_registry.csv  详细结果表，保留更多训练、切分和超参数字段
```

核心结果表字段：

```text
model, task, target, data_version, mae, rmse, r2,
high_pain_recall_at_4, high_pain_precision_at_4,
high_pain_f1_at_4, high_pain_subset_mae_at_4,
auroc, auprc, recall, precision, f1, source_file, updated_at
```

约定：

- 回归模型优先输出 `metrics.json`，由 registry 自动汇总 `mae/rmse/r2` 和 `threshold>=4` 高痛识别指标。
- 如果旧版 `metrics.json` 缺少高痛识别指标，registry 会优先从同目录 `test_predictions.csv` 反算 `high_pain_*_at_4`。
- `undefined` 表示该指标分母为 0 或无对应样本，不等同于 `0.0`；例如没有任何预测高痛样本时，precision 为 `undefined`，但 recall 可以是真实的 `0.0`。
- 多目标分类模型当前保留 `prediction_overview_all_targets.csv`，由 registry 自动汇总 `auroc/auprc/recall/precision/f1`。
- 现有训练脚本在训练结束后会调用 `refresh_registry()`，因此新结果会自动写入 summary 和详细表。

运行：

```bash
pixi run registry-refresh
```

直接运行：

```bash
pixi run python registry/model_output_registry.py
```

输出：

```text
registry/model_results_summary.csv
registry/model_output_registry.csv
```
