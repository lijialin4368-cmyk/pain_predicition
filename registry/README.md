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
auroc, auprc, recall, precision, f1, source_file, updated_at
```

约定：

- 回归模型优先输出 `metrics.json`，由 registry 自动汇总 `mae/rmse/r2`。
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
