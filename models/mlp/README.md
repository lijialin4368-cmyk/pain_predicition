# MLP Baseline

增强版分类 baseline，使用 MLP/shared backbone、focal loss 和按天阈值策略。它原来位于 `baseline_enhance_logistic/`，现在统一放在 `models/mlp/`。

运行：

```bash
pixi run mlp-train
```

使用 BorderlineSMOTE 训练集增强数据运行单目标高痛分类：

```bash
pixi run smote-build
pixi run mlp-train-smote
```

直接运行脚本：

```bash
pixi run python models/mlp/train.py \
  --data-path data/processed/data_vectorized.csv \
  --split-file splits/reference_splits_seed_42.csv \
  --split-seed 42 \
  --output-dir models/mlp/outputs
```

默认输入为 `data/processed/data_vectorized.csv`，输出保存到 `models/mlp/outputs/`。

`--data-path` 是 `--input` 的别名。若输入数据包含 `__meta_dataset_split`，脚本会优先使用该列中的 `train`、`validation`、`test` 标记；否则使用 `--split-file` 或随机分层切分。`--single-target` 会关闭 shared-backbone 多目标路径，用于类似 BorderlineSMOTE 这类只服务于一个目标的增强数据。
