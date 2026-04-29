# MLP Baseline

增强版分类 baseline，使用 MLP/shared backbone、focal loss 和按天阈值策略。它原来位于 `baseline_enhance_logistic/`，现在统一放在 `models/mlp/`。

运行：

```bash
pixi run mlp-train
```

直接运行脚本：

```bash
pixi run python models/mlp/train.py \
  --split-file splits/reference_splits_seed_42.csv \
  --split-seed 42 \
  --output-dir models/mlp/outputs
```

默认输入为 `data/processed/data_vectorized.csv`，输出保存到 `models/mlp/outputs/`。

