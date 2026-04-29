# High Pain Augmentation Tuning

用于调试 logistic baseline 的高痛样本 oversampling 和 loss weight 参数。

运行：

```bash
pixi run hp-logistic-tune
```

直接运行：

```bash
pixi run python experiments/high_pain_augmentation/tune_logistic_high_pain.py \
  --train-script models/logistic/train.py \
  --extra-args --split-file splits/reference_splits_seed_42.csv --split-seed 42
```

输出默认写入 `experiments/high_pain_augmentation/tuning_records/`。

