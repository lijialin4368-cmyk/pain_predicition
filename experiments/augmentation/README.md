# Augmentation

构建规则增强数据。输入默认是 `data/processed/data_vectorized.csv`，输出默认是 `experiments/augmentation/generated/`。

运行：

```bash
pixi run augment-build
```

直接运行：

```bash
pixi run python experiments/augmentation/build_augmented_data.py
```

主要输出：

```text
generated/augmented_dataset.csv
generated/train_original.csv
generated/validation_original.csv
generated/test_original.csv
generated/generated_only.csv
generated/augmentation_summary.json
```

