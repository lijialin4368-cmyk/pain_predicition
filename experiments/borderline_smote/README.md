# BorderlineSMOTE

单目标二分类增强实验。默认目标为 `术后第一天_静息痛 >= 4`，只对 train split 做 BorderlineSMOTE；validation/test 保持原始样本，避免评估污染。

运行：

```bash
pixi run smote-build
```

输出：

```text
generated/augmented_dataset.csv
generated/generated_only.csv
generated/train_original.csv
generated/validation_original.csv
generated/test_original.csv
generated/borderline_smote_summary.json
```

元数据列：

```text
__meta_is_generated
__meta_dataset_split
__meta_source_row_id
__meta_augmentation_method
__meta_target_column
__meta_target_threshold
__meta_binary_target
```

使用 logistic：

```bash
pixi run logistic-train-smote
```

注意：该数据集只服务于当前目标的二分类风险预测，不应直接作为连续评分回归数据使用。
