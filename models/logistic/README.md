# Logistic Baseline

线性 logistic regression baseline，面向二分类风险预测。默认使用固定 80/10/10 split。

运行：

```bash
pixi run logistic-train
```

使用 BorderlineSMOTE 训练集增强数据运行单目标高痛分类：

```bash
pixi run smote-build
pixi run logistic-train-smote
```

直接运行脚本：

```bash
pixi run python models/logistic/train.py \
  --data-path data/processed/data_vectorized.csv \
  --split-file splits/reference_splits_seed_42.csv \
  --split-seed 42 \
  --output-dir models/logistic/outputs
```

`--data-path` 是 `--input` 的别名。若输入数据包含 `__meta_dataset_split`，脚本会优先使用该列中的 `train`、`validation`、`test` 标记；否则使用 `--split-file` 或随机分层切分。

输出：

```text
outputs/prediction_overview_all_targets.csv
outputs/confusion_matrix_prob_all_targets.csv
outputs/confusion_matrix_prob_all_targets.png
outputs/training_acc_all_targets.png
```

重点指标包括 `auprc`、`auprc_lift`、`balanced_accuracy`、`recall`、`precision` 和混淆矩阵。
