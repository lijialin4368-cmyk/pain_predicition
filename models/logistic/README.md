# Logistic Baseline

线性 logistic regression baseline，面向二分类风险预测。默认使用固定 80/10/10 split。

运行：

```bash
pixi run logistic-train
```

直接运行脚本：

```bash
pixi run python models/logistic/train.py \
  --split-file splits/reference_splits_seed_42.csv \
  --split-seed 42 \
  --output-dir models/logistic/outputs
```

输出：

```text
outputs/prediction_overview_all_targets.csv
outputs/confusion_matrix_prob_all_targets.csv
outputs/confusion_matrix_prob_all_targets.png
outputs/training_acc_all_targets.png
```

重点指标包括 `auprc`、`auprc_lift`、`balanced_accuracy`、`recall`、`precision` 和混淆矩阵。

