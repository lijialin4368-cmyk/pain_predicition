# Random Forest

RandomForest 回归 baseline。默认预测目标在 `config.py` 的 `TARGET_COLUMN` 中配置，当前默认使用时间防泄漏过滤。

训练原始数据 baseline：

```bash
pixi run rf-train-raw
```

训练规则增强数据：

```bash
pixi run rf-train-aug
```

训练直接复制对照数据：

```bash
pixi run rf-train-copy
```

推理和绘图：

```bash
pixi run rf-predict-raw
pixi run rf-plot-raw
pixi run rf-plot-aug
pixi run rf-plot-copy
```

主要脚本：

```text
train_regression.py       训练、调参、评估、保存模型
predict.py                批量推理
plot.py                   读取输出文件绘图
config.py                 数据、目标列、超参数配置
temporal_feature_filter.py 时间防泄漏特征过滤
```

目录布局：

```text
models/random_forest/
  train_regression.py
  predict.py
  plot.py
  config.py
  artifacts/
    raw/
    augmented/
    copy_control/
  outputs/
    raw/
    augmented/
    copy_control/
```

输出中包含总体回归指标，以及 `threshold_4_*`、`threshold_5_*` 高痛分层指标。

目前的结论：
RF raw预测能力最强，在基本的randomforest模型中
