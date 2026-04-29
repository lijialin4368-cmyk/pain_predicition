# XGBoost

XGBoost 回归 baseline。默认预测目标在 `config.py` 的 `TARGET_COLUMN` 中配置，当前默认使用时间防泄漏过滤。

训练原始数据 baseline：

```bash
pixi run xgb-train-raw
```

训练规则增强数据：

```bash
pixi run xgb-train-aug
```

推理和绘图：

```bash
pixi run xgb-predict-raw
pixi run xgb-plot-raw
pixi run xgb-plot-aug
```

主要脚本：

```text
train_regression.py       训练、调参、评估、保存模型
predict.py                批量推理
plot.py                   读取输出文件绘图
config.py                 数据、目标列、超参数配置
temporal_feature_filter.py 时间防泄漏特征过滤
```

