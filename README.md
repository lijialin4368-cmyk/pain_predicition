# Pain Prediction

本项目用于围绕手术当天和术后 1/2/3 天结局做预测建模，重点关注疼痛评分和高痛风险。当前训练默认采用 `80/10/10` 的 train/validation/test reference split。

## 目录

```text
data/                         原始数据和向量化数据
splits/                       固定 80/10/10 切分
models/                       基线和传统模型
experiments/                  数据增强和高痛增强实验
reports/                      分布、缺失、图表分析
registry/                     模型输出总表
scripts/                      数据清洗等通用脚本
```

迁移前的 `randomforest/` 已并入 `models/random_forest/`；其余 legacy 目录仅作为历史输出/缓存保留。新的运行入口以 `models/`、`experiments/`、`reports/` 为准。

## 常用命令

```bash
pixi run clean-data
pixi run build-reference-splits
pixi run dummy-train-mean
pixi run dummy-train-median
pixi run linear-train-ridge
pixi run linear-train-elasticnet
pixi run logistic-train
pixi run mlp-train
pixi run rf-train-raw
pixi run xgb-train-raw
pixi run registry-refresh
```

模型结果汇总：

```text
registry/model_results_summary.csv  核心比较表
registry/model_output_registry.csv  详细结果表
```

现有训练脚本会在训练结束后自动刷新这两个表；也可以手动运行 `pixi run registry-refresh`。

增强数据相关：

```bash
pixi run augment-build
pixi run augment-copy-build
pixi run rf-train-aug
pixi run rf-train-copy
pixi run xgb-train-aug
```

## 数据与切分

- 原始数据：`data/raw/data.csv`
- 向量化数据：`data/processed/data_vectorized.csv`
- 固定切分：`splits/reference_splits_seed_42.csv`
- 切分汇总：`splits/reference_splits_summary.csv`

低事件目标例如 POD3 静息痛和 POD3 恶心呕吐在 10% 测试集中阳性样本很少，只应作为探索性结果解释。
