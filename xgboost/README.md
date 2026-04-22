# XGBoost 回归工程（pain_prediction/xgboost）

这个目录提供了一个和现有 `randomforest/` 平行的 XGBoost 回归模板，方便你在相同数据与输出习惯下继续做模型训练、推理和可视化。

## 1. 工程结构

```text
pain_prediction/xgboost/
├── README.md                        # 使用说明（本文件）
├── config.py                        # 集中配置：数据路径/目标列/超参数
├── temporal_feature_filter.py       # 时间防泄漏过滤
├── train_xgboost_regression.py      # 训练 + 评估 + 保存模型
├── predict.py                       # 加载模型做批量推理
├── visualize_outputs.py             # 基于输出文件绘图
├── artifacts/
│   └── xgb_regressor.joblib         # 训练后保存的模型文件
└── outputs/
    ├── metrics.json                 # 回归指标（MAE/RMSE/R2）
    ├── test_predictions.csv         # 测试集真实值与预测值
    ├── feature_importance.csv       # 特征重要性
    ├── xgb_search_results.csv       # 随机搜索每组参数的 CV 结果
    ├── inference_predictions.csv    # 全量数据推理结果（来自 predict.py）
    └── plots/                       # 可视化图片输出目录
```

## 2. 快速开始

1. 安装或更新 pixi 环境

```bash
cd pain_prediction
pixi install
```

2. 训练模型

```bash
cd pain_prediction
pixi run xgb-train
```

默认行为和 `randomforest/` 保持一致：

- 如果 `data_augmentation/generated/augmented_dataset.csv` 存在，会优先读取它
- 否则回退到 `xgboost/config.py` 里的 `DATA_PATH`

3. 推理

```bash
cd pain_prediction
pixi run xgb-predict
```

4. 生成图表

```bash
cd pain_prediction
pixi run xgb-plot
```

## 3. 数据调用规则

如果输入的是带 metadata 的增强数据，例如 `data_augmentation/generated/augmented_dataset.csv`，训练脚本会自动识别：

- `__meta_is_generated`
- `__meta_dataset_split`
- `__meta_source_row_id`

这意味着：

- 固定测试集会被保留下来
- 增强样本只会进入训练池
- 训练时仍会采用“未增强原始样本全保留；从增强池随机抽取等量样本”的平衡采样规则

如果输入数据没有显式测试集标记，则会回退到 `config.py` 里的 `TEST_SIZE=0.2` 做随机切分。

## 4. 你最可能手动调整的地方

1. `config.py -> TARGET_COLUMN`
- 改你要预测的结局变量

2. `config.py -> XGB_PARAMS`
- 改 XGBoost 默认超参数

3. `config.py -> XGB_SEARCH_PARAM_DISTRIBUTIONS`
- 改随机搜索的参数空间

4. `config.py -> ENABLE_TEMPORAL_FILTER / STRICT_PAST_ONLY`
- 控制是否启用时间防泄漏过滤

## 5. 输出结果怎么解读

- `metrics.json`
  - `mae` 越小越好
  - `rmse` 越小越好
  - `r2` 越接近 1 越好
- `feature_importance.csv`
  - 表示 XGBoost 当前对各特征的相对重要性
- `xgb_search_results.csv`
  - 记录每组超参数在交叉验证中的表现
- `outputs/plots/`
  - 包含真实值 vs 预测值、误差分布、不同真实值下的 MAE、Top20 特征重要性等图
