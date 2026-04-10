# RandomForest 回归工程（pain_prediction/randomforest）

这个目录提供了一个可直接运行的随机森林回归模板，目标是帮助你快速做“疼痛评分”等连续变量预测。

## 1. 工程结构

```text
pain_prediction/randomforest/
├── README.md                         # 使用说明（本文件）
├── config.py                         # 集中配置：数据路径/目标列/超参数
├── train_random_forest_regression.py # 训练 + 评估 + 保存模型
├── predict.py                        # 加载模型做批量推理
├── visualize_outputs.py              # 基于输出文件绘图
├── artifacts/
│   └── rf_regressor.joblib           # 训练后保存的模型文件
└── outputs/
    ├── metrics.json                  # 回归指标（MAE/RMSE/R2）
    ├── test_predictions.csv          # 测试集真实值与预测值
    ├── feature_importance.csv        # 特征重要性
    └── inference_predictions.csv     # 全量数据推理结果（来自 predict.py）
    └── plots/                        # 可视化图片输出目录
```

## 2. 快速开始

1. 安装/更新 pixi 环境依赖

```bash
cd pain_prediction
pixi install
```

2. 修改配置（可选）

编辑 `config.py`，重点看以下字段：
- `DATA_PATH`：输入 CSV 路径
- `TARGET_COLUMN`：你要预测的目标列（必须是数值）
- `MANUAL_FEATURE_COLUMNS`：是否手动指定特征列
- `RF_PARAMS`：随机森林超参数
- `TEST_SIZE`、`RANDOM_STATE`：训练测试拆分参数

3. 训练模型

```bash
pixi run rf-train
```

4. 使用模型推理

```bash
pixi run rf-predict
```

5. 可视化输出

```bash
pixi run rf-plot
```

## 3. 你最可能手动调整的地方

1. 目标列：`config.py -> TARGET_COLUMN`
- 比如从 `术后第一天_静息痛` 改为 `术后第二天_活动痛`。

2. 特征列：`config.py -> MANUAL_FEATURE_COLUMNS`
- 默认 `None` 会自动使用除了目标列以外的全部列。
- 如果你只想用部分变量，可填写一个列表。

3. 时间防泄漏开关：`config.py -> ENABLE_TEMPORAL_FILTER / STRICT_PAST_ONLY`
- 默认已开启，且 `STRICT_PAST_ONLY=True`。
- 规则是“只允许目标时点之前的时间列”。
- 例如目标为 `术后第一天_静息痛` 时，只会保留 `手术当天_*`（以及年龄/体重等非时间列），会自动剔除 `术后第一天_*`、`术后第二天_*`、`术后第三天_*`。

4. 模型复杂度：`config.py -> RF_PARAMS`
- `n_estimators` 增加：通常更稳，但训练更慢。
- `max_depth` 调小：可降低过拟合风险。
- `min_samples_leaf` 调大：预测更平滑。

5. 数据切分：`config.py -> TEST_SIZE`
- 数据量小可适当减小测试集比例，比如从 `0.2` 调到 `0.1`。

6. 缺失值策略：`train_random_forest_regression.py`
- 目前使用“每列中位数填补”，你可替换成更复杂策略。

## 4. 输出结果怎么解读

- `metrics.json`
  - `mae` 越小越好
  - `rmse` 越小越好
  - `r2` 越接近 1 越好（可为负，负值代表效果差于简单基线）
- `feature_importance.csv`
  - 越靠前代表模型越依赖该特征（仅表示重要性，不代表因果关系）
- `outputs/plots/true_vs_pred.png`
  - 真实值与预测值越贴近对角线，整体拟合通常越好
- `outputs/plots/error_distribution.png`
  - 误差分布是否以 0 为中心，可看偏差方向
- `outputs/plots/mae_by_true_value.png`
  - 看不同疼痛水平下的误差是否均衡
- `outputs/plots/feature_importance_top20.png`
  - 直观看最重要的前 20 个特征（标签已自动转英文）
- `outputs/plots/feature_importance_top20_mapping.csv`
  - 英文标签 `feature_en` 与原始特征名 `feature` 的映射表

## 5. 常见报错

1. `目标列不存在`
- 说明 `TARGET_COLUMN` 拼写与 CSV 表头不一致。

2. `ModuleNotFoundError`
- 说明环境没有同步成功，先执行 `pixi install`。

3. `可用样本过少`
- 说明目标列空值太多或筛选后数据不足。
