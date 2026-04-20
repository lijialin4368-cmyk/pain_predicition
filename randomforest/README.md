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
├── artifacts_comparison/             # “直接复制样本”对比组模型文件
└── outputs/
    ├── metrics.json                  # 回归指标（MAE/RMSE/R2）
    ├── test_predictions.csv          # 测试集真实值与预测值
    ├── feature_importance.csv        # 特征重要性
    ├── rf_search_results.csv         # 随机搜索每组参数的CV结果
    ├── inference_predictions.csv     # 全量数据推理结果（来自 predict.py）
    └── plots/                        # 可视化图片输出目录

pain_prediction/randomforest/outputs_comparison/
├── metrics.json                      # “直接复制样本”对比组指标
├── test_predictions.csv              # “直接复制样本”对比组测试集预测
├── feature_importance.csv            # “直接复制样本”对比组特征重要性
├── rf_search_results.csv             # “直接复制样本”对比组调参结果
├── inference_predictions.csv         # “直接复制样本”对比组全量推理结果
└── plots/                            # “直接复制样本”对比组可视化
```

## 2. 快速开始

1. 安装/更新 pixi 环境依赖

```bash
cd pain_prediction
pixi install
```

2. 修改配置（可选）

编辑 `config.py`，重点看以下字段：
- `DATA_PATH`：输入 CSV 路径（作为默认/兜底路径）
- `TARGET_COLUMN`：你要预测的目标列（必须是数值）
- `MANUAL_FEATURE_COLUMNS`：是否手动指定特征列
- `RF_PARAMS`：随机森林超参数
- `TEST_SIZE`、`RANDOM_STATE`：训练测试拆分参数

3. 用数据增强后的数据训练模型（推荐）

```bash
cd pain_prediction
pixi run augment-data
pixi run python3 randomforest/train_random_forest_regression.py \
  --data-path data_augmentation/generated/augmented_dataset.csv
```

如果你已经生成过 `data_augmentation/generated/augmented_dataset.csv`，也可以直接运行：

```bash
cd pain_prediction
pixi run rf-train
```

`rf-train` 会优先读取 `data_augmentation/generated/augmented_dataset.csv`；只有当这个文件不存在时，才会回退到 `randomforest/config.py` 里的 `DATA_PATH`。

4. 使用模型推理

```bash
cd pain_prediction
pixi run python3 randomforest/predict.py \
  --data-path data_augmentation/generated/augmented_dataset.csv
```

5. 可视化输出

```bash
cd pain_prediction
pixi run rf-plot
```

如果你想运行“直接复制样本”对比组，对应命令是：

```bash
cd pain_prediction
pixi run augment-data-comparison
pixi run rf-train-comparison
pixi run rf-plot-comparison
pixi run rf-predict-comparison
```

## 3. 数据调用规则

推荐把输入数据路径显式写在命令里，这样最不容易混淆：

```bash
cd pain_prediction
pixi run python3 randomforest/train_random_forest_regression.py \
  --data-path data_augmentation/generated/augmented_dataset.csv
```

如果你想改用原始向量化数据训练：

```bash
cd pain_prediction
pixi run python3 randomforest/train_random_forest_regression.py \
  --data-path data_vectorized.csv
```

当前训练脚本的数据读取优先级是：

1. 命令行 `--data-path`
2. `data_augmentation/generated/augmented_dataset.csv`（如果存在）
3. `randomforest/config.py` 中的 `DATA_PATH`

如果输入的是 `augmented_dataset.csv`，脚本会自动识别里面的 metadata 列：

- `__meta_is_generated`
- `__meta_dataset_split`

这意味着训练时会自动把增强样本只放进训练池，并保留原始 held-out 样本作为固定测试集。

`config.py` 里的 `TEST_SIZE` 只会在输入数据中没有显式测试集标记时才作为兜底拆分比例生效。

## 4. 目录与实验对应关系

这个仓库里和随机森林相关的结果目录不止一个，最容易混淆的是下面这几类：

### 4.1 `randomforest/outputs`

这是当前 `randomforest/` 目录下的主输出目录，默认对应“主规则增强策略”：

- 数据通常来自 `data_augmentation/generated/augmented_dataset.csv`
- 训练命令通常是 `pixi run rf-train`
- 如果 `data_augmentation/generated/augmented_dataset.csv` 存在，`rf-train` 会优先读取它
- 这里的结果代表：使用 `data_augmentation/augment_dataset.py` 那套规则增强后的随机森林输出

也就是说，`randomforest/outputs` 不是“纯原始数据 baseline”，而是当前主增强方案下的随机森林结果。

### 4.2 `randomforest/outputs_comparison`

这是“直接复制样本对比组”的输出目录，对应：

- 数据来自 `data_augmentation/comparison_direct_copy/generated/augmented_dataset.csv`
- 训练命令是 `pixi run rf-train-comparison`
- 这个对比组不会改年龄、体重、疼痛值
- 它只会把原来在主增强流程里会被增强的父样本，按相同次数直接复制

这个目录的作用是回答：

- 如果保持训练集里的新增样本比例不变，但新增样本不做规则扰动，只是原样复制，随机森林会表现成什么样

因此：

- `outputs` = 主规则增强策略
- `outputs_comparison` = 直接复制样本对比策略

### 4.3 `randomforest_standard_rawdata`

这是完全独立的原始数据 baseline 工程，不在本目录下，但和这里容易混淆：

- 直接读取 `data_vectorized.csv`
- 不依赖增强数据
- 默认对原始有效样本做随机 `80/20` 切分
- 它的结果目录在 `pain_prediction/randomforest_standard_rawdata/outputs`

这个 baseline 的意义是：

- 看“直接拿原始数据随机切分训练随机森林”时的结果

它和 `randomforest/outputs` 不同，也和 `data_argmentation_A/original_only` 不同。

### 4.4 `data_argmentation_A/outputs`

这是另一套独立实验，不在 `randomforest/` 目录下，但它也是随机森林结果：

- 它使用固定参考切分，不重新随机拆数据
- 里面包含四组严格对照：
  - `original_only`
  - `high_pain_duplicate`
  - `high_pain_local_jitter`
  - `high_pain_local_resampling`
- 每组结果在 `data_argmentation_A/outputs/<group_name>/`
- 四组汇总表在 `data_argmentation_A/outputs/experiment_comparison.csv`

这套实验的意义是：

- 在同一批固定训练/测试样本上，严格比较不同“高痛增强机制”本身的效果

### 4.5 怎么快速判断自己看到的是哪种结果

可以按下面的规则快速判断：

- 如果路径是 `randomforest/outputs/`，它通常是主规则增强策略的结果
- 如果路径是 `randomforest/outputs_comparison/`，它是“直接复制样本”对比组结果
- 如果路径是 `randomforest_standard_rawdata/outputs/`，它是原始数据随机 `80/20` baseline
- 如果路径是 `data_argmentation_A/outputs/<group>/`，它是固定参考切分下的四组高痛增强对照实验
## 5. 你最可能手动调整的地方

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

## 6. 输出结果怎么解读

- `metrics.json`
  - `mae` 越小越好
  - `rmse` 越小越好
  - `r2` 越接近 1 越好（可为负，负值代表效果差于简单基线）
- `feature_importance.csv`
  - 越靠前代表模型越依赖该特征（仅表示重要性，不代表因果关系）
- `rf_search_results.csv`
  - 记录每组随机搜索参数的交叉验证得分，可用于比较调参结果
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

## 7. 常见报错

1. `目标列不存在`
- 说明 `TARGET_COLUMN` 拼写与 CSV 表头不一致。

2. `ModuleNotFoundError`
- 说明环境没有同步成功，先执行 `pixi install`。

3. `可用样本过少`
- 说明目标列空值太多或筛选后数据不足。
