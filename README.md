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

## 实验思路

### 第一阶段：数据清洗与向量化

- 对于手术类型采用one-hot编码方式，对于麻醉剂量等可能包含多个字段的数据采用多标签one-hot编码方式
- 对于药物配方等需要区分剂量的数据，每种药物均生成两类特征：一类为是否使用该药物的0/1特征，另一类为该药物的剂量特征。若文本中未出现该药物，则对应剂量设为0

### 第二阶段：基线模型确定与问题分析

- 结局变量处理：对于缺失结局变量的情况（疼痛相关结局变量缺失情况不多于0.8%），故可以直接删去这部分数据
- baseline：
    1. 分类任务：逻辑回归基线模型，对“是否发生高痛”进行判断
    2. 回归任务：随机森林基线模型，对“术后疼痛分数”做出预测
- 存在的问题：由于数据极度不平衡，高痛占比不足4%，且对很多疼痛值存在缺失，这也是接下来需要攻克的问题

### 第三阶段：数据增强、模型增强与现有结果汇总

- 数据增强方法：
    - 过采样（已经加在模型里）
    - 手动规则加抖动
    - Borderline-SMOTE处理（仅分类任务）
    - CTGAN与TVAE处理

- 模型增强方法：
    - logistic regression：
        1. 针对类别不平衡做了正类加权，训练时会根据 batch 里的正负样本比例，给正类更高权重
        2. 对高疼痛样本额外oversampling，在 mini-batch 采样时更频繁抽到小样本
        3. 对高疼痛样本额外loss weight，漏判惩罚
        4. 引入阈值调优策略，使用验证集自动调分类阈值
    - MLP：
        1. 学习非线性关系与特征交互
        2. 引入了shared backbone、focal loss等优化策略
    -random forest：
        还需要调整：引入类别不平衡处理、focal loss、阈值优化
        **暂时没有模型上的提高**
    - xgboost:
        同上

- 现有的**术后第一天-静息痛**预测情况：




        



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
pixi run smote-build
pixi run logistic-train-smote
pixi run mlp-train-smote
pixi run gen-tvae-cls-build
pixi run gen-tvae-reg-build
pixi run logistic-train-tvae
pixi run mlp-train-tvae
pixi run rf-train-tvae
pixi run xgb-train-tvae
```

## 数据与切分

- 原始数据：`data/raw/data.csv`
- 向量化数据：`data/processed/data_vectorized.csv`
- 固定切分：`splits/reference_splits_seed_42.csv`
- 切分汇总：`splits/reference_splits_summary.csv`

低事件目标例如 POD3 静息痛和 POD3 恶心呕吐在 10% 测试集中阳性样本很少，只应作为探索性结果解释。
