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

### 第三阶段：数据增强、模型增强与之后的plan

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
    - randomforest：
        此模型是一个比较基本的随机森林，没有太多buff叠加
    - randomforest tail aware：
        此模型为随机森林升级版，主要有以下提高：
        1. 高痛样本加权训练
        2. 回归训练的类focal loss：先训练一版rf，抓住训练集上误差大的样本，第二轮增加这些难样本的权重
        3. 优化预测分位数：收集每棵树的预测情况，在validation set上进行加权尝试，找到最合适的输出策略
        4. 偏差校正：RF base prediction -> residual calibrator -> corrected prediction
    - xgboost:
        - 目前回归任务预测整体效果最好，但对于高痛的预测始终不够
        - CTGAN与TVAE处理后，虽然在整体表现上不如raw，但是**提高了高痛的recall**

- 各种情况的输出以表格形式列出

- 现存问题与plan
    1. 发现数据不平衡的问题得到缓解，但是整体而言2000+样本有些少，下一步打算等比例（低痛高痛比）放大一下数据集
        - 通过学习现有数据的趋势，来给出一些模拟数据，扩大一下数据集
            - 现在的ctgan、tvae就在做，但是会抹平尾端，而且测试集似乎有点小（？）
        - 针对**进行数据增强后，各项指标反而不如raw的情况**：的确有反映这种情况的论文，正在研究
    2. 目标更改：主要预测手术当天+第一天的各项指标，第二、三天的指标预测作为探索进行尝试性预测
    3. 模型扩增：
        - LightGBM
        - CatBoost
    4. 交叉验证+将已有的模型扩展到对手术当天、术后第一天的静息痛、活动痛、恶心呕吐等五类情况（共10个结局变量）的预测上





        



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
