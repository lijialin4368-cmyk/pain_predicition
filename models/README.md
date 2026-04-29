# Models

模型目录：

```text
dummy/             分类任务最低baseline，是mean/median predictor（均值中位数预测）
linear_regression/ 线性回归默认baseline，Ridge稳定、可解释、对共线性更稳，ElasticNet用于稀疏化特征和观察变量选择趋势
logistic/          线性 logistic classification baseline
mlp/               MLP/shared-backbone classification baseline，是对logistic regression的优化
random_forest/     RandomForest regression baseline
xgboost/           XGBoost regression baseline
```

统一约定：

```text
outputs/raw/            原始数据 baseline 输出
outputs/augmented/      规则增强数据输出
outputs/copy_control/   直接复制对照输出
artifacts/raw/          原始数据模型文件
artifacts/augmented/    增强数据模型文件
artifacts/copy_control/ 直接复制对照模型文件
```

