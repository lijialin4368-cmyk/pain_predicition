# Baseline README

## 目录说明
这个目录当前用于保存 pain prediction 的基础版二分类 baseline。
目前实际用于复现和导出结果的主脚本是：

- `final_baseline_logistic_re/train_logistic_regression.py`

对应输出目录是：

- `final_baseline_logistic_re/output_prediction_report_en/`

## 任务定义
该 baseline 不是直接回归术后疼痛原始分值，而是把结局变量离散化为二分类标签，再预测样本属于高风险类别的概率。

当前脚本支持的结局日：

- `手术当天`
- `术后第一天`
- `术后第二天`
- `术后第三天`

当前脚本支持的结局指标：

- `静息痛`
- `活动痛`
- `镇静评分`
- `活动状态`
- `恶心呕吐`

标签构造规则如下：

- `静息痛` / `活动痛`: `0-3 -> class 0`, `>=4 -> class 1`
- `镇静评分`: `1-2 -> class 0`, `3-5 -> class 1`
- `活动状态`: `1-2 -> class 0`, `3-4 -> class 1`
- `恶心呕吐`: `0-1 -> class 0`, `2-3 -> class 1`

因此，模型输出的 `test_prob` 表示“该样本属于 class 1 的概率分数”，最终再通过阈值转换成 `0/1` 分类结果。

## 模型概览
这是一个手写的 numpy 版本 logistic regression baseline，不依赖 sklearn 的 `LogisticRegression`，也不依赖 PyTorch。

核心特征：

- 单输出 logistic regression
- `sigmoid` 输出概率
- mini-batch 梯度下降训练
- 支持 L2 正则
- 支持类别不平衡下的正类加权
- 支持对高痛样本额外加权和过采样
- 支持在验证集上调阈值

从实现上看，它更接近“可控的研究用 baseline”，而不是标准库封装模型。

## 默认配置
脚本默认行为：

- 默认数据文件：`pain_prediction/data_vectorized.csv`
- 默认预测日：`术后第一天`
- 默认任务类型：`all`
- 默认特征模式：`strict`
- 默认缺失值填补：`median`
- 默认测试集比例：`0.2`
- 默认训练轮数：`2000`
- 默认 batch size：`128`
- 默认学习率：`0.05`
- 默认阈值策略：`tune`

其中 `feature-mode` 的含义：

- `strict`: 不把任何 outcome 列作为特征
- `all`: 除当前 target 外，其他列都可作为特征，可能造成泄漏
- `temporal`: 使用非 outcome 特征和更早时间点的 outcome 特征

## 训练流程
整体流程如下：

1. 读取 `data_vectorized.csv`
2. 按目标列提取原始结局分数
3. 把原始分数转换为二分类标签
4. 删除该 target 缺失的样本
5. 划分 train / val / test
6. 标准化特征
7. 用 logistic regression 训练概率模型
8. 在验证集上选择分类阈值
9. 在测试集上输出 metrics 和混淆矩阵

## 常用运行方式
在仓库根目录 `pain_prediction/` 下运行：

```bash
python3 baseline/final_baseline_logistic_re/train_logistic_regression.py
```

只跑 POD3 静息痛：

```bash
python3 baseline/final_baseline_logistic_re/train_logistic_regression.py \
  --day "术后第三天" \
  --pain-type rest
```

只跑 POD3 活动痛，并使用 temporal 特征：

```bash
python3 baseline/final_baseline_logistic_re/train_logistic_regression.py \
  --day "术后第三天" \
  --pain-type movement \
  --feature-mode temporal
```

固定阈值而不是验证集调阈值：

```bash
python3 baseline/final_baseline_logistic_re/train_logistic_regression.py \
  --threshold-strategy fixed \
  --decision-threshold 0.5
```

## 输出文件
运行完成后，脚本会在输出目录中保留四个总表文件：

- `prediction_overview_all_targets.csv`
  - 每个 target 的 AUC、accuracy、precision、recall、specificity、F1、log loss、Brier score 等汇总结果
- `confusion_matrix_prob_all_targets.csv`
  - 每个 target 的归一化混淆矩阵和计数
- `confusion_matrix_prob_all_targets.png`
  - 各 target 的混淆矩阵图
- `training_acc_all_targets.png`
  - 各 target 的训练准确率曲线

脚本结束时会清理输出目录，只保留上述四个聚合结果文件。

## 关键参数
常用参数如下：

- `--day`: 选择结局日，或使用 `all`
- `--pain-type`: 选择 `rest`、`movement`、`both`、`all`
- `--feature-mode`: 选择特征构造方式
- `--pain-threshold`: 控制疼痛二值化阈值，默认 `4.0`
- `--positive-weight-mode`: 处理类不平衡，支持 `balanced`、`sqrt_balanced`、`none`
- `--disable-high-pain-oversampling`: 关闭高痛样本过采样
- `--high-pain-loss-weight`: 提高高痛样本损失权重
- `--threshold-strategy`: `fixed` 或 `tune`
- `--output-dir`: 自定义输出目录

## 适用场景
这个 baseline 适合：

- 做最基础的可解释二分类对照实验
- 评估仅线性边界时的性能上限或下限
- 与增强版多层模型进行公平对比
- 快速检验特征和标签定义是否合理

## 局限性
这个 baseline 的局限性也比较明确：

- 只能学习线性决策边界
- 每个 target 独立训练，不能共享不同任务之间的信息
- 对复杂非线性关系的表达能力有限
- 对极端类别不平衡时，性能往往依赖阈值调整和样本加权

## 与增强版的关系
如果需要更强的非线性建模能力、多任务共享表示、focal loss 或更灵活的阈值策略，建议看：

- `../baseline_enhance_logistic/README.md`

