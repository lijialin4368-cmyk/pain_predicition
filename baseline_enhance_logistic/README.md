# Enhanced Logistic README

## 目录说明
这个目录现在保存了四个增强版 logistic 类模型脚本：

- `1_logistic_regression.py`
- `2_train_logistic_regression.py`
- `3_pain_logistic.py`
- `3_1_pain_logistic.py`

它们都围绕术后疼痛/相关结局预测展开，但建模方式已经分成两条路线：

- `1` / `2`: 先把原始结局离散化成二分类标签，再直接学习高风险概率
- `3`: 对 pain 任务先保留真实分数做训练，再把预测分数映射成高痛概率

如果把兄弟目录 `baseline/` 里的基础版一起算上，目前可以视为五套模型：

- baseline: 线性 logistic regression
- enhance v1: 三层非线性网络 + shared backbone + focal loss / day-relaxed threshold
- enhance v2: 在 v1 基础上增加 task-adaptive threshold
- enhance v3: pain-only 分数回归 + 概率映射 + conservative threshold tuning
- enhance v3.1: 在 v3 基础上增加高痛辅助分类损失和训练 loss 曲线导出

## 共同背景
`baseline`、`1`、`2` 的共同点是：

- 最终目标仍然是对高风险样本做 `0/1` 判断
- 最终都会输出概率，再通过阈值变成分类结果

但它们对 label 的处理方式不同。

### 路线 A：先二值化再训练
适用于：

- `baseline/final_baseline_logistic_re/train_logistic_regression.py`
- `1_logistic_regression.py`
- `2_train_logistic_regression.py`

规则如下：

- `静息痛` / `活动痛`: `0-3 -> class 0`, `>=4 -> class 1`
- `镇静评分`: `1-2 -> class 0`, `3-5 -> class 1`
- `活动状态`: `1-2 -> class 0`, `3-4 -> class 1`
- `恶心呕吐`: `0-1 -> class 0`, `2-3 -> class 1`

### 路线 B：先保留真实 pain 分数再训练
适用于：

- `3_pain_logistic.py`

这个版本只针对 pain 任务：

- `静息痛`
- `活动痛`

训练时直接保留原始 `0-10` 分数，不先转成二分类标签。
最终仍然会把预测分数映射成高痛概率，再做 `0/1` 判断。

## 五个模型的关系

| 模型 | 入口脚本 | 核心结构 | 训练目标 | 默认阈值策略 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| Baseline | `baseline/final_baseline_logistic_re/train_logistic_regression.py` | 单层线性 logistic regression | 二分类 | `tune` | 做基础对照、保留较强可解释性 |
| Enhance v1 | `baseline_enhance_logistic/1_logistic_regression.py` | 三层网络，支持 shared backbone multitask | 二分类 | `day_relaxed` | 想引入非线性表达和多任务共享 |
| Enhance v2 | `baseline_enhance_logistic/2_train_logistic_regression.py` | 三层网络，支持 shared backbone multitask | 二分类 | `task_adaptive` | 想把阈值调得更灵活 |
| Enhance v3 | `baseline_enhance_logistic/3_pain_logistic.py` | 三层网络，pain-only shared backbone | 原始 pain 分数回归 + 概率映射 | `conservative_tune` | 想保留真实 pain 分数训练，同时避免 v2 那种偏激进的阈值策略 |
| Enhance v3.1 | `baseline_enhance_logistic/3_1_pain_logistic.py` | 三层网络，pain-only shared backbone | 原始 pain 分数回归 + 辅助高痛分类 + 概率映射 | `conservative_tune` | 想在保留真实分数训练的同时，进一步拉回高痛识别能力 |

## 模型 1：`1_logistic_regression.py`
### 核心思路
这个版本是在 baseline 基础上的第一版增强：

- 从线性 logistic regression 升级为三层前馈网络
- 增加 `ReLU` 隐层，默认结构为 `input -> 96 -> 48 -> sigmoid`
- 支持 `focal loss` 和 `BCE`
- 支持多任务 shared backbone 训练
- 支持按天放宽阈值的 `day_relaxed` 策略
- 支持高痛样本权重增强和临床高痛额外损失惩罚

### 默认配置
- 默认 `feature-mode`: `strong_signal_temporal`
- 默认 `loss-type`: `focal`
- 默认阈值策略：`day_relaxed`
- 默认 day thresholds: `0.55,0.50,0.45,0.40`
- 默认隐藏层：`96 / 48`
- 默认学习率：`0.01`

### 适合什么时候用
建议在这些场景使用 v1：

- 你已经确认线性 baseline 不够表达复杂关系
- 你希望利用前几天 outcome 作为强信号
- 你希望在多个相关任务之间共享表示
- 你希望对类别不平衡使用 focal loss

## 模型 2：`2_train_logistic_regression.py`
### 核心思路
这个版本是在 v1 基础上的第二版增强，主要改进集中在阈值选择策略上。

它保留了 v1 的大多数结构和训练机制，同时增加了：

- `task_adaptive` 阈值策略
- 基于验证集正样本数的保护机制
- `recall_keep_ratio` / `recall_keep_abs_drop` 召回率护栏
- `task_adaptive_max_shift` 阈值偏移限制
- 在保证 recall 不明显下降的前提下，优先提升 accuracy

### 默认配置
- 默认 `feature-mode`: `strong_signal_temporal`
- 默认 `loss-type`: `focal`
- 默认阈值策略：`task_adaptive`
- 默认隐藏层：`96 / 48`
- 默认学习率：`0.01`
- 默认 `task_adaptive_min_val_pos`: `8`
- 默认 `task_adaptive_max_shift`: `0.12`

### 和 v1 的关键区别
v1 和 v2 的网络结构基本一致，最大的差别不在 backbone，而在“如何把概率切成最终分类”。

v1 主要使用：

- `fixed`
- `tune`
- `day_relaxed`

v2 额外提供：

- `task_adaptive`

`task_adaptive` 的目标不是简单追求单个指标最大，而是在每个任务上根据验证集表现自适应地微调阈值，同时给 recall 设置底线，避免为了降低假阳性而把真正高痛样本漏掉太多。

### 适合什么时候用
建议在这些场景使用 v2：

- 你已经有一套相对稳定的概率模型
- 你发现不同任务共用同一个阈值并不理想
- 你希望控制 recall 不掉太多的前提下提升总体 accuracy
- 你想让阈值选择更贴近任务特性而不是全局固定

## 模型 3：`3_pain_logistic.py`
### 核心思路
这个版本是专门为 pain 任务新增的一条分支，不继续沿用“先二值化再训练”的做法，而是：

1. 保留真实 pain 分数作为训练目标
2. 用三层 shared-backbone 网络预测连续 pain score
3. 再把预测分数平滑映射成高痛概率
4. 最后再用阈值完成 `0/1` 判断

也就是说，它不是直接回归后就结束，而是把“分数学习”和“高痛概率判定”接在一起。

### 当前实现范围
这个版本只处理 pain 任务：

- `静息痛`
- `活动痛`

不包含：

- `镇静评分`
- `活动状态`
- `恶心呕吐`

### 模型和损失
`3_pain_logistic.py` 的默认结构和 `1` 基本一致：

- 三层网络
- 默认 `input -> 96 -> 48 -> score`
- shared backbone 多任务训练
- 纯 numpy 手写训练循环
- mini-batch 梯度下降
- 支持高痛样本过采样和额外 loss weight

但损失函数改成了回归损失：

- `Huber loss`，默认
- `MSE`，可选

默认输出不是直接分类概率，而是先输出 `0-10` pain score 预测值。

### 从分数到概率
这个版本的关键变化是概率定义：

- 网络先输出 `score_pred`
- 再通过平滑函数映射成高痛概率 `p_high`

形式上可以理解为：

```python
p_high = sigmoid((score_pred - pain_threshold) / temperature)
```

默认：

- `pain_threshold = 4.0`
- `temperature = 1.0`

这使得模型在保留原始分数信息的同时，仍然可以输出“属于高痛类的概率分数”。

### 阈值策略
这个版本默认不用 `2` 那套更激进的 `task_adaptive`，而是新增了更保守的：

- `conservative_tune`

它的基本原则是：

- 以 `day_relaxed` 为基线
- 只允许在小范围内调整阈值
- 只有验证集上出现明确收益时才改
- 正样本太少时直接保持基线阈值

默认相关参数：

- `conservative_min_val_pos = 10`
- `conservative_max_shift = 0.05`
- `conservative_min_acc_gain = 0.003`
- `conservative_max_recall_drop = 0.02`
- `conservative_min_f1_delta = 0.0`

这套策略是为了让 `3` 更像“稳步优化的 v1 pain 版”，而不是继续朝 `2` 那种更强阈值自适应方向走。

### 输出指标
这个版本会同时输出两类结果：

- 回归指标
  - `MAE`
  - `RMSE`
  - test 集真实/预测平均分数
- 分类指标
  - `AUC`
  - `accuracy`
  - `precision`
  - `recall`
  - `specificity`
  - `F1`
  - `log loss`
  - `Brier score`

所以它更适合回答两个问题：

- 模型有没有更好地拟合原始 pain score
- 最终高痛分类有没有因此受益

### 默认配置
- 默认 `pain-type`: `both`
- 默认 `feature-mode`: `strong_signal_temporal`
- 默认 `loss-type`: `huber`
- 默认 `huber-delta`: `1.0`
- 默认 `prob-temperature`: `1.0`
- 默认阈值策略：`conservative_tune`
- 默认 day thresholds: `0.55,0.50,0.45,0.40`
- 默认输出目录：`3_output_prediction_report_en_scoreprob/`

### 适合什么时候用
建议在这些场景使用 v3：

- 你不想在训练前丢掉原始 pain score 信息
- 你希望模型先学分数，再转成高痛概率
- 你觉得 `2` 的阈值和提升策略偏激进
- 你希望得到“回归表现 + 分类表现”双重评估

## 模型 3.1：`3_1_pain_logistic.py`
### 核心思路
`3.1` 是在 `3_pain_logistic.py` 基础上的直接增强版，目标很明确：

- 保留原始 pain score 回归这条主线
- 同时增加“高痛/非高痛”的辅助分类损失
- 让模型在学分数的同时，也更关注最终的高痛识别任务

它适合解决 `3` 常见的问题：

- MAE / RMSE 还可以
- 但高痛 recall 偏低
- accuracy 看起来更高，但其实预测过于保守

### 相比 v3 的新增内容
`3.1` 新增了三个关键点：

1. 联合训练目标
   - 主任务：原始 pain score 回归
   - 辅助任务：高痛二分类 BCE loss
2. 可调的辅助分类损失权重
   - `--aux-cls-weight`
3. 训练过程中的 loss 曲线导出
   - `training_loss_all_targets.png`

另外，当前 `3.1` 还加入了面向小样本正类的敏感性增强：

- 稀有正样本 head 会获得额外权重 boost
- 稀有正样本对应样本会获得更高采样概率
- 验证集正样本太少时，会优先尝试更偏 recall 的小幅阈值下调
- 不同术后天数使用不同增强倍率：手术当天 / POD1 更激进，POD2 / POD3 温和增强

### 联合损失
可以把它理解成：

```python
total_loss = regression_loss + aux_cls_weight * auxiliary_classification_loss + l2
```

默认：

- 回归损失：`Huber loss`
- 辅助分类损失：`BCE`
- `aux_cls_weight = 1.0`

### 为什么 3.1 可能比 3 更合理
`3` 的问题在于：

- 它只直接优化“分数拟合”
- 但最终评价依然是“高痛分类”

`3.1` 则把这两件事一起纳入训练目标，所以更有机会在：

- 保留分数信息
- 不完全退化成纯二分类
- 同时把 recall / F1 往回拉

### loss 曲线
`3.1` 会额外导出一张训练 loss 图：

- `training_loss_all_targets.png`

图中包含：

- `Total Loss`
- `Regression Loss`
- `Aux Classification Loss`

这张图主要用于看训练期间：

- 回归主任务是否在下降
- 辅助分类任务是否在真正起作用
- 总 loss 是否稳定

### 默认配置
- 默认 `pain-type`: `both`
- 默认 `feature-mode`: `strong_signal_temporal`
- 默认 `loss-type`: `huber`
- 默认 `huber-delta`: `1.0`
- 默认 `prob-temperature`: `1.0`
- 默认 `aux-cls-weight`: `1.0`
- 默认 `rare-positive-target-count`: `48`
- 默认 `rare-positive-max-boost`: `3.0`
- 默认 `rare-positive-oversample-boost`: `1.5`
- 默认阈值策略：`conservative_tune`
- 默认输出目录：`3_1_output_prediction_report_en_scoreprob_auxcls/`

### 适合什么时候用
建议在这些场景使用 v3.1：

- 你认可 v3 的“先学真实 pain 分数”思路
- 但觉得 v3 的高痛 recall 不够好
- 你希望在回归和分类之间做更平衡的折中
- 你想看训练过程中总 loss、回归 loss、辅助分类 loss 的变化

## 共同训练特征
四个增强版在训练实现上有一些共同点：

- 纯 numpy 手写训练循环，不依赖 torch optimizer
- 手写前向传播、损失和梯度更新
- mini-batch 梯度下降
- 支持正类或高痛样本加权
- 支持高痛样本过采样
- 支持早停
- 支持验证集阈值选择

## 特征模式说明
增强版里最常用的是 `strong_signal_temporal`。

可选模式包括：

- `strict`: 不使用任何 outcome 作为特征
- `all`: 使用除当前 target 外的所有列，可能泄漏
- `temporal`: 使用非 outcome 特征和更早时间点的 outcome
- `strong_signal_temporal`: 实现上和 temporal 同类，强调更早时间点的 outcome 作为强信号

如果目标是尽量避免时间泄漏，优先建议使用：

- `strong_signal_temporal`
- `temporal`

## 常用运行方式
在仓库根目录 `pain_prediction/` 下运行。

运行增强版 v1：

```bash
python baseline_enhance_logistic/1_logistic_regression.py
```

运行增强版 v2：

```bash
python baseline_enhance_logistic/2_train_logistic_regression.py
```

运行增强版 v3：

```bash
python baseline_enhance_logistic/3_pain_logistic.py
```

运行增强版 v3.1：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py
```

只跑 POD3 的静息痛和活动痛：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --day "术后第三天" \
  --pain-type both
```

只跑 POD3 静息痛：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --day "术后第三天" \
  --pain-type rest
```

使用固定阈值：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --threshold-strategy fixed \
  --decision-threshold 0.5
```

使用保守阈值微调：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --threshold-strategy conservative_tune \
  --conservative-max-shift 0.05 \
  --conservative-max-recall-drop 0.02
```

调整分数到概率的映射温度：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --prob-temperature 0.8
```

运行 v3.1，并启用默认辅助分类损失：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --day "术后第三天" \
  --pain-type both
```

调整辅助分类损失权重：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --aux-cls-weight 1.5
```

## 输出文件
其中 `1`、`2`、`3` 会在输出目录中保留四个聚合文件：

- `prediction_overview_all_targets.csv`
- `confusion_matrix_prob_all_targets.csv`
- `confusion_matrix_prob_all_targets.png`
- `training_acc_all_targets.png`

其中 `3_pain_logistic.py` 和 `3_1_pain_logistic.py` 的 summary / confusion CSV 里额外包含：

- `mae`
- `rmse`
- `mean_true_score_test`
- `mean_pred_score_test`

当前目录下已有的旧版示例输出：

- `1_output_prediction_report_en_after_fix/`
- `2_output_prediction_report_en_curvecheck/`

`3` 默认会把结果写到：

- `3_output_prediction_report_en_scoreprob/`

`3.1` 默认会把结果写到：

- `3_1_output_prediction_report_en_scoreprob_auxcls/`

另外，`3.1` 会比 `3` 多输出一张 loss 曲线图：

- `training_loss_all_targets.png`

## 依赖环境
从项目当前的 `pixi.toml` 看，运行这些脚本至少需要：

- Python 3.10
- `numpy`
- `pandas`
- `matplotlib`

项目环境里也安装了 `scikit-learn`，但这些脚本本身不是依赖 sklearn 的封装训练。

## 使用建议
如果只是要一个最稳、最容易解释的对照版本：

- 先跑 baseline

如果你要提升表达能力，同时接受更复杂的训练逻辑：

- 先跑 enhance v1

如果你想更积极地调阈值：

- 再看 enhance v2

如果你想保留真实 pain 分数训练，同时让阈值策略更克制：

- 优先看 enhance v3

## 进一步阅读
基础版说明见：

- `../baseline/README.md`
