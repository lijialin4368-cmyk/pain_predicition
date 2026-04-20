# Enhanced Logistic README

## 目录说明
这个目录现在包含四个增强版 pain logistic 脚本：

- `1_logistic_regression.py`
- `2_train_logistic_regression.py`
- `3_pain_logistic.py`
- `3_1_pain_logistic.py`

它们都服务于同一个大方向：围绕术后结局做疼痛风险预测，但不同版本在建模目标、阈值策略、是否使用增强数据、以及对高痛样本的处理方式上逐步增强。

和兄弟目录 `baseline/` 中的基础版一起看，目前可以理解为五套主要方案：

- baseline: 线性 logistic regression
- enhance v1: 三层非线性网络 + shared backbone + focal loss / day-relaxed threshold
- enhance v2: 在 v1 基础上进一步加入 task-adaptive threshold
- enhance v3: 疼痛分数回归 + 概率映射 + conservative threshold tuning
- enhance v3.1: 在 v3 基础上加入 auxiliary high-pain classification loss，并支持增强数据平衡采样

## 共同任务定义
增强版模型的核心目标，是围绕术后多天的多个结局做预测。不同脚本对标签的处理方式不完全相同：

- `1` 和 `2` 主要直接围绕二分类标签训练
- `3` 和 `3.1` 先回归原始疼痛分数，再把预测分数映射为高痛概率，最后通过阈值转成分类

支持的结局日：

- `手术当天`
- `术后第一天`
- `术后第二天`
- `术后第三天`

增强版整体支持的结局指标：

- `静息痛`
- `活动痛`
- `镇静评分`
- `活动状态`
- `恶心呕吐`

其中：

- `1` 和 `2` 面向更完整的多任务结局集合
- `3` 和 `3.1` 目前只做 `静息痛` 与 `活动痛` 两类 pain-only 任务

统一的高痛标签规则：

- `静息痛` / `活动痛`: `0-3 -> class 0`, `>=4 -> class 1`
- `镇静评分`: `1-2 -> class 0`, `3-5 -> class 1`
- `活动状态`: `1-2 -> class 0`, `3-4 -> class 1`
- `恶心呕吐`: `0-1 -> class 0`, `2-3 -> class 1`

对 `3` 和 `3.1` 来说，这个规则主要用于：

- 从原始疼痛分数导出高痛二分类标签
- 把回归得到的疼痛分数转换为高痛概率
- 在验证集上选择最终 decision threshold

## 模型关系

| 模型 | 入口脚本 | 核心结构 | 默认阈值策略 | 默认输入 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| Baseline | `baseline/final_baseline_logistic_re/train_logistic_regression.py` | 单层线性 logistic regression | `tune` | `data_vectorized.csv` | 做基础对照、保留较强可解释性 |
| Enhance v1 | `baseline_enhance_logistic/1_logistic_regression.py` | 三层网络，支持 shared backbone multitask | `day_relaxed` | `data_vectorized.csv` | 希望引入非线性表达和多任务共享 |
| Enhance v2 | `baseline_enhance_logistic/2_train_logistic_regression.py` | 三层网络，支持 shared backbone multitask | `task_adaptive` | `data_vectorized.csv` | 希望进一步优化各任务阈值 |
| Enhance v3 | `baseline_enhance_logistic/3_pain_logistic.py` | 三层网络，直接回归 pain score，再映射为高痛概率 | `conservative_tune` | `data_vectorized.csv` | 关注疼痛分数本身，并希望分类阈值更稳健 |
| Enhance v3.1 | `baseline_enhance_logistic/3_1_pain_logistic.py` | v3 + auxiliary high-pain classification loss + rare-positive boost | `conservative_tune` | 优先 `data_augmentation/generated/augmented_dataset.csv` | 希望把增强数据、稀有正类和高痛召回一起纳入训练 |

## 模型 1：`1_logistic_regression.py`
### 核心思路
这个版本是在 baseline 基础上的第一版增强：

- 从线性 logistic regression 升级为三层前馈网络
- 增加 `ReLU` 隐层，默认结构为 `input -> 96 -> 48 -> sigmoid`
- 支持 `focal loss` 和 `BCE`
- 支持多任务 shared backbone 训练
- 支持按天放宽阈值的 `day_relaxed` 策略
- 支持高痛样本权重增强和临床高痛额外损失惩罚

### 结构特点
这个脚本同时保留了：

- 单任务训练接口 `run_one_target`
- 多任务 shared-backbone 主流程 `run_shared_backbone_targets`

但在 `main()` 中，默认走的是 shared backbone 多任务训练流程。

另外有一个重要实现细节：

- `SHARED_BACKBONE_METRICS = {"静息痛", "活动痛", "镇静评分", "恶心呕吐"}`
- 也就是说，虽然总指标列表里有 `活动状态`，但 shared-backbone 主流程里默认不会训练它

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
这个版本把问题改成了“先回归疼痛分数，再做高痛分类”。

整体流程是：

1. 只针对 pain-only 目标训练，即 `静息痛` 和 `活动痛`
2. 模型先输出连续疼痛分数
3. 再用 `score_to_probability()` 把预测分数映射成“高痛概率”
4. 最后通过阈值得到 `0/1` 分类结果

和前两个版本相比，v3 不再把训练目标直接固定成 hard class，而是显式保留分数尺度信息。

### 默认训练策略
- 默认任务范围：四天的 `静息痛 + 活动痛`
- 默认 `feature-mode`: `strong_signal_temporal`
- 默认回归损失：`huber`
- 默认隐藏层：`96 / 48`
- 默认高痛阈值：`pain-threshold=4.0`
- 默认高痛强化阈值：`high-pain-score-threshold=3.0`
- 默认阈值策略：`conservative_tune`
- 默认输出目录：`3_output_prediction_report_en_scoreprob/`

### 高痛强化方式
v3 在训练阶段对高痛样本做了双重增强：

- 对高痛样本做额外 oversampling
- 对高痛样本加更大的 loss weight

相关默认参数：

- `high-pain-oversample-factor=2.5`
- `high-pain-loss-weight=2.0`
- `positive-weight-mode=balanced`

### 阈值策略
v3 支持：

- `fixed`
- `tune`
- `day_relaxed`
- `conservative_tune`

默认的 `conservative_tune` 是在 `day_relaxed` 基线附近做小范围验证集微调，不追求激进改阈值，而是加上这些保护条件：

- 验证集正样本太少时，直接保留 day-relaxed 基线
- 限制阈值偏移范围
- 要求 accuracy 至少达到最小提升
- 限制 recall 下滑
- 限制 F1 明显下降

这套设计比单纯 `tune` 更稳，尤其适合小样本或某些天数高痛样本很少的情况。

### 输入与切分
默认从以下路径顺序寻找输入：

- `pain_prediction/data_vectorized.csv`
- `baseline_enhance_logistic/data_vectorized.csv`
- 当前脚本目录下的 `data_vectorized.csv`

数据切分上，v3 采用普通 train/test 划分，不依赖增强数据元信息。

### 输出文件
v3 会在输出目录中汇总保留：

- `prediction_overview_all_targets.csv`
- `confusion_matrix_prob_all_targets.csv`
- `confusion_matrix_prob_all_targets.png`
- `training_acc_all_targets.png`

### 适合什么时候用
建议在这些场景使用 v3：

- 你觉得疼痛分数本身比硬分类标签更值得保留
- 你希望高痛分类结果能来自连续 score 的平滑映射
- 你希望阈值调整更稳，而不是只追求某个验证指标极值

## 模型 3.1：`3_1_pain_logistic.py`
### 核心思路
v3.1 是当前 pain-only 路线里更完整的一版，在 v3 基础上新增了三类强化：

- 加入 `auxiliary high-pain classification loss`
- 对 rare positive head 做额外 boost
- 对增强数据启用平衡采样训练

也就是说，v3.1 仍然保留“回归分数 -> 映射概率 -> 决策阈值”的主链路，但训练时不只看回归误差，还显式让模型学习“是否高痛”这件事。

### 默认训练策略
- 默认任务范围：四天的 `静息痛 + 活动痛`
- 默认 `feature-mode`: `strong_signal_temporal`
- 默认回归损失：`huber`
- 默认 `aux-cls-weight=1.5`
- 默认阈值策略：`conservative_tune`
- 默认 `conservative-selection-scope=per_target`
- 默认 `conservative-min-accuracy=0.60`
- 默认 `conservative-accuracy-upshift=0.35`
- 默认输出目录：`3_1_output_prediction_report_en_scoreprob_auxcls/`
- 默认输入优先级：
  - `data_augmentation/generated/augmented_dataset.csv`
  - `data_vectorized.csv`
  - 当前脚本目录下的 `data_vectorized.csv`

### 相比 v3 多出来的关键机制
#### 1. Auxiliary classification loss
v3.1 在训练时同时优化两部分：

- 原始疼痛分数回归损失
- 高痛二分类 BCE 辅助损失

辅助损失会通过 `aux-cls-weight` 进入总 loss，用来拉近模型对高痛边界的感知。

#### 2. Rare-positive boost
对于阳性特别少的 target head，v3.1 会根据正样本稀缺程度额外提高该 head 的权重和采样强度，避免模型过度偏向多数类。

默认相关参数：

- `rare-positive-target-count=80`
- `rare-positive-boost-power=0.5`
- `rare-positive-max-boost=3.0`
- `rare-positive-oversample-boost=2.0`

#### 3. Day sensitivity scale
不同术后日期的高痛敏感度不同，v3.1 内置了按天的 sensitivity scale：

- `手术当天: 1.35`
- `术后第一天: 1.25`
- `术后第二天: 1.12`
- `术后第三天: 1.08`

这个 scale 会参与辅助分类损失的加权，让更关键的早期高痛头获得更高训练关注度。

### 增强数据采样策略
如果输入数据中带有增强元信息列：

- `__meta_is_generated`
- `__meta_dataset_split`
- `__meta_source_row_id`

那么 v3.1 会启用当前项目里约定的增强数据训练规则：

- test set 优先直接使用 `__meta_dataset_split == test` 的原始样本
- 如果没有显式 test 标记，也会强制从原始数据里切 test
- test set 不使用生成样本
- train pool 中保留所有“从未被增强过的原始样本”
- 再从“被增强过的原始样本 + 新增生成样本”组成的 augmented pool 中，随机抽取与前者等量的数据
- 然后再从这个平衡后的 train pool 中切 validation

这就是之前已经在代码里落地的“原始未增强样本全保留，增强侧等量抽样”的训练模式。

### 阈值策略
v3.1 与 v3 一样支持：

- `fixed`
- `tune`
- `day_relaxed`
- `conservative_tune`

但 v3.1 在 `conservative_tune` 下额外补了一个低正样本保护分支：

- 当验证集阳性特别少时，只探索向下的小范围阈值移动
- 同时限制 FPR 上升幅度

并且当前 3.1 的默认优化方向已经进一步调整为：

- 默认走 `per_target` 严格模式，每个 target 单独调阈值
- 每个 target 的验证集 `accuracy` 都优先要求不低于 `0.60`
- 在满足这个单目标 accuracy 下限后，再尽量提高高痛样本的 recall
- 如果 day-relaxed 阈值不够保守，会继续向上搜索更高阈值
- 对低正样本 target 仍然保留更谨慎的搜索方式

这使得 3.1 更偏向“每个目标先守住基本准确率，再尽量多召回高痛样本”。

### 训练强化默认值
为了更主动地抓高痛样本，3.1 当前默认把训练强化也同步加重了：

- `high-pain-oversample-factor=4.0`
- `high-pain-loss-weight=3.0`
- `aux-cls-weight=1.5`
- `rare-positive-target-count=80`
- `rare-positive-oversample-boost=2.0`

这一组默认值对应的就是目前采用的 “方案 B”。

### 低痛保护阈值
为了避免 `手术当天_活动痛` 和 `术后第一天_活动痛` 在高召回配置下把过多低痛样本误判成高痛，当前 3.1 还额外加了一个 target-specific low-pain guard：

- `手术当天_活动痛`: 先优先搜索能把验证集 `specificity` 控到至少 `0.35` 的更高阈值
- `术后第一天_活动痛`: 同样优先搜索验证集 `specificity >= 0.35` 的更高阈值

如果在给定搜索范围内达不到这个标准，就退回到“尽量提高 specificity 和 accuracy”的 best-effort 阈值。

这层保护只对这两个活动痛头生效，目的是把“低痛样本被错抓成高痛”的问题先压下来。

### 输出文件
v3.1 会在输出目录中汇总保留：

- `prediction_overview_all_targets.csv`
- `confusion_matrix_prob_all_targets.csv`
- `confusion_matrix_prob_all_targets.png`
- `training_acc_all_targets.png`
- `training_loss_all_targets.png`

相较 v3，多了一个总训练损失图。

### 适合什么时候用
建议在这些场景使用 v3.1：

- 你已经开始使用增强数据训练
- 你希望 test set 严格保持原始数据、且不混入生成样本
- 你更关心高痛识别能力，尤其是稀有正类任务
- 你希望在 score regression 之外，再给模型一个显式的 high-pain 分类目标

## 共同训练特征
这四个增强版在训练实现上有一些共同点：

- 纯 numpy 手写训练循环，不依赖 torch optimizer
- 手写前向传播、损失和梯度更新
- mini-batch 梯度下降
- 支持早停
- 支持验证集阈值选择

其中：

- `1` / `2` 更偏向直接分类
- `3` / `3.1` 更偏向分数回归后再映射概率

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
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --day "术后第三天" \
  --pain-type both
```

显式指定原始数据跑 v3.1：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --input data_vectorized.csv
```

使用增强数据跑 v3.1：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --input data_augmentation/generated/augmented_dataset.csv
```

指定固定阈值：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --threshold-strategy fixed \
  --decision-threshold 0.5
```

使用 day-relaxed 阈值：

```bash
python baseline_enhance_logistic/3_pain_logistic.py \
  --threshold-strategy day_relaxed \
  --day-thresholds 0.55,0.50,0.45,0.40
```

显式使用 conservative_tune：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --threshold-strategy conservative_tune \
  --conservative-selection-scope per_target \
  --conservative-min-accuracy 0.60 \
  --conservative-accuracy-upshift 0.35 \
  --conservative-max-shift 0.05
```

调整辅助分类损失权重：

```bash
python baseline_enhance_logistic/3_1_pain_logistic.py \
  --aux-cls-weight 1.5
```

## 输出文件说明
增强版最终都会在输出目录中保留聚合结果文件，但不同版本略有差异：

- `1` / `2`:
  - `prediction_overview_all_targets.csv`
  - `confusion_matrix_prob_all_targets.csv`
  - `confusion_matrix_prob_all_targets.png`
  - `training_acc_all_targets.png`
- `3`:
  - `prediction_overview_all_targets.csv`
  - `confusion_matrix_prob_all_targets.csv`
  - `confusion_matrix_prob_all_targets.png`
  - `training_acc_all_targets.png`
- `3.1`:
  - `prediction_overview_all_targets.csv`
  - `confusion_matrix_prob_all_targets.csv`
  - `confusion_matrix_prob_all_targets.png`
  - `training_acc_all_targets.png`
  - `training_loss_all_targets.png`

当前目录下已有的典型输出示例：

- `1_output_prediction_report_en_after_fix/`
- `2_output_prediction_report_en_curvecheck/`
- `3_output_prediction_report_en_scoreprob/`
- `3_1_output_prediction_report_en_scoreprob_auxcls/`

## 依赖环境
从项目当前的 `pixi.toml` 看，运行这些脚本至少需要：

- Python 3.10
- `numpy`
- `pandas`
- `matplotlib`

项目环境里也安装了 `scikit-learn`，但这些增强版脚本主要仍是手写训练流程，不是 sklearn 的封装训练。

## 使用建议
如果只是要一个最稳、最容易解释的对照版本：

- 先跑 baseline

如果你要提升表达能力，同时接受更复杂的训练逻辑：

- 先跑 enhance v1

如果你已经确认概率输出合理，想把阈值策略继续打磨：

- 再跑 enhance v2

如果你想显式利用疼痛分数尺度，而不是直接硬分类：

- 优先跑 enhance v3

如果你已经引入数据增强，希望训练采样规则与当前增强工程保持一致：

- 优先跑 enhance v3.1

## 文档维护约定
这份 README 现在作为 `baseline_enhance_logistic/` 的总入口说明。

后续如果 `3` 或 `3.1` 的默认训练策略、增强采样规则、输出目录或运行命令继续变化，应同步更新这里，避免 README 和脚本实现脱节。

## 进一步阅读
基础版说明见：

- `../baseline/README.md`
