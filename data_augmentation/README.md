# 数据增强工程

这个目录专门负责在 `pain_prediction` 内生成“固定原始保留集 + 增强训练集”的新数据，不改动原始 `data_vectorized.csv`。

## 目录说明

- `config.py`: 增强规则、列名、随机种子和输出路径。
- `augment_dataset.py`: 主脚本。
- `generated/`: 生成的新数据集与统计汇总。

## 新增优化

- 生成样本支持自动去重，避免完全相同的增强样本重复进入训练。
- 支持通过 `--generated-keep-fraction` 在生成后按比例保留增强样本。
- 原始样本与固定保留池不受这个比例影响。

## 当前默认规则

1. 先从原始 2689 条样本中按固定随机种子切出 `30%` 作为 validation/test 的原始保留池。
2. 剩余 `70%` 原始样本作为 training set，并且只对这部分训练原始样本做增强。
3. 保留池中的样本不做任何增强，避免验证/测试阶段发生数据泄漏。
4. 性别规则按 `data_cleaning.py` 的编码解释:
   - `性别_num = 1` 表示男性
   - `性别_num = 0` 表示女性
5. 高痛抬升规则:
   - 若样本满足“男性”或“年龄 > 60 岁”
   - 且“手术当天静息痛 / 活动痛任一 >= 3”
   - 则把 4 天的静息痛和活动痛整体上移 1 分
   - 再围绕“原始样本”和“抬升样本”分别对 `年龄/体重` 做 `[-2, -1, 0, 1, 2]` 扰动
   - 这样共有 `25 + 25 - 1 = 49` 个新候选样本，其中减去的是已经存在的原始样本本身
   - 每个父样本从这 49 个候选里随机保留 10 个
6. 四天波动规则:
   - 对静息痛 4 天序列单独判断一次
   - 对活动痛 4 天序列单独判断一次
   - 如果同一序列里有至少 2 个值 `> 4`，就分别生成 `+1` 和 `-1` 两个波动样本
   - `-1` 时只做下界裁切到 `0`，不做上界裁切

## 运行方式

在项目根目录下:

```bash
pixi run augment-data
```

如果你想直接控制增强样本保留比例，例如只保留 50%:

```bash
cd pain_prediction
pixi run python3 data_augmentation/augment_dataset.py --generated-keep-fraction 0.5
```

## 输出文件

- `generated/augmented_dataset.csv`: 合并后的完整数据，可直接传给随机森林脚本。
- `generated/train_original.csv`: 增强前的原始训练子集（默认约 `70%`）。
- `generated/test_original.csv`: 固定保留的原始保留池（默认约 `30%`，供后续 validation/test 再划分）。
- `generated/generated_only.csv`: 只包含新生成样本。
- `generated/augmentation_summary.json`: 规则与样本数汇总。
- `generated/augmentation_rule_counts.csv`: 每类增强规则生成的条数。

其中 `augmentation_summary.json` 现在会额外记录:

- 原始生成样本数
- 去重后的生成样本数
- 最终保留比例

## 和随机森林脚本的衔接

`randomforest/train_random_forest_regression.py` 已经支持识别以下 metadata 列:

- `__meta_is_generated`
- `__meta_dataset_split`

因此可以直接这样训练:

```bash
cd pain_prediction
pixi run python randomforest/train_random_forest_regression.py \
  --data-path data_augmentation/generated/augmented_dataset.csv
```
