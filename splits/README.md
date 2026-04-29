# Splits

本目录保存固定 reference split。当前默认比例是：

```text
train / validation / test = 80 / 10 / 10
```

生成命令：

```bash
pixi run build-reference-splits
```

主要文件：

```text
build_reference_splits.py        生成 reference split
split_utils.py                   共享切分和目标分层工具
reference_splits_seed_42.csv     每个目标的 row_id -> split 映射
reference_splits_summary.csv     每个目标的样本量和阳性数汇总
```

`target_tier` 用于标记结果解释强度：

```text
main                         主分析目标
exploratory                  探索性目标
low_event                    极少阳性目标
low_event_sedation           镇静评分极少阳性目标
secondary_activity_status    活动状态辅助目标
```

