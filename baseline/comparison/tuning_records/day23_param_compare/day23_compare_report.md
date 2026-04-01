# 第2/3天参数对比报告

对比参数：
- A: `--high-pain-oversample-factor 3.0 --high-pain-loss-weight 10.0`
- B: `--high-pain-oversample-factor 2.5 --high-pain-loss-weight 2.0`

## 分天结果

| day | config | pain_mean_f1 | pain_mean_auc | mean_f1 | mean_auc | mean_accuracy | output_dir |
|---|---|---:|---:|---:|---:|---:|---|
| 术后第三天 | os_2.5_lw_2.0 | 0.0198 | 0.6172 | 0.1630 | 0.7496 | 0.9103 | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/day23_param_compare/os_2.5_lw_2.0/day3` |
| 术后第三天 | os_3.0_lw_10.0 | 0.0252 | 0.6082 | 0.1652 | 0.7461 | 0.9012 | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/day23_param_compare/os_3.0_lw_10.0/day3` |
| 术后第二天 | os_2.5_lw_2.0 | 0.0526 | 0.5882 | 0.2247 | 0.6571 | 0.9112 | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/day23_param_compare/os_2.5_lw_2.0/day2` |
| 术后第二天 | os_3.0_lw_10.0 | 0.0331 | 0.6112 | 0.2169 | 0.6663 | 0.8740 | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/day23_param_compare/os_3.0_lw_10.0/day2` |

## 第2/3天合并(平均)

| config | pain_mean_f1 | pain_mean_auc | mean_f1 | mean_auc | mean_accuracy |
|---|---:|---:|---:|---:|---:|
| os_2.5_lw_2.0 | 0.0362 | 0.6027 | 0.1939 | 0.7034 | 0.9107 |
| os_3.0_lw_10.0 | 0.0291 | 0.6097 | 0.1910 | 0.7062 | 0.8876 |

## 结论

按 `pain_mean_f1 -> pain_mean_auc -> mean_f1` 排序，较优参数是 `os_2.5_lw_2.0`（oversample=2.5, loss_weight=2.0）。
