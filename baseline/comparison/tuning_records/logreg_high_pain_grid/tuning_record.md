# Logistic Regression 高痛样本增强参数调参记录

- 生成时间: `2026-04-01 14:39:54 UTC`
- 运行总耗时: `1268.7s`
- 搜索范围: `--high-pain-oversample-factor` = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
- 搜索范围: `--high-pain-loss-weight` = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
- 总组合数: `102`，成功: `102`，失败: `0`
- 最优模型选择准则: `pain_f1`

## 最优参数

- `--high-pain-oversample-factor = 3.0`
- `--high-pain-loss-weight = 10.0`
- `pain_mean_f1 = 0.2597`
- `pain_mean_auc = 0.6957`
- `mean_f1 = 0.3214`
- `mean_auc = 0.7564`
- `mean_accuracy = 0.9041`
- 输出目录（含2总图2总表）: `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_10.0`

## 调参明细（每行对应一次参数变更）

| run_id | oversample | loss_weight | pain_mean_f1 | pain_mean_auc | mean_f1 | mean_auc | mean_acc | status | output_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 2.5 | 2.0 | 0.2249 | 0.7048 | 0.3075 | 0.7600 | 0.8672 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_2.0` |
| 2 | 2.5 | 2.5 | 0.1900 | 0.7128 | 0.2936 | 0.7632 | 0.9179 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_2.5` |
| 3 | 2.5 | 3.0 | 0.1875 | 0.7114 | 0.2926 | 0.7627 | 0.9239 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_3.0` |
| 4 | 2.5 | 3.5 | 0.2052 | 0.7094 | 0.2997 | 0.7619 | 0.9127 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_3.5` |
| 5 | 2.5 | 4.0 | 0.1857 | 0.7063 | 0.2919 | 0.7606 | 0.9138 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_4.0` |
| 6 | 2.5 | 4.5 | 0.1847 | 0.7052 | 0.2915 | 0.7602 | 0.9134 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_4.5` |
| 7 | 2.5 | 5.0 | 0.1519 | 0.7042 | 0.2783 | 0.7598 | 0.8612 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_5.0` |
| 8 | 2.5 | 5.5 | 0.1500 | 0.7028 | 0.2776 | 0.7592 | 0.8604 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_5.5` |
| 9 | 2.5 | 6.0 | 0.1838 | 0.7017 | 0.2911 | 0.7588 | 0.9131 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_6.0` |
| 10 | 2.5 | 6.5 | 0.1514 | 0.7013 | 0.2781 | 0.7586 | 0.8552 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_6.5` |
| 11 | 2.5 | 7.0 | 0.1469 | 0.7006 | 0.2763 | 0.7584 | 0.8534 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_7.0` |
| 12 | 2.5 | 7.5 | 0.1514 | 0.7004 | 0.2781 | 0.7583 | 0.8519 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_7.5` |
| 13 | 2.5 | 8.0 | 0.1506 | 0.6999 | 0.2778 | 0.7581 | 0.8515 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_8.0` |
| 14 | 2.5 | 8.5 | 0.1628 | 0.6994 | 0.2827 | 0.7579 | 0.8414 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_8.5` |
| 15 | 2.5 | 9.0 | 0.1611 | 0.6964 | 0.2820 | 0.7567 | 0.8373 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_9.0` |
| 16 | 2.5 | 9.5 | 0.1594 | 0.6960 | 0.2813 | 0.7565 | 0.8369 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_9.5` |
| 17 | 2.5 | 10.0 | 0.1608 | 0.6957 | 0.2819 | 0.7564 | 0.8377 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_2.5_lw_10.0` |
| 18 | 3.0 | 2.0 | 0.2010 | 0.7033 | 0.2980 | 0.7594 | 0.9190 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_2.0` |
| 19 | 3.0 | 2.5 | 0.1582 | 0.7027 | 0.2809 | 0.7592 | 0.8701 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_2.5` |
| 20 | 3.0 | 3.0 | 0.1655 | 0.7018 | 0.2838 | 0.7588 | 0.8731 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_3.0` |
| 21 | 3.0 | 3.5 | 0.1563 | 0.7006 | 0.2801 | 0.7584 | 0.8724 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_3.5` |
| 22 | 3.0 | 4.0 | 0.1557 | 0.6995 | 0.2799 | 0.7579 | 0.8724 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_4.0` |
| 23 | 3.0 | 4.5 | 0.1560 | 0.6988 | 0.2800 | 0.7576 | 0.8720 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_4.5` |
| 24 | 3.0 | 5.0 | 0.1515 | 0.6982 | 0.2782 | 0.7574 | 0.8698 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_5.0` |
| 25 | 3.0 | 5.5 | 0.1543 | 0.6979 | 0.2793 | 0.7573 | 0.8709 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_5.5` |
| 26 | 3.0 | 6.0 | 0.1494 | 0.6973 | 0.2773 | 0.7570 | 0.8687 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_6.0` |
| 27 | 3.0 | 6.5 | 0.2519 | 0.7118 | 0.3183 | 0.7628 | 0.9172 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_6.5` |
| 28 | 3.0 | 7.0 | 0.2221 | 0.7068 | 0.3064 | 0.7608 | 0.9160 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_7.0` |
| 29 | 3.0 | 7.5 | 0.2587 | 0.7064 | 0.3211 | 0.7607 | 0.9019 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_7.5` |
| 30 | 3.0 | 8.0 | 0.2529 | 0.7065 | 0.3187 | 0.7607 | 0.9019 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_8.0` |
| 31 | 3.0 | 8.5 | 0.2290 | 0.7036 | 0.3092 | 0.7596 | 0.9108 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_8.5` |
| 32 | 3.0 | 9.0 | 0.2514 | 0.7035 | 0.3181 | 0.7595 | 0.9112 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_9.0` |
| 33 | 3.0 | 9.5 | 0.2529 | 0.7035 | 0.3187 | 0.7595 | 0.9019 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_9.5` |
| 34 | 3.0 | 10.0 | 0.2597 | 0.6957 | 0.3214 | 0.7564 | 0.9041 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.0_lw_10.0` |
| 35 | 3.5 | 2.0 | 0.2005 | 0.7036 | 0.2978 | 0.7595 | 0.9172 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_2.0` |
| 36 | 3.5 | 2.5 | 0.2335 | 0.7037 | 0.3110 | 0.7596 | 0.9175 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_2.5` |
| 37 | 3.5 | 3.0 | 0.1951 | 0.7040 | 0.2956 | 0.7597 | 0.9007 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_3.0` |
| 38 | 3.5 | 3.5 | 0.1515 | 0.7007 | 0.2782 | 0.7584 | 0.8597 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_3.5` |
| 39 | 3.5 | 4.0 | 0.1421 | 0.6916 | 0.2744 | 0.7547 | 0.8511 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_4.0` |
| 40 | 3.5 | 4.5 | 0.1411 | 0.6910 | 0.2740 | 0.7545 | 0.8489 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_4.5` |
| 41 | 3.5 | 5.0 | 0.1361 | 0.6904 | 0.2720 | 0.7543 | 0.8541 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_5.0` |
| 42 | 3.5 | 5.5 | 0.1343 | 0.6902 | 0.2713 | 0.7542 | 0.8522 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_5.5` |
| 43 | 3.5 | 6.0 | 0.1395 | 0.6894 | 0.2734 | 0.7539 | 0.8511 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_6.0` |
| 44 | 3.5 | 6.5 | 0.2409 | 0.6983 | 0.3139 | 0.7574 | 0.9183 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_6.5` |
| 45 | 3.5 | 7.0 | 0.2354 | 0.6981 | 0.3118 | 0.7574 | 0.9160 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_7.0` |
| 46 | 3.5 | 7.5 | 0.2463 | 0.6981 | 0.3161 | 0.7574 | 0.9146 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_7.5` |
| 47 | 3.5 | 8.0 | 0.2420 | 0.6979 | 0.3144 | 0.7573 | 0.9127 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_8.0` |
| 48 | 3.5 | 8.5 | 0.2561 | 0.7013 | 0.3200 | 0.7586 | 0.9030 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_8.5` |
| 49 | 3.5 | 9.0 | 0.2575 | 0.7009 | 0.3206 | 0.7585 | 0.9037 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_9.0` |
| 50 | 3.5 | 9.5 | 0.2540 | 0.7009 | 0.3192 | 0.7585 | 0.9045 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_9.5` |
| 51 | 3.5 | 10.0 | 0.2568 | 0.7006 | 0.3203 | 0.7583 | 0.9034 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_3.5_lw_10.0` |
| 52 | 4.0 | 2.0 | 0.2053 | 0.7030 | 0.2997 | 0.7593 | 0.9138 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_2.0` |
| 53 | 4.0 | 2.5 | 0.1561 | 0.7021 | 0.2800 | 0.7589 | 0.8679 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_2.5` |
| 54 | 4.0 | 3.0 | 0.1471 | 0.6799 | 0.2764 | 0.7501 | 0.8612 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_3.0` |
| 55 | 4.0 | 3.5 | 0.1549 | 0.6935 | 0.2795 | 0.7555 | 0.8657 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_3.5` |
| 56 | 4.0 | 4.0 | 0.1398 | 0.6919 | 0.2735 | 0.7549 | 0.8660 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_4.0` |
| 57 | 4.0 | 4.5 | 0.1474 | 0.6913 | 0.2765 | 0.7546 | 0.8657 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_4.5` |
| 58 | 4.0 | 5.0 | 0.1457 | 0.6973 | 0.2759 | 0.7570 | 0.8507 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_5.0` |
| 59 | 4.0 | 5.5 | 0.1497 | 0.6971 | 0.2775 | 0.7570 | 0.8515 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_5.5` |
| 60 | 4.0 | 6.0 | 0.2419 | 0.6966 | 0.3143 | 0.7567 | 0.9187 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_6.0` |
| 61 | 4.0 | 6.5 | 0.1990 | 0.7009 | 0.2972 | 0.7585 | 0.9183 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_6.5` |
| 62 | 4.0 | 7.0 | 0.1935 | 0.7004 | 0.2950 | 0.7583 | 0.9160 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_7.0` |
| 63 | 4.0 | 7.5 | 0.1495 | 0.7005 | 0.2774 | 0.7583 | 0.8250 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_7.5` |
| 64 | 4.0 | 8.0 | 0.2514 | 0.6979 | 0.3181 | 0.7573 | 0.9112 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_8.0` |
| 65 | 4.0 | 8.5 | 0.1607 | 0.6981 | 0.2819 | 0.7574 | 0.8097 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_8.5` |
| 66 | 4.0 | 9.0 | 0.2574 | 0.7035 | 0.3205 | 0.7595 | 0.9112 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_9.0` |
| 67 | 4.0 | 9.5 | 0.2533 | 0.7033 | 0.3189 | 0.7594 | 0.9015 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_9.5` |
| 68 | 4.0 | 10.0 | 0.2517 | 0.7031 | 0.3183 | 0.7594 | 0.9037 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.0_lw_10.0` |
| 69 | 4.5 | 2.0 | 0.2277 | 0.7106 | 0.3087 | 0.7624 | 0.8672 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_2.0` |
| 70 | 4.5 | 2.5 | 0.1507 | 0.6691 | 0.2779 | 0.7457 | 0.8530 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_2.5` |
| 71 | 4.5 | 3.0 | 0.1522 | 0.6685 | 0.2784 | 0.7455 | 0.8556 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_3.0` |
| 72 | 4.5 | 3.5 | 0.1515 | 0.6897 | 0.2782 | 0.7540 | 0.8597 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_3.5` |
| 73 | 4.5 | 4.0 | 0.1618 | 0.6888 | 0.2823 | 0.7536 | 0.8619 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_4.0` |
| 74 | 4.5 | 4.5 | 0.1393 | 0.6918 | 0.2733 | 0.7548 | 0.8575 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_4.5` |
| 75 | 4.5 | 5.0 | 0.1428 | 0.6898 | 0.2747 | 0.7540 | 0.8642 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_5.0` |
| 76 | 4.5 | 5.5 | 0.1476 | 0.6902 | 0.2766 | 0.7542 | 0.8526 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_5.5` |
| 77 | 4.5 | 6.0 | 0.1479 | 0.6895 | 0.2767 | 0.7539 | 0.8530 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_6.0` |
| 78 | 4.5 | 6.5 | 0.1514 | 0.6892 | 0.2781 | 0.7538 | 0.8552 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_6.5` |
| 79 | 4.5 | 7.0 | 0.2457 | 0.6963 | 0.3158 | 0.7566 | 0.9172 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_7.0` |
| 80 | 4.5 | 7.5 | 0.1397 | 0.6961 | 0.2735 | 0.7566 | 0.8239 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_7.5` |
| 81 | 4.5 | 8.0 | 0.1591 | 0.6961 | 0.2812 | 0.7565 | 0.8116 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_8.0` |
| 82 | 4.5 | 8.5 | 0.1544 | 0.6957 | 0.2793 | 0.7564 | 0.8216 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_8.5` |
| 83 | 4.5 | 9.0 | 0.1538 | 0.6974 | 0.2791 | 0.7571 | 0.8049 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_9.0` |
| 84 | 4.5 | 9.5 | 0.1508 | 0.6971 | 0.2779 | 0.7570 | 0.8052 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_9.5` |
| 85 | 4.5 | 10.0 | 0.1484 | 0.6968 | 0.2769 | 0.7568 | 0.8075 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_4.5_lw_10.0` |
| 86 | 5.0 | 2.0 | 0.2351 | 0.7059 | 0.3116 | 0.7605 | 0.8675 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_2.0` |
| 87 | 5.0 | 2.5 | 0.1532 | 0.6804 | 0.2789 | 0.7503 | 0.8690 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_2.5` |
| 88 | 5.0 | 3.0 | 0.1540 | 0.6799 | 0.2792 | 0.7501 | 0.8713 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_3.0` |
| 89 | 5.0 | 3.5 | 0.1463 | 0.6904 | 0.2761 | 0.7543 | 0.8552 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_3.5` |
| 90 | 5.0 | 4.0 | 0.1381 | 0.6915 | 0.2728 | 0.7547 | 0.8537 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_4.0` |
| 91 | 5.0 | 4.5 | 0.1466 | 0.6911 | 0.2762 | 0.7546 | 0.8556 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_4.5` |
| 92 | 5.0 | 5.0 | 0.1512 | 0.6958 | 0.2780 | 0.7564 | 0.8474 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_5.0` |
| 93 | 5.0 | 5.5 | 0.1512 | 0.6958 | 0.2780 | 0.7564 | 0.8474 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_5.5` |
| 94 | 5.0 | 6.0 | 0.1442 | 0.6958 | 0.2753 | 0.7564 | 0.8481 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_6.0` |
| 95 | 5.0 | 6.5 | 0.1389 | 0.6957 | 0.2731 | 0.7564 | 0.8459 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_6.5` |
| 96 | 5.0 | 7.0 | 0.1454 | 0.6953 | 0.2757 | 0.7562 | 0.8448 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_7.0` |
| 97 | 5.0 | 7.5 | 0.1415 | 0.6952 | 0.2742 | 0.7562 | 0.8280 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_7.5` |
| 98 | 5.0 | 8.0 | 0.1559 | 0.6955 | 0.2799 | 0.7563 | 0.8231 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_8.0` |
| 99 | 5.0 | 8.5 | 0.1522 | 0.7039 | 0.2785 | 0.7597 | 0.8075 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_8.5` |
| 100 | 5.0 | 9.0 | 0.1549 | 0.7036 | 0.2795 | 0.7596 | 0.8101 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_9.0` |
| 101 | 5.0 | 9.5 | 0.1528 | 0.7030 | 0.2787 | 0.7593 | 0.8104 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_9.5` |
| 102 | 5.0 | 10.0 | 0.1574 | 0.7016 | 0.2805 | 0.7587 | 0.8067 | ok | `/home/maolab-data/disk1/jlli/pain_prediction/baseline/tuning_records/logreg_high_pain_grid/runs/os_5.0_lw_10.0` |
