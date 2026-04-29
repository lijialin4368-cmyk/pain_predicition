# Data

```text
raw/data.csv                    原始表
processed/data_vectorized.csv   清洗和向量化后的建模表
```

重新生成向量化数据：

```bash
pixi run clean-data
```

`data/processed/data_vectorized.csv` 是后续 split、baseline、random forest、XGBoost 和增强数据实验的默认输入。

