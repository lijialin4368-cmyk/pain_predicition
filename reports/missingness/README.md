# Missingness Report

缺失率分析输入：

```text
data/raw/data.csv
data/processed/data_vectorized.csv
```

运行：

```bash
pixi run missingness-report
```

直接运行：

```bash
pixi run python reports/missingness/run_missingness_analysis.py
```

输出：

```text
covariates_missingness.png
outcomes_missingness.png
covariates_missingness.csv
outcomes_missingness.csv
```