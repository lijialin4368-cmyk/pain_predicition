# Distribution Reports

生成数据分布表格和图片：

```bash
pixi run distribution-report
```

直接运行：

```bash
pixi run python reports/distribution/data_analysis/data_distribution_analysis.py \
  --input data/processed/data_vectorized.csv \
  --output-dir reports/distribution/distribution_analysis
```

生成泵配方剂量热图：

```bash
pixi run python reports/distribution/distribution_analysis/plot_pump_formula_dose.py
```

