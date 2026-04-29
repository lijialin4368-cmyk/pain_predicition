# Distribution Analysis Output

本目录保存数据分布分析生成的表格和图片。

重新生成完整分布报告：

```bash
pixi run distribution-report
```

重新生成泵配方剂量热图：

```bash
pixi run python reports/distribution/distribution_analysis/plot_pump_formula_dose.py
```

- Input: `data/processed/data_vectorized.csv`
- Total rows: 2689
- All chart titles/axes are in English.
- All bar charts are annotated with count and percentage.
- Age/Weight histograms are annotated with bin counts.
- Grouped charts include: surgery type, anesthesia method, analgesia mode, block site, block drug presence.
- Pump formula dose detail table: `frequency_tables/pump_formula_drug_dose_detail.csv`.
- Pump formula wide table (dose as columns): `frequency_tables/pump_formula_drug_dose_wide.csv`.
- Excluded from plotting: `镇痛泵配方_has_*`, `手术星期`.
- Surgery month is plotted in fixed order 1..12.
- Outcomes are split by day (Surgery Day/POD1/POD2/POD3) with trend charts per metric.
- Frequency tables: `reports/distribution/distribution_analysis/frequency_tables`
- Plots: `reports/distribution/distribution_analysis/plots`
- Histogram tables: `reports/distribution/distribution_analysis/histogram_tables`
- Matplotlib enabled: True