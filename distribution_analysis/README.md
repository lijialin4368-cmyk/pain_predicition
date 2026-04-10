# Distribution Analysis Output

- Input: `/home/maolab-data/disk1/jlli/pain_prediction/data_vectorized.csv`
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
- Frequency tables: `/home/maolab-data/disk1/jlli/pain_prediction/distribution_analysis/frequency_tables`
- Plots: `/home/maolab-data/disk1/jlli/pain_prediction/distribution_analysis/plots`
- Histogram tables: `/home/maolab-data/disk1/jlli/pain_prediction/distribution_analysis/histogram_tables`
- Matplotlib enabled: True