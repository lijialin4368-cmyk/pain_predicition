# Generative Tabular

基于 `CTGAN` / `TVAE` 的表格生成式增强实验。该实验家族同时支持：

- `classification`：面向高痛/高风险二分类任务，默认只补充训练集中的阳性 synthetic rows
- `regression`：面向连续结局回归任务，默认补充整体验证前的训练分布

运行：

```bash
pixi run gen-ctgan-cls-build
pixi run gen-tvae-cls-build
pixi run gen-ctgan-reg-build
pixi run gen-tvae-reg-build
```

输出目录：

```text
generated/ctgan/classification/
generated/tvae/classification/
generated/ctgan/regression/
generated/tvae/regression/
```

每个输出目录会包含：

```text
augmented_dataset.csv
generated_only.csv
train_original.csv
validation_original.csv
test_original.csv
generation_summary.json
metadata.json
synthesizer.pkl
```

当前约定：

- 生成器只在 `train split` 上拟合。
- `classification` 模式默认在训练集正类子集上拟合生成器，再补充正类 synthetic rows。
- `validation/test` 始终保留真实原始样本。
- 分类增强默认面向 `target >= threshold` 的正类补充。
- 回归增强默认保留连续目标值，不做阈值化。

下游训练入口：

```bash
pixi run logistic-train-ctgan
pixi run logistic-train-tvae
pixi run mlp-train-ctgan
pixi run mlp-train-tvae
pixi run rf-train-ctgan
pixi run rf-train-tvae
pixi run xgb-train-ctgan
pixi run xgb-train-tvae
```
