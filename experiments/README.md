# Experiments

实验目录：

```text
augmentation/              规则增强数据
augmentation_copy_control/ 直接复制对照数据
borderline_smote/          BorderlineSMOTE 训练集增强
generative_tabular/        CTGAN / TVAE 生成式增强
high_pain_augmentation/    高痛增强参数调参
```

建议运行顺序：

```bash
pixi run augment-build
pixi run augment-copy-build
pixi run smote-build
pixi run gen-tvae-cls-build
pixi run gen-tvae-reg-build
pixi run rf-train-aug
pixi run rf-train-copy
pixi run logistic-train-smote
pixi run logistic-train-tvae
pixi run mlp-train-tvae
pixi run rf-train-tvae
pixi run xgb-train-tvae
```
