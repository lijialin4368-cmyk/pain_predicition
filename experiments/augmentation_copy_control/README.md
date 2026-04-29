# Augmentation Copy Control

构建“直接复制样本”的增强对照组，用于区分“样本量增加”与“规则扰动增强”的效果。

先生成规则增强数据：

```bash
pixi run augment-build
```

再生成复制对照：

```bash
pixi run augment-copy-build
```

直接运行：

```bash
pixi run python experiments/augmentation_copy_control/build_dataset.py
```

输出默认保存到 `experiments/augmentation_copy_control/generated/`。

