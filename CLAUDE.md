# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

STN-CLIP 在冻结的 CLIP 视觉编码器之上训练空间变换网络（STN），用于细粒度图像分类。共享的定位网络从 CLIP ViT 的 patch token 中预测 N 个裁剪位置。每个裁剪区域重新输入 CLIP，融合模块将 N 个视角特征与原始全局 CLS 特征合并。融合后的表示通过余弦相似度 + 交叉熵损失与预计算的文本特征进行匹配。

存在两种训练范式：
- **有监督**（配置文件在 `STN-Config/`，脚本：`train_ddp_stn.py`、`train_multi_view_stn.py`）
- **无监督**（配置文件在 `UN-STN-Config/`，脚本：`train_unsupervised_ddp.py`）—— 使用两阶段 KL 一致性目标：预热阶段将局部/融合预测对齐到冻结的全局 CLIP logits，然后周期性目标阶段将局部视角对齐到定期刷新的教师模型。

## 入口点

```
# 单卡有监督训练
python main_stn.py --dataset_name=cub --stn_config=cub --num_workers=8 --seed=42

# 多卡有监督训练（DDP）
torchrun --nproc_per_node=2 train_ddp_stn.py --dataset cub --stn_config cub --num_workers 8 --seed 42

# 多卡无监督训练（DDP）
torchrun --nproc_per_node=2 train_unsupervised_ddp.py --dataset cub --config cub --num_workers 8 --seed 42

# 断点续训（用 nohup 包装 train_ddp_stn.py）
bash resume_train.sh

# 测试有监督模型
python test_multi_stn.py --dataset_name cub

# 测试无监督模型
python test_unsupervised_stn.py --dataset_name cub --visual_batches 3 --max_vis_samples 8
```

## 架构

### 模型（`stn/multi_view_stn.py`）

- **`MultiViewSTNModel`** — 顶层模块。前向传播流程：(1) 将 448→224 下采样作为 CLIP 输入，(2) `SharedLocalizationNetwork` 从 ViT patch token 预测 N 个视角中心位置，(3) `get_transformation_matrices` 构建仿射矩阵（固定缩放 0.5，可变平移），(4) `MultiViewSTNProcessor` 在 448 分辨率下对每个视角执行 `grid_sample` 并通过 CLIP 编码每个裁剪区域，(5) 在最前面拼接原始全局 CLS 特征，(6) `fusion_module` 将 N+1 个视角合并为一个特征向量。训练模式下返回 `(fused_features, view_features)`。

- **`SharedLocalizationNetwork`** — 1×1 卷积降低 patch 维度 → 空间卷积 → 自适应池化到 5×5 → 线性层→128→ReLU→dropout→线性层→2*N。位置参数经过 `tanh`（范围 [-1,1]）。初始化为裁剪块平铺整个图像平面（网格布局）。

- **融合模块**：`SimpleFusion`（简单平均）、`WeightedFusion`（相似度加权 + 门控）、`ConcatFusion`（拼接→MLP→归一化）、`TransformerFusion`（CLS Token + 位置编码 + Transformer 编码器）。

- CLIP 参数被冻结；只训练 `localization_network` 和 `fusion_module`。

### 损失函数（`stn/loss_multi.py`）

**`MultiViewSTNLoss`** 组合最多 5 个损失项（权重=0 表示禁用）：
1. `ClassificationLoss` — 对融合相似度做交叉熵
2. `FeatureDecorrelationLoss` — 视角间余弦相似度矩阵与单位矩阵的 MSE（鼓励视角正交）
3. `AdaptiveClassificationLoss` — 逐视角交叉熵，用 `softmax((1-confidence)/penalty_temp)` 加权，使表现差的视角获得更大权重
4. `KLConsistencyLoss` — KL(fused || view)，使用不对称温度（fused=0.05 硬目标，views=0.1 软目标）
5. `FairnessRegularizationLoss` — 最大化批次平均预测的熵（防止坍缩）

无监督训练（`train_unsupervised_ddp.py`）有自己独立的 `compute_two_stage_unsupervised_loss`，绕过 `MultiViewSTNLoss.forward()`，直接计算 KL 损失：
- **阶段一（预热）**：KL(global || local) + KL(global || fused)，其中 global 来自冻结 CLIP 对 224 图像的编码
- **阶段二（周期性目标）**：每 N 个 epoch 刷新一个教师 MultiViewSTNModel；其融合预测作为 KL(teacher || local_views) 的目标

### 数据流水线（`data_preprocess.py`）

`MultiViewDataset` 从 `my_datasets/` 加载数据集（CUB、Food101、OxfordPets、DTD、FGVC-Aircraft、StanfordCars、StanfordDogs、Flowers102、Places365、EuroSAT 等）。图像短边缩放至 512，然后裁剪为 448×448（训练时随机裁剪，验证时中心裁剪）。`prepare_clip_input()` 使用双三次插值将 448→224 下采样，并使用 CLIP 归一化常数。

### 文本特征

`train_multi_view_stn.py` 中的 `compute_and_save_text_features()` 从 `prompts/{dataset}/cupl.json` 加载类别特定的文本提示，通过 `WeightedTextAggregator`（来自 `text_aggregation.py`）进行聚合，保存到 `text_features/{dataset}_{model}.pt`。这些特征一次性计算后，由所有训练/测试脚本加载使用。

### 检查点

保存在 `checkpoints/{dataset}/`（有监督）或 `checkpoints/unsupervised/{dataset}/`（无监督）。每次运行生成三个检查点：`*_best_loss.pth`（最佳验证损失）、`*_best_acc.pth`（最佳验证准确率）和 `*_latest.pth`（用于断点续训）。文件名编码了 num_views、fusion_mode、hidden_dim 和损失权重信息。

## 关键配置参数

配置文件为 `STN-Config/` 或 `UN-STN-Config/` 下的 YAML 文件。数据集到配置的映射见 `main_stn.py:get_stn_config_path()`。

| 参数 | 含义 |
|------|------|
| `model_size` | CLIP 骨干网络（`ViT-B/16`、`ViT-B/32`） |
| `stn_config.num_views` | STN 裁剪数量（2/4/5/6/8） |
| `stn_config.fusion_mode` | 融合模式：`simple`、`weighted`、`concat`、`transformer` |
| `stn_config.hidden_dim` | 定位 MLP 隐藏维度 |
| `stn_config.logits_temp` | 相似度缩放温度（默认 0.07） |
| `stn_config.classification_weight` | 交叉熵损失权重 |
| `stn_config.decorrelation_weight` | 视角正交性损失权重 |
| `stn_config.fairness_weight` | 防坍缩正则化权重（无监督） |
| `stn_config.kl_consistency_weight` | KL 一致性损失权重（无监督） |
| `stn_config.two_stage.*` | 无监督阶段控制（warmup_epochs、target_update_interval_epochs、温度参数） |
| `training.epochs` | 最大训练轮数 |
| `training.batch_size` | 单卡批次大小 |
| `training.learning_rate` | AdamW 学习率 |
| `training.warmup_epochs` | 学习率预热轮数 |
| `training.patience` | 早停耐心值 |
| `training.max_grad_norm` | 梯度裁剪阈值 |
| `data_path` | 数据集文件根路径 |

## 精度说明

混合精度训练（AMP）已禁用，全部使用 float32。自定义 ViT 实现的 `encode_image()` 返回 `(cls_features, patch_features)` 元组。

## CLI 库

`main_stn.py` 使用 Google Fire（`fire.Fire(main)`）作为命令行接口。其余脚本均使用 `argparse`。
