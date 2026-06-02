# STN-CLIP 无监督细粒度图像分类 — 方法概述

## 1. 模型架构

```
输入图像 448×448
    │
    ├──→ CLIP ViT-B/32 (冻结) ──→ global_CLS ──────────────┐
    │                                                        │
    └──→ LocalizationNetwork ──→ 4 个仿射变换 ──→ 4× crop   │
           (可训练)              grid_sample        (224×224) │
                                        │                     │
                                        └──→ CLIP (冻结) ──→ 4 个 local_features
                                                          │
                                          ┌───────────────┘
                                          │
                                        FusionModule (可训练, Transformer)
                                          │
                                        fused_features [512]
                                          │
                                    × text_features [512, C]
                                          │
                                    fused_logits [C]
```

**三个可训练组件**：

- **LocalizationNetwork**: 1×1 卷积 → 空间卷积 → 池化 → MLP → 4 对 (x,y) 位置参数
- **FusionModule**: Transformer 融合 4 个 local + 1 个 global → fused
- **CLIP**: 全程冻结，作为特征提取器和外部知识源

**测试时**：fused_logits × text_features → argmax → 预测类别

---

## 2. 训练方法：两阶段

### 阶段一（warmup, epoch 1-15）：冻结 CLIP 做 teacher

```
Loss = KL(CLIP || local) + KL(CLIP || fused)
```

| 项 | 梯度流向 |
|---|---------|
| KL(CLIP.detach \|\| local) | CLIP(crop) → localization_network |
| KL(CLIP.detach \|\| fused) | fusion_module → CLIP(crop) → localization_network |

两个模块同时从 CLIP 学习。teacher 温度 0.05，student 温度 0.09。

### 阶段二（epoch 16+）：对称一致性 + CLIP 锚定

```
Loss = KL(fused.detach() || local)          ← Term 1: 全局教局部
     + KL(local_mean.detach() || fused)     ← Term 2: 局部教融合
     + 1.0 × KL(CLIP.detach() || fused)     ← Term 3: CLIP 锚定防坍缩
```

| 项 | 梯度流向 |
|---|---------|
| Term 1 | localization_network |
| Term 2 | fusion_module |
| Term 3 | fusion_module → CLIP(crop) → localization_network |

**设计原理**：

- **Term 1**：fused 综合了全图 + 4 crop 的信息，比单个 local 的认知更完整 → fused 教 local（全局指导局部）
- **Term 2**：4 个 local 聚焦于不同区域，平均后提供多视角共识 → local 教 fused（多样性补充融合）
- **Term 3**：CLIP 作为外部知识锚点，打破 fused↔local 闭环，防止两端合谋坍缩到平凡解

**detach 约定**：所有 teacher 方均 `.detach()`，梯度只流向 student 方。这是自蒸馏的标准做法，防止双向互追导致坍缩。

---

## 3. 温度设计

| 参数 | 阶段一 | 阶段二 |
|------|--------|--------|
| Teacher 侧 | teacher_temp=0.05 (CLIP) | 无（用 detach 后的分布） |
| Student 侧 | warmup_student_temp=0.09 | dec_student_temp=0.15 (两路共享) |

温度逻辑：teacher 尖锐（0.05）给强信号，student 平滑（0.09/0.15）给探索空间。

---

## 4. 损失函数详情

### KL 散度计算

```python
# 单条 KL: KL(target || student)
def _kl_from_logits(student_logits, target_probs, student_temp):
    student_log_probs = log_softmax(student_logits / student_temp)
    return kl_div(student_log_probs, target_probs, reduction='batchmean')

# 多视角 KL: 对 N 个 local view 分别计算 KL 后取 batchmean
def _multi_view_kl_from_logits(view_logits, target_probs, student_temp):
    # view_logits: [B, N, C]
    # 展平为 [B*N, C]，target 扩展为 [B*N, C]
    return kl_div(log_softmax(view_logits_flat / student_temp),
                  target_expand_flat, reduction='batchmean')
```

### 阶段一 loss

```python
global_probs = softmax(CLIP_logits.detach() / 0.05)  # 尖锐 teacher 目标

warmup_local = multi_view_kl(view_logits, global_probs, T=0.09)  # 每个 local → CLIP
warmup_fused = kl(fused_logits, global_probs, T=0.09)            # fused → CLIP

stage1_loss = 1.0 × warmup_local + 1.0 × warmup_fused
```

### 阶段二 loss

```python
# Term 1: fused 教 local
fused_probs = softmax(fused_logits.detach() / 0.15)
dec_local = multi_view_kl(view_logits, fused_probs, T=0.15)

# Term 2: local 教 fused
local_mean_logits = view_logits.mean(dim=1)  # [B, C]
local_mean_probs = softmax(local_mean_logits.detach() / 0.15)
dec_fused = kl(fused_logits, local_mean_probs, T=0.15)

# Term 3: CLIP 锚定
clip_probs = softmax(CLIP_logits.detach() / 0.05)
clip_guidance = kl(fused_logits, clip_probs, T=0.09)

stage2_loss = dec_local + dec_fused + 1.0 × clip_guidance
```

---

## 5. 为什么阶段一有效但原始阶段二无效

**根因**：原始阶段二 `KL(EMA(fused) || local)` 中 teacher = EMA(fused)，fused 在阶段二没有梯度来源：

- EMA teacher 被 detach → fusion_module 断粮
- local 追着一个锁死的 teacher → 学不到新东西
- BestEpoch 永远停在 warmup 结束处

**大量调试实验验证**：
- 使用真实标签替代 teacher → 无效（fused 仍无梯度）
- 调整 EMA 动量 (0.99/0.995/0.999) → 无差异
- EMA vs Periodic 更新方式 → 无差异
- 直接给 fused 做 CE 有监督 → 有效（证明关键在 fused 梯度）

**修复**：Term 2（local→fused）给 fusion_module 梯度，Term 3（CLIP→fused）防止坍缩。

---

## 6. 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| model_size | ViT-B/32 | CLIP 骨干 |
| num_views | 4 | STN 裁剪数量 |
| fusion_mode | transformer | 融合模式 |
| hidden_dim | 256 | 定位网络隐藏维度 |
| batch_size | 64 (per GPU) | 双卡 DDP |
| learning_rate | 5e-5 | AdamW |
| warmup_epochs | 15 | 阶段一持续轮数 |
| teacher_temp | 0.05 | CLIP teacher 温度 |
| warmup_student_temp | 0.09 | 阶段一 student 温度 |
| dec_student_temp | 0.15 | 阶段二 student 温度 |
| clip_guidance_weight | 1.0 | 阶段二 CLIP 约束权重 |
| epochs | 100 | 最大训练轮数 |
| patience | 8 | 早停耐心 |
