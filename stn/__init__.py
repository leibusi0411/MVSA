"""
Multi-View Spatial Transformer Network (MV-STN) module for diverse visual perspective learning.

多视角空间变换网络模块

核心设计理念：
通过多个并联的STN分支学习不同的空间视角，捕获目标对象的多样化特征表示，
提升视觉-语言模型在复杂场景下的分类性能和泛化能力。

主要组件：
1. MultiViewSTNModel: 多视角STN核心模型，包含N个并联的变换分支
2. MultiViewSTNLoss: 综合损失管理器，统一管理多种损失的组合
3. ClassificationLoss: 标准分类损失，基于交叉熵的传统分类损失
4. FeatureDecorrelationLoss: 特征去相关损失，确保不同视角学习互补特征
5. AdaptiveClassificationLoss: 自适应分类损失，动态平衡各视角的训练权重
6. ContrastiveLoss: 对比损失，CLIP风格的视觉-文本对比学习损失

架构流程：
输入图像 → N个STN分支 → N个变换后视角 → N个CLIP特征 → 特征融合 → 最终特征

技术特点：
- 多样性约束：确保不同分支关注不同的图像区域
- 自适应融合：支持简单平均、加权平均、注意力机制、拼接降维、Transformer融合
- 共享CLIP编码器：降低参数量和内存占用
- 端到端训练：统一优化所有分支和融合策略

支持的融合模式：
- simple: 简单平均融合
- weighted: 基于相似度的加权融合  
- attention: 基于注意力机制的自适应融合
- concat: 特征拼接后通过MLP降维
- transformer: 基于Transformer的序列建模融合

使用方法：
```python
from stn import MultiViewSTNModel, MultiViewSTNLoss
from clip import clip

# 加载CLIP模型
clip_model, _ = clip.load("ViT-B/32")

# 配置多视角STN参数
stn_config = {
    'hidden_dim': 256,
    'num_views': 8,
    'fusion_mode': 'attention',
    'dropout': 0.1,
    'decorrelation_weight': 0.1,
    'adaptive_weight': 1.0
}

# 创建多视角STN模型
mv_stn_model = MultiViewSTNModel(clip_model, stn_config, num_views=8)

# 创建多视角STN损失函数
criterion = MultiViewSTNLoss(
    decorrelation_weight=0.1,
    adaptive_weight=1.0,
    logits_temp=0.07
)

# 前向传播
similarity_logits, view_features = mv_stn_model(images, return_intermediate=True)

# 计算损失
loss, loss_details = criterion(
    labels=labels,
    similarity_or_logits=similarity_logits,
    view_features=view_features,
    text_features=text_features
)
```

This module implements a multi-view STN that learns diverse spatial transformations
to capture different perspectives of objects for enhanced visual representation.
"""

# 子模块导入
# 只导入实际存在的模块

# 1. 多视角STN模型（主要模块）
from .multi_view_stn import MultiViewSTNModel

# 2. 多视角STN损失函数
from .loss_multi import (
    MultiViewSTNLoss,
    ClassificationLoss,
    FeatureDecorrelationLoss, 
    AdaptiveClassificationLoss,
    ContrastiveLoss
)

# __all__ 列表明确指定了当其他Python文件使用 from stn import * 语法时会导入哪些类或函数
# 这样可以控制模块的公开接口，避免导入内部实现细节
__all__ = [
    'MultiViewSTNModel',              # 多视角STN模型
    'MultiViewSTNLoss',               # 多视角STN综合损失管理器
    'ClassificationLoss',             # 标准分类损失（交叉熵）
    'FeatureDecorrelationLoss',       # 特征去相关损失
    'AdaptiveClassificationLoss',     # 自适应分类损失
    'ContrastiveLoss'                 # 对比损失（CLIP风格）
]
