"""
多视角STN损失函数模块

Multi-View STN Loss Functions Module

功能特点：
1. 分类损失：传统的交叉熵损失，预测类别标签
2. 对比损失：CLIP风格的对比学习损失，最大化图像-文本相似度
3. 特征去相关损失：鼓励不同视角学习互补特征
4. 自适应分类损失：动态平衡不同视角的训练
5. 灵活配置：支持启用/禁用不同损失组合

设计原则：
- 模块化设计：每个损失可独立使用
- 灵活配置：通过权重控制不同损失的重要性
- 数值稳定：添加必要的稳定性处理
- 高效计算：优化的批处理操作

Multi-view STN loss functions with modular design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast  # 禁用混合精度训练
import math


#顶层损失函数
class MultiViewSTNLoss(nn.Module):
    """
    多视角STN综合损失管理器
    
    支持多种损失的灵活组合：
    1. 分类损失（交叉熵）- 基础分类损失
    2. 特征去相关损失 - 鼓励不同视角学习互补特征
    3. 自适应分类损失 - 动态平衡不同视角的训练
    4. KL散度一致性损失 - 局部视角与融合特征的预测分布一致性
    5. 公平性正则化损失 - 鼓励跨类别的均匀预测，防止模型坍缩
    
    设计原则：
    - 模块化设计，每个损失可独立启用/禁用
    - 通过权重参数控制不同损失的重要性
    - 数值稳定的损失计算
    """
    
    def __init__(self, 
                 logits_temp=0.07,              # 温度参数（用于相似度计算）
                 classification_weight=1.0,     # 标准分类损失权重（0=禁用）
                 decorrelation_weight=0.0,      # 特征去相关损失权重（0=禁用）
                 adaptive_weight=0.0,           # 自适应分类损失权重（0=禁用）
                 kl_consistency_weight=0.0,     # KL散度一致性损失权重（0=禁用）
                 fairness_weight=0.0,           # 公平性正则化损失权重（0=禁用）
                 stn_config=None):              # STN配置字典（用于读取温度参数）
        """
        Args:
            logits_temp (float): 温度参数（用于相似度计算）
            classification_weight (float): 标准分类损失权重，0表示禁用
            decorrelation_weight (float): 特征去相关损失权重，0表示禁用
            adaptive_weight (float): 自适应分类损失权重，0表示禁用
            kl_consistency_weight (float): KL散度一致性损失权重，0表示禁用
            fairness_weight (float): 公平性正则化损失权重，0表示禁用
            stn_config (dict): STN配置字典，用于读取KL损失的温度参数
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.decorrelation_weight = decorrelation_weight
        self.adaptive_weight = adaptive_weight
        self.kl_consistency_weight = kl_consistency_weight
        self.fairness_weight = fairness_weight
        self.temperature = logits_temp
        
        # 保存配置用于KL损失初始化
        if stn_config is None:
            stn_config = {}
        
        # 初始化损失函数组件
        # 标准分类损失
        if classification_weight > 0:
            self.classification_loss = ClassificationLoss()
            print(f"📊 启用标准分类损失 (权重={classification_weight})")
        else:
            self.classification_loss = None
            print("⚠️  标准分类损失已禁用")
        
        # 特征去相关损失（可选）
        if decorrelation_weight > 0:
            self.decorrelation_loss = FeatureDecorrelationLoss()
            print(f"🔗 启用特征去相关损失 (权重={decorrelation_weight})")
        else:
            self.decorrelation_loss = None
            print("⚠️  特征去相关损失已禁用")
        
        # 自适应分类损失（可选）
        if adaptive_weight > 0:
            self.adaptive_loss = AdaptiveClassificationLoss(temperature=logits_temp)
            print(f"🎯 启用自适应分类损失 (权重={adaptive_weight})")
        else:
            self.adaptive_loss = None
            print("⚠️  自适应分类损失已禁用")
        
        # KL散度一致性损失（可选）
        if kl_consistency_weight > 0:
            # 硬编码温度参数：局部视角温度高，融合特征温度低
            self.kl_consistency_loss = KLConsistencyLoss()
            print(f"🔄 启用KL散度一致性损失 (权重={kl_consistency_weight})")
        else:
            self.kl_consistency_loss = None
            print("⚠️  KL散度一致性损失已禁用")
        
        # 公平性正则化损失（可选）
        if fairness_weight > 0:
            self.fairness_loss = FairnessRegularizationLoss()
            print(f"⚖️  启用公平性正则化损失 (权重={fairness_weight})")
        else:
            self.fairness_loss = None
            print("⚠️  公平性正则化损失已禁用")
    
    def forward(self, labels, similarity_or_logits, view_features=None, text_features=None):
        """
        计算综合损失
        
        Args:
            labels (torch.Tensor): 真实标签 [B]
            similarity_or_logits (torch.Tensor): 融合特征的相似度矩阵或logits [B, num_classes]
            view_features (torch.Tensor, optional): 多视角特征 [B, N, D] (去相关、自适应和KL一致性损失需要)
            text_features (torch.Tensor, optional): 文本特征 [D, num_classes] (自适应、KL一致性和公平性损失需要)
                
        Returns:
            tuple: (总损失, 损失详情字典)
        """

        #分别计算分类损失、特征去相关损失、自适应分类损失、KL散度一致性损失、公平性正则化损失
        loss_details = {}

        # 初始化总损失和损失详情
        total_loss = torch.tensor(0.0, device=similarity_or_logits.device, dtype=torch.float32)
        
        # 计算标准分类损失（如果启用）
        classification_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.classification_loss is not None and self.classification_weight > 0:
            classification_loss_value = self.classification_loss(similarity_or_logits, labels)  #传入相似度矩阵logits和真实标签
            weighted_classification_loss = classification_loss_value * self.classification_weight
            total_loss += weighted_classification_loss
            loss_details['classification'] = classification_loss_value.item()
            loss_details['classification_weighted'] = weighted_classification_loss.item()
        else:
            loss_details['classification'] = 0.0
            loss_details['classification_weighted'] = 0.0
        
        # 计算特征去相关损失（如果启用）
        decorrelation_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.decorrelation_loss is not None and self.decorrelation_weight > 0:
            if view_features is None:
                raise ValueError("特征去相关损失需要view_features参数")
            decorrelation_loss_value = self.decorrelation_loss(view_features)   #只传入多视角特征
            weighted_decorrelation_loss = decorrelation_loss_value * self.decorrelation_weight
            total_loss += weighted_decorrelation_loss
            loss_details['decorrelation'] = decorrelation_loss_value.item()
            loss_details['decorrelation_weighted'] = weighted_decorrelation_loss.item()
        else:
            loss_details['decorrelation'] = 0.0
            loss_details['decorrelation_weighted'] = 0.0
        
        # 计算自适应分类损失（如果启用）
        adaptive_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.adaptive_loss is not None and self.adaptive_weight > 0:
            if view_features is None or text_features is None:
                raise ValueError("自适应分类损失需要view_features和text_features参数")
            adaptive_loss_value = self.adaptive_loss(view_features, text_features, labels)  #传入多视角特征、文本特征和真实标签
            weighted_adaptive_loss = adaptive_loss_value * self.adaptive_weight
            total_loss += weighted_adaptive_loss
            loss_details['adaptive'] = adaptive_loss_value.item()
            loss_details['adaptive_weighted'] = weighted_adaptive_loss.item()
        else:
            loss_details['adaptive'] = 0.0
            loss_details['adaptive_weighted'] = 0.0
        
        # 计算KL散度一致性损失（如果启用）
        kl_consistency_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.kl_consistency_loss is not None and self.kl_consistency_weight > 0:
            if view_features is None or text_features is None:
                raise ValueError("KL散度一致性损失需要view_features和text_features参数")
            # 注意：这里传入的similarity_or_logits应该是未经温度缩放的原始相似度
            # 如果已经缩放过，需要在训练脚本中传入原始相似度
            kl_consistency_loss_value = self.kl_consistency_loss(
                view_features=view_features,
                text_features=text_features,
                fused_logits=similarity_or_logits  # 应该是未缩放的原始相似度
            )
            weighted_kl_consistency_loss = kl_consistency_loss_value * self.kl_consistency_weight
            total_loss += weighted_kl_consistency_loss
            loss_details['kl_consistency'] = kl_consistency_loss_value.item()
            loss_details['kl_consistency_weighted'] = weighted_kl_consistency_loss.item()
        else:
            loss_details['kl_consistency'] = 0.0
            loss_details['kl_consistency_weighted'] = 0.0
        
        # 计算公平性正则化损失（如果启用）
        fairness_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.fairness_loss is not None and self.fairness_weight > 0:
            if view_features is None or text_features is None:
                raise ValueError("公平性正则化损失需要view_features和text_features参数")
            fairness_loss_value = self.fairness_loss(
                view_features=view_features,
                text_features=text_features
            )
            weighted_fairness_loss = fairness_loss_value * self.fairness_weight
            total_loss += weighted_fairness_loss
            loss_details['fairness'] = fairness_loss_value.item()
            loss_details['fairness_weighted'] = weighted_fairness_loss.item()
        else:
            loss_details['fairness'] = 0.0
            loss_details['fairness_weighted'] = 0.0
        
        loss_details['total'] = total_loss.item()
        
        return total_loss, loss_details




class ClassificationLoss(nn.Module):
    """
    分类损失：标准的交叉熵损失
    
    计算融合特征与真实标签之间的交叉熵损失
    适用于传统的分类任务设置
    """
    
    def __init__(self):
        """
        标准分类损失，无需额外参数
        """
        super().__init__()
        
    def forward(self, logits, labels):
        """
        计算分类损失
        
        Args:
            logits (torch.Tensor): 经过温度缩放的logits [B, num_classes]
            labels (torch.Tensor): 真实标签 [B]
            
        Returns:
            torch.Tensor: 分类损失标量
        """
        # 直接计算交叉熵损失（logits已经过温度缩放）
        loss = F.cross_entropy(logits.float(), labels)
        return loss




class FeatureDecorrelationLoss(nn.Module):
    """
    特征级去相关损失
    
    核心思想：
    1. 计算N个视角特征的两两相似度矩阵（余弦相似度）
    2. 理想的相似度矩阵应该是单位矩阵（不同视角完全正交）
    3. 计算当前相似度矩阵与单位矩阵的MSE损失
    4. 鼓励不同视角学习不相关的特征表示，实现互补
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, view_features):
        """
        计算特征去相关损失
        
        Args:
            view_features (torch.Tensor): 多视角特征 [B, N, D]
                                        必须已经过L2归一化
             
        Returns:
            torch.Tensor: 去相关损失标量
        """
        batch_size, num_views, feature_dim = view_features.shape
        
        # === 步骤1: 确保特征已经L2归一化 ===
        view_features_norm = F.normalize(view_features, p=2, dim=-1)  # [B, N, D]
        
        # === 步骤2: 计算相似度矩阵（余弦相似度）===
        # 对每个batch计算N×N的相似度矩阵
        similarity_matrices = torch.bmm(
            view_features_norm,  # [B, N, D]
            view_features_norm.transpose(1, 2)  # [B, D, N]
        )  # [B, N, N]
        
        # === 步骤3: 创建理想的单位矩阵   对角线为1，其他为0
        identity_matrix = torch.eye(
            num_views, 
            device=view_features.device, 
            dtype=view_features.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        
        # === 步骤4: 计算MSE损失 ===
        # 计算当前相似度矩阵与单位矩阵的均方误差
        mse_loss = F.mse_loss(similarity_matrices, identity_matrix)
        
        return mse_loss


class AdaptiveClassificationLoss(nn.Module):
    """
    自适应分类损失
    
    核心思想：
    1. 计算每个视角独立的分类logits和损失
    2. 根据每个视角的置信度计算自适应权重
    3. 表现差的视角获得更高的权重，促进改进
    4. 实现视角间的动态平衡训练
    """
    
    def __init__(self, temperature=0.07, penalty_temperature=0.1):
        super().__init__()
        self.temperature = temperature  #使用自定义温度参数，不是外部传入的

        # 引入一个新的超参数，专门用于锐化惩罚权重
        self.penalty_temperature = penalty_temperature
        
    def forward(self, view_features, text_features, labels):
        """
        
        Args:
            view_features (torch.Tensor): 多视角特征 [B, N, D]
            text_features (torch.Tensor): 文本特征 [D, num_classes]
            labels (torch.Tensor): 真实标签 [B]
            
        Returns:
            torch.Tensor: 自适应分类损失标量
        """
        batch_size, num_views, feature_dim = view_features.shape
        num_classes = text_features.shape[1]
        
        # === 步骤1: 确保特征已经L2归一化 ===
        view_features_norm = F.normalize(view_features, p=2, dim=-1)  # [B, N, D]
        text_features_norm = F.normalize(text_features, p=2, dim=0)  # [D, num_classes]
        
        # === 步骤2: 向量化计算所有视角的logits ===
        # 重塑view_features以便进行批量矩阵乘法: [B*N, D]
        view_features_reshaped = view_features_norm.reshape(-1, feature_dim)  # [B*N, D]
        
        # 一次性计算所有视角与文本特征的相似度: [B*N, num_classes]
        similarity_all = torch.matmul(view_features_reshaped, text_features_norm)  # [B*N, num_classes]
        
        # 应用温度缩放
        logits_all = similarity_all / self.temperature  # [B*N, num_classes]   计算损失的logits
        
        # 重塑回原始维度: [B, N, num_classes]
        logits_matrix = logits_all.reshape(batch_size, num_views, num_classes)  # [B, N, num_classes]
        
        # === 步骤3: 向量化计算损失和置信度 ===
        # 扩展labels以匹配所有视角: [B] -> [B, N]  N为视角数
        labels_expanded = labels.unsqueeze(1).expand(-1, num_views)  # [B, N]
        
        # 重塑为[B*N]以便计算交叉熵损失
        labels_reshaped = labels_expanded.reshape(-1)  # [B*N]
        
        # 计算所有视角的交叉熵损失
        losses_all = F.cross_entropy(logits_all, labels_reshaped, reduction='none')  # [B*N]
        
        # 重塑损失矩阵: [B, N]
        losses_matrix = losses_all.reshape(batch_size, num_views)  # [B, N]
        
        # 计算所有视角的softmax概率
        probs_all = F.softmax(logits_all, dim=-1)  # [B*N, num_classes]
        
        # 获取正确类别的概率（置信度）
        # 使用高级索引一次性提取所有正确类别概率
        batch_indices = torch.arange(batch_size * num_views, device=view_features.device)
        correct_class_probs_all = probs_all[batch_indices, labels_reshaped]  # [B*N]
        
        # 重塑置信度矩阵: [B, N]
        confidences_matrix = correct_class_probs_all.reshape(batch_size, num_views)  # [B, N]
        
        # # === 步骤4: 计算自适应权重和加权损失 ===
        # # 计算与表现成反比的权重（表现差的视角权重更高）
        # penalty_weights = (1.0 - confidences_matrix).detach()  # [B, N] 阻止梯度回传
        
        # # 应用自适应权重
        # weighted_losses = losses_matrix * penalty_weights  # [B, N]
        
        # # 计算平均损失
        # adaptive_loss = weighted_losses.mean()
        
        # 4.1: 计算原始的、与表现成反比的权重
        raw_penalty_weights = (1.0 - confidences_matrix) # [B, N]
        
        # --- 新增步骤: 4.2 ---
        # 使用低温Softmax对原始惩罚权重进行“锐化”
        # 表现最差的视角（(1-p)最大），其权重会最接近1
        # 其他视角的权重会被压缩到接近0   除以温度参数进行放大锐化，然后softmax是计算概率作为权重
        sharpened_penalty_weights = F.softmax(
            raw_penalty_weights / self.penalty_temperature, 
            dim=1
        ).detach() # .detach() 仍然是关键，阻止不必要的梯度流
        
        # --- 修改步骤: 4.3 ---
        # 应用*锐化后*的自适应权重
        weighted_losses = losses_matrix * sharpened_penalty_weights  # [B, N]
        
        # --- 修改步骤: 4.4 --- 
        # 计算总损失。这里使用.sum()可能比.mean()更直观，
        # 因为权重已经经过softmax归一化了。
        # .sum()会得到每个样本的加权损失，再.mean()得到批次平均。
        adaptive_loss = weighted_losses.sum(dim=1).mean()  #每个样本的加权损失，再.mean()得到批次平均。

        return adaptive_loss



class FairnessRegularizationLoss(nn.Module):
    """
    公平性正则化损失：鼓励跨类别的均匀预测，防止模型坍缩
    
    核心思想：
    1. 计算当前batch中所有样本对每个类别的平均预测概率 p̄_c
    2. 最大化平均预测分布的熵：L_reg = -Σ log(p̄_c)
    3. 强迫模型"雨露均沾"，不要只盯着某一个类别输出
    4. 防止无监督训练中的模型坍缩问题
    
    数学公式：
        L_reg = -Σ_{c=1}^{C} log(p̄_c)
        其中 p̄_c = (1/B) Σ_{i=1}^{B} p_i(c)
    
    优势：
    - 防止坍缩：避免模型将所有样本预测为同一类别
    - 无监督友好：不需要标签信息
    - 数值稳定：使用log-sum-exp技巧
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, view_features, text_features):
        """
        计算公平性正则化损失
        
        Args:
            view_features (torch.Tensor): 多视角特征 [B, N, D]
            text_features (torch.Tensor): 文本特征 [D, num_classes]
            
        Returns:
            torch.Tensor: 公平性正则化损失标量
        """
        batch_size, num_views, feature_dim = view_features.shape
        num_classes = text_features.shape[1]
        
        # === 步骤1: 确保特征已经L2归一化 ===
        view_features_norm = F.normalize(view_features, p=2, dim=-1)  # [B, N, D]
        text_features_norm = F.normalize(text_features, p=2, dim=0)  # [D, num_classes]
        
        # === 步骤2: 计算所有视角的logits ===
        # 重塑view_features: [B*N, D]
        view_features_reshaped = view_features_norm.reshape(-1, feature_dim)  # [B*N, D]
        
        # 计算相似度: [B*N, num_classes]
        similarity_all = torch.matmul(view_features_reshaped, text_features_norm)  # [B*N, num_classes]
        
        # === 步骤3: 计算预测概率分布 ===
        # 使用softmax得到概率分布（不需要温度缩放，因为我们关注的是分布形状）
        probs_all = F.softmax(similarity_all, dim=-1)  # [B*N, num_classes]
        
        # === 步骤4: 计算batch内的平均预测概率 ===
        # 对所有样本和视角求平均，得到每个类别的平均预测概率
        mean_probs = probs_all.mean(dim=0)  # [num_classes]
        
        # === 步骤5: 计算负对数似然（最大化熵）===
        # L_reg = -Σ log(p̄_c)
        # 添加epsilon防止log(0)
        epsilon = 1e-8
        fairness_loss = -torch.log(mean_probs + epsilon).sum()
        
        return fairness_loss


class KLConsistencyLoss(nn.Module):
    """
    KL散度一致性损失：局部视角预测分布与融合特征预测分布的一致性
    
    核心思想：
    1. 计算每个局部视角的预测分布 P_view(y|x)
    2. 使用融合特征的预测分布 P_fused(y|x) 作为目标分布
    3. 最小化 KL(P_fused || P_view)，使局部视角的预测接近融合预测
    4. 适用于无监督场景，不需要真实标签
    
    温度蒸馏策略（硬编码）：
    - 局部视角温度高（0.14）→ 预测分布平滑（soft），鼓励探索
    - 融合特征温度低（0.05）→ 预测分布尖锐（hard），提供明确监督
    
    优势：
    - 无监督：不依赖标签信息
    - 一致性：鼓励局部视角与全局融合保持一致
    - 互补性：配合去相关损失，实现"一致但互补"的视角学习
    - 温度蒸馏：通过不同温度控制学习难度
    """
    
    def __init__(self):
        """
        温度参数硬编码：
        - view_temperature = 0.1 (局部视角，soft分布)
        - fused_temperature = 0.05 (融合特征，hard分布)
        """
        super().__init__()
        # 硬编码温度参数
        self.view_temperature = 0.1    # 局部视角温度（约1.4倍CLIP标准温度0.07）
        self.fused_temperature = 0.05  # 融合特征温度（0.7倍CLIP标准温度）
        
        print(f"    🌡️  KL一致性损失温度配置（硬编码）:")
        print(f"       - 局部视角温度: {self.view_temperature} (soft, 鼓励探索)")
        print(f"       - 融合特征温度: {self.fused_temperature} (hard, 明确监督)")
        
    def forward(self, view_features, text_features, fused_logits):
        """
        计算KL散度一致性损失（支持不同温度参数）
        
        Args:
            view_features (torch.Tensor): 多视角特征 [B, N, D]
            text_features (torch.Tensor): 文本特征 [D, num_classes]
            fused_logits (torch.Tensor): 融合特征的logits [B, num_classes] (未经温度缩放的原始相似度)
            
        Returns:
            torch.Tensor: KL散度一致性损失标量
        """
        batch_size, num_views, feature_dim = view_features.shape
        num_classes = text_features.shape[1]
        
        # === 步骤1: 确保特征已经L2归一化 ===
        view_features_norm = F.normalize(view_features, p=2, dim=-1)  # [B, N, D]
        text_features_norm = F.normalize(text_features, p=2, dim=0)  # [D, num_classes]
        
        # === 步骤2: 计算融合特征的目标分布（使用融合温度）===
        # 注意：fused_logits是未经温度缩放的原始相似度
        # 使用较低的温度使分布更尖锐（hard target）
        fused_logits_scaled = fused_logits / self.fused_temperature  # [B, num_classes]
        target_distribution = F.softmax(fused_logits_scaled.detach(), dim=-1)  # [B, num_classes]
        
        # === 步骤3: 向量化计算所有局部视角的logits ===
        # 重塑view_features以便进行批量矩阵乘法: [B*N, D]
        view_features_reshaped = view_features_norm.reshape(-1, feature_dim)  # [B*N, D]
        
        # 一次性计算所有视角与文本特征的相似度: [B*N, num_classes]
        similarity_all = torch.matmul(view_features_reshaped, text_features_norm)  # [B*N, num_classes]
        
        # 应用局部视角温度缩放（使用较高的温度使分布更平滑）
        view_logits_all = similarity_all / self.view_temperature  # [B*N, num_classes]
        
        # 重塑回原始维度: [B, N, num_classes]
        view_logits_matrix = view_logits_all.reshape(batch_size, num_views, num_classes)  # [B, N, num_classes]
        
        # === 步骤4: 计算每个视角的预测分布 ===
        view_distributions = F.softmax(view_logits_matrix, dim=-1)  # [B, N, num_classes]
        
        # === 步骤5: 计算KL散度 ===
        # KL(P_target || P_view) = sum(P_target * log(P_target / P_view))
        # 使用PyTorch的kl_div函数：kl_div(log(P_view), P_target)
        # 注意：kl_div期望第一个参数是log概率，第二个参数是目标概率
        
        # 扩展目标分布以匹配所有视角: [B, num_classes] -> [B, N, num_classes]
        target_distribution_expanded = target_distribution.unsqueeze(1).expand(-1, num_views, -1)  # [B, N, num_classes]
        
        # 计算log概率（添加epsilon防止log(0)）
        epsilon = 1e-8
        log_view_distributions = torch.log(view_distributions + epsilon)  # [B, N, num_classes]
        
        # 计算KL散度（reduction='batchmean'会对所有元素求平均）
        kl_loss = F.kl_div(
            log_view_distributions.reshape(-1, num_classes),  # [B*N, num_classes]
            target_distribution_expanded.reshape(-1, num_classes),  # [B*N, num_classes]
            reduction='batchmean'
        )
        
        return kl_loss




