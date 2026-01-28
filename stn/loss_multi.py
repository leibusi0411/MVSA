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
    
    设计原则：
    - 模块化设计，每个损失可独立启用/禁用
    - 通过权重参数控制不同损失的重要性
    - 数值稳定的损失计算
    """
    
    def __init__(self, 
                 logits_temp=0.07,              # 温度参数（用于相似度计算）
                 classification_weight=1.0,     # 标准分类损失权重（0=禁用）
                 contrastive_weight=0.0,        # 对比损失权重（0=禁用）
                 decorrelation_weight=0.0,      # 特征去相关损失权重（0=禁用）
                 adaptive_weight=0.0):          # 自适应分类损失权重（0=禁用）
        """
        Args:
            logits_temp (float): 温度参数（用于相似度计算）
            classification_weight (float): 标准分类损失权重，0表示禁用
            contrastive_weight (float): 对比损失权重，0表示禁用
            decorrelation_weight (float): 特征去相关损失权重，0表示禁用
            adaptive_weight (float): 自适应分类损失权重，0表示禁用
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.decorrelation_weight = decorrelation_weight
        self.adaptive_weight = adaptive_weight
        self.temperature = logits_temp
        
        # 初始化损失函数组件
        # 标准分类损失
        if classification_weight > 0:
            self.classification_loss = ClassificationLoss()
            print(f"📊 启用标准分类损失 (权重={classification_weight})")
        else:
            self.classification_loss = None
            print("⚠️  标准分类损失已禁用")
        
        # 对比损失
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss(temperature=logits_temp, use_cosine_sim=True)
            print(f"🌟 启用对比损失 (权重={contrastive_weight}, 温度={logits_temp})")
        else:
            self.contrastive_loss = None
            print("⚠️  对比损失已禁用")
        
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
    
    def forward(self, labels, similarity_or_logits, view_features=None, text_features=None):
        """
        计算综合损失
        
        Args:
            labels (torch.Tensor): 真实标签 [B]
            similarity_or_logits (torch.Tensor): 相似度矩阵或logits [B, num_classes]
            view_features (torch.Tensor, optional): 多视角特征 [B, N, D] (去相关和自适应损失需要)
            text_features (torch.Tensor, optional): 文本特征 [D, num_classes] (自适应损失需要)
                
        Returns:
            tuple: (总损失, 损失详情字典)
        """

        #分别计算分类损失、特征去相关损失、自适应分类损失
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
        
        # 计算对比损失（如果启用）
        contrastive_loss_value = torch.tensor(0.0, device=total_loss.device)
        if self.contrastive_loss is not None and self.contrastive_weight > 0:
            # 使用预计算的相似度矩阵
            contrastive_loss_value = self.contrastive_loss(
                labels=labels,
                similarity_matrix=similarity_or_logits
            )
            weighted_contrastive_loss = contrastive_loss_value * self.contrastive_weight
            total_loss += weighted_contrastive_loss
            loss_details['contrastive'] = contrastive_loss_value.item()
            loss_details['contrastive_weighted'] = weighted_contrastive_loss.item()
        else:
            loss_details['contrastive'] = 0.0
            loss_details['contrastive_weighted'] = 0.0
        
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



class ContrastiveLoss(nn.Module):
    """
    批内对比损失：图-文批内对比学习损失
    
    核心思想：
    1. 根据labels从logits中提取对应类别的相似度，构建B×B的相似度矩阵
    2. 对角线元素为正样本对（图片i与其对应文本i）
    3. 非对角线元素为负样本对（图片i与其他文本j）
    4. 使用交叉熵损失优化，期望正样本相似度最大
    """
    
    def __init__(self, temperature=0.07, use_cosine_sim=True):
        """
        Args:
            temperature (float): 温度参数，控制相似度分布的尖锐程度
            use_cosine_sim (bool): 是否使用余弦相似度（推荐）
        """
        super().__init__()
        self.temperature = temperature
        self.use_cosine_sim = use_cosine_sim
        
    def forward(self, labels, similarity_matrix):
        """
        计算批内对比损失
        
        Args:
            labels (torch.Tensor): 真实标签 [B]
            similarity_matrix (torch.Tensor): 经过温度缩放的logits矩阵 [B, num_classes]
            
        Returns:
            torch.Tensor: 对比损失标量
        """
        batch_size = labels.size(0)
        device = labels.device
        
        # === 步骤1: 使用预计算的温度缩放logits矩阵 ===
        logits_matrix = similarity_matrix.float()  # [B, num_classes] 已经过温度缩放
        
        # === 步骤2: 根据labels提取对应列，构建B×B相似度矩阵 ===
        # 提取每个样本对应类别的列
        batch_text_logits = logits_matrix[:, labels]  # [B, B]
        # batch_text_logits[i, j] = 图片i与样本j对应类别文本的相似度（已温度缩放）
        
        # === 步骤3: 构建有监督的正样本掩码 ===
        # positive_mask[i, j] is True if image_i and text_j have the same class
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # [B, B]
        
        # === 步骤4: 计算有监督对比损失 (InfoNCE形式) ===
        # 为了数值稳定性，从logits中减去最大值（logits已经过温度缩放）
        logits_max, _ = torch.max(batch_text_logits, dim=1, keepdim=True)
        stable_logits = batch_text_logits - logits_max.detach()  # [B, B]

        # 计算 exp(logits)
        exp_logits = torch.exp(stable_logits)  # [B, B]对矩阵中的每个元素都计算指数值
        
        # 分子: sum of exp(logits) for all positive pairs (包括对角线)
        # (bs, bs) * (bs, bs) -> 逐元素相乘，只保留正样本的exp_logits
        # .sum(1) -> 按行求和
        sum_exp_positives = (exp_logits * positive_mask).sum(dim=1) #两个矩阵逐元素相乘，然后按行求和  分子，维度为[B]

        # 分母: sum of exp(logits) for all pairs (包括正样本和负样本)
        sum_exp_all = exp_logits.sum(dim=1)  #logits矩阵按行求和，分母，维度为[B]
        
        # 计算每个样本的损失
        # -log(分子 / 分母)
        # 加上epsilon防止log(0)
        epsilon = 1e-8
        log_probs = torch.log(sum_exp_positives / (sum_exp_all + epsilon) + epsilon)  #维度为[B]，计算每个样本的损失
        
        # 我们希望最大化log_probs，所以损失是它的负数
        # 现在所有样本都有正样本（至少包括自身），所以不需要过滤
        loss = -log_probs.mean()
            
        return loss
        

