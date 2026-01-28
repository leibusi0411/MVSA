"""
文本特征加权聚合模块 - 简化版

基于WCA方法对多个文本描述进行加权聚合
"""

import torch
import torch.nn.functional as F
import clip
from typing import Dict, List


class WeightedTextAggregator:
    """文本特征加权聚合器"""
    
    def __init__(self, text_scale: float, device: str = "cuda"):
        """
        初始化文本聚合器
        
        Args:
            text_scale: 温度参数，与WCA保持一致的使用方式
                       - 现在直接接收已经exp变换后的温度值（如7.39）
                       - 与WCA的helper.py中直接使用tt_scale的方式保持一致
            device: 计算设备
        """
        # 与WCA保持一致：直接使用传入的温度值，不再进行exp变换
        # 调用方负责传入正确的温度值（已经exp变换后的值）
        self.temperature = text_scale
        self.device = device
        
    def aggregate_class_descriptions(
        self, 
        model: torch.nn.Module,
        class_names: List[str],
        descriptions_dict: Dict[str, List[str]],
        use_weighted_aggregation: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """聚合所有类别的文本描述特征"""
        with torch.no_grad():
            aggregated_features = []
            
            # 遍历所有类别名
            for class_name in class_names:
                # 从字典中获取该类别的描述列表
                descriptions = descriptions_dict.get(class_name, [])
                
                
                class_feature = self._aggregate_single_class(
                    model, class_name, descriptions, use_weighted_aggregation
                )
                
                aggregated_features.append(class_feature)
            
            result = torch.stack(aggregated_features, dim=0)
            
            if show_progress:
                print(f"✅ 文本聚合完成: {len(class_names)} 个类别, 特征矩阵: {result.shape}")
            
            return result.to(self.device)
    
    def _aggregate_single_class(
        self,
        model: torch.nn.Module,
        class_name: str,
        descriptions: List[str],
        use_weighted_aggregation: bool
    ) -> torch.Tensor:
        """聚合单个类别的文本描述特征"""
        # 编码所有描述文本
        description_tokens = clip.tokenize(descriptions, truncate=True).to(self.device)
        description_embeddings = model.encode_text(description_tokens)
        description_embeddings = F.normalize(description_embeddings, dim=-1)
        
        if use_weighted_aggregation and len(descriptions) > 1:
            # 生成参考文本并编码
            reference_text = f"a photo of a {class_name}."
            reference_tokens = clip.tokenize([reference_text], truncate=True).to(self.device)
            reference_embedding = model.encode_text(reference_tokens)
            reference_embedding = F.normalize(reference_embedding, dim=-1)
            
            # 计算相似度和权重
            similarities = description_embeddings @ reference_embedding.T #在特征维度上计算相似度
            similarities = similarities.squeeze(-1)  # 移除最后一个维度
            weights = F.softmax(similarities * self.temperature, dim=0)
            
            # 加权聚合
            aggregated_feature = (description_embeddings * weights.unsqueeze(-1)).sum(dim=0)
        else:
            # 简单平均聚合
            aggregated_feature = description_embeddings.mean(dim=0)
        
        return F.normalize(aggregated_feature, dim=-1)
