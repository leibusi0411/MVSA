"""
CLIP Model Wrapper for Fine-tuning
Supports: Linear Probing (LP), Fine-Tuning (FT), and LP-FT


"""
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 先将项目根目录加入搜索路径，确保能访问本地CLIP库和data_preprocess模块
# 路径计算：Baseline/models/clip_model.py -> Baseline/models -> Baseline -> STN-CLIP (项目根目录)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入本地CLIP库（在项目根目录的clip文件夹中）
from clip import clip

# 导入STN主实验的CLIP预处理函数
from data_preprocess import prepare_clip_input

# 支持的模型及其特征维度
SUPPORTED_MODELS = {
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'ViT-L/14': 768,
    'ViT-L/14@336px': 768,
}


#CLIPClassifier类，用于分类任务的CLIP模型封装
class CLIPClassifier(nn.Module):
    """
    CLIP model wrapper for classification tasks.
    Supports flexible fine-tuning strategies.
    """
    
    def __init__(self, model_name='ViT-B/32', num_classes=10, device='cuda', classifier_type='linear', head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        
        # Load pretrained CLIP model 加载预训练CLIP模型 
        self.clip_model, _ = clip.load(model_name, device=device)
        self.feature_dim = SUPPORTED_MODELS[model_name]
        
        # 归一化策略：现统一在数据预处理阶段完成（Base 与 STN 均在 dataloader 中进行 CLIP Normalize）
        
        # Classification head   分类头
        # 支持两种头：'linear' 或 'mlp'
        self.classifier_type = str(classifier_type).lower()
        if self.classifier_type not in ('linear', 'mlp'):
            raise ValueError("classifier_type must be 'linear' or 'mlp'")

        if self.classifier_type == 'linear':
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            # MLP head: Linear -> GeLU -> Dropout -> Linear
            hidden = head_hidden_dim if head_hidden_dim is not None else 256
            drop_p = float(head_dropout) if head_dropout is not None else 0.0
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(p=drop_p),
                nn.Linear(hidden, num_classes)
            )
        self.classifier.to(device)
        
    def forward(self, images):
        """
        Forward pass through CLIP encoder and classifier
        
        Args:
            images: [B, 3, 448, 448] 高分辨率图像（已标准化）
            
        Returns:
            logits: [B, num_classes] 分类logits
        """
        features = self.get_features(images)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, images):
        """
        Extract visual features from CLIP encoder
        
        处理流程：
        1. 输入：[B, 3, 448, 448] 高分辨率图像（已标准化）
        2. 下采样到224x224（复用STN主实验的prepare_clip_input函数）
        3. 输入CLIP ViT编码器
        4. 输出：[B, feature_dim] 特征向量
        
        Args:
            images: [B, 3, 448, 448] 已标准化的高分辨率图像
            
        Returns:
            features: [B, feature_dim] CLIP特征向量
        """
        # 步骤1：下采样到CLIP标准尺寸 (448x448 -> 224x224)
        # 复用STN主实验的prepare_clip_input函数，保证处理流程完全一致
        if images.shape[-1] != 224:  # 如果不是224，则需要下采样
            images = prepare_clip_input(images, clip_size=224)
        # images现在是 [B, 3, 224, 224]
        
        # 步骤2：通过CLIP encoder提取特征（输入已完成 CLIP 标准化）
        
        # 根据训练模式选择是否计算梯度
        with torch.no_grad() if not self.clip_model.visual.training else torch.enable_grad():
            features = self.clip_model.encode_image(images)

        # 新增：处理encode_image可能返回元组的情况
        if isinstance(features, tuple):
            features = features[0]
        
        return features.float()
    
    def freeze_backbone(self):  #冻结backbone（LP模式）
        """Freeze CLIP backbone (for Linear Probing)"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("✓ CLIP backbone frozen (Linear Probing mode)")
    
    def unfreeze_backbone(self):  #解冻backbone（FT模式）
        """
        Unfreeze CLIP's visual backbone and the classifier for Fine-Tuning.
        The text encoder remains frozen.
        """
        # --- 冻结所有参数，然后只解冻需要的 ---
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 1. 解冻图像编码器 (visual backbone)
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
        # 重要：将视觉 backbone 转为 float32，避免在 fp16 下用 AdamW 更新导致 NaN
        try:
            self.clip_model.visual.float()
        except Exception:
            pass
            
        # 2. 解冻分类头
        for param in self.classifier.parameters():
            param.requires_grad = True
        # 确保分类头 dtype 与 backbone 对齐（float32）
        try:
            self.classifier.float()
        except Exception:
            pass
            
        print("✓ CLIP visual backbone and classifier unfrozen (Fine-Tuning mode)")
        print("  - Text encoder remains frozen.")
    
    def freeze_bottom_k_layers(self, k):  #部分冻结
        """Freeze bottom k transformer layers"""
        if 'ViT' not in self.model_name:
            raise NotImplementedError("Layer-wise freezing only implemented for ViT models")
        
        blocks = self.clip_model.visual.transformer.resblocks
        for i in range(min(k, len(blocks))):
            for param in blocks[i].parameters():
                param.requires_grad = False
        print(f"✓ Frozen bottom {k} transformer layers")
    
    def get_trainable_params(self):
        """Get trainable parameters for optimizer"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def load_classifier(self, state_dict):
        """Load classifier weights (for LP-FT)"""
        self.classifier.load_state_dict(state_dict)
        print("✓ Loaded classifier weights")
    
    def save_checkpoint(self, path, epoch, optimizer, best_acc):
        """Save model checkpoint in two formats (full + split)

        - full: 保存整网 state_dict 到给定 path（向后兼容）
        - split: 额外保存 backbone_state_dict 与 classifier_state_dict 到标准化命名的兄弟文件（*_split.pth）

        增强点：加入 schema_version 与更丰富的 head_config，便于 LP-FT 与跨头结构复用。
        """
        schema_version = '1.1'

        # 构造 full checkpoint（原有格式 + schema 与头信息）
        head_type = getattr(self, 'classifier_type', 'linear')
        head_in_features = getattr(self.classifier, 'in_features', None)
        head_out_features = getattr(self.classifier, 'out_features', None)
        full_ckpt = {
            'schema_version': schema_version,
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'best_acc': best_acc,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'classifier_type': head_type,
            'head_in_features': head_in_features,
            'head_out_features': head_out_features,
        }
        torch.save(full_ckpt, path)

        # 构造 split checkpoint（便于跨 head 结构复用）
        backbone_state = {k: v for k, v in self.state_dict().items() if not k.startswith('classifier.')}
        classifier_state = self.classifier.state_dict()
        head_cfg = {
            'type': head_type,
            'in_features': head_in_features,
            'out_features': head_out_features,
        }
        # 尝试提取 hidden_dim / dropout（仅 MLP 头）
        if head_type == 'mlp':
            try:
                lin1 = self.classifier[0]
                drop = self.classifier[2]
                head_cfg['hidden_dim'] = getattr(lin1, 'out_features', None)
                head_cfg['dropout'] = getattr(drop, 'p', 0.0)
            except Exception:
                pass
        split_ckpt = {
            'schema_version': schema_version,
            'epoch': epoch,
            'backbone_state_dict': backbone_state,
            'classifier_state_dict': classifier_state,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'best_acc': best_acc,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'head_config': head_cfg,
        }
        # 统一 split 文件名为 *_split.pth
        base = path[:-4] if path.endswith('.pth') else path
        split_path = f"{base}_split.pth"
        try:
            torch.save(split_ckpt, split_path)
            print(f"✓ Saved checkpoint (full): {path}\n✓ Saved checkpoint (split): {split_path}")
        except Exception:
            print(f"✓ Saved checkpoint (full): {path} (split save skipped)")
    
    def load_checkpoint(self, path, load_optimizer=True):
        """Load model checkpoint (supports full or split formats)

        full 格式包含键 'model_state_dict'；split 格式包含 'backbone_state_dict' 与 'classifier_state_dict'。
        返回原始 checkpoint 字典以便外部按需恢复 optimizer/scheduler。
        """
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # 兼容旧格式/完整格式
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint (full) from {path}")
        elif 'backbone_state_dict' in checkpoint:
            # 分离格式：先 backbone，再 classifier（宽松加载提高兼容性）
            try:
                self.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
            except Exception:
                current = self.state_dict()
                filtered = {k: v for k, v in checkpoint['backbone_state_dict'].items() if k in current and v.shape == current[k].shape}
                current.update(filtered)
                self.load_state_dict(current)
            # 分类头
            try:
                missing, unexpected = self.classifier.load_state_dict(checkpoint.get('classifier_state_dict', {}), strict=False)
                if missing or unexpected:
                    print(f"⚠️  Loaded classifier with strict=False. Missing: {missing}, Unexpected: {unexpected}")
            except Exception as e:
                print(f"⚠️  Failed to load classifier_state_dict: {e}. Using current head init.")
            print(f"✓ Loaded checkpoint (split) from {path}")
        else:
            raise ValueError(f"Unrecognized checkpoint format: {list(checkpoint.keys())}")
        return checkpoint


