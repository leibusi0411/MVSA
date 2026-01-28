"""
STN模型测试脚本

简化版STN独立测试流程：
1. 加载预训练的STN模型
2. 加载预计算的文本特征
3. 对每张图片进行STN空间变换
4. 计算变换后图像与所有文本特征的相似度
5. 选择最高分数的类别作为预测结果
6. 计算准确率等评估指标

核心优势：
- STN自主学习空间变换，无需文本指导
- 直接与所有类别比较，测试流程简单直接
- 不依赖WCA筛选，独立完成图像分类任务
- 保持STN空间变换的精度优势
- 自动可视化第一个批次的STN变换结果

配置管理：
- 自动加载STN训练配置文件 (STN-Config/{dataset_name}.yaml)
- 自动获取训练时的model_size，确保配置一致性
- 使用双重图像数据加载方式，但不进行随机裁剪增强
- 确保测试配置与训练配置完全一致
- 支持命令行参数灵活配置
- 简化版：不需要WCA筛选结果，直接与所有类别比较

模型文件路径：
- 自动加载训练好的多视角STN模型: checkpoints/multi_view_stn_{dataset_name}_{model_size}_{architecture}_{loss_weights}.pth
- 例如: checkpoints/multi_view_stn_cub_ViT-B_32_views4_concat_dim256_temp0.07_cls1.0.pth
- 注意: 模型文件由train_multi_view_stn.py生成，保存在checkpoints/目录下

使用方法：

python test_multi_stn.py --dataset_name=cub

# 不同数据集测试
python test_multi_stn.py --dataset_name=imagenet
python test_multi_stn.py --dataset_name=food101



# 查看帮助
python test_multi_stn.py --help

"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import numpy as np
from collections import defaultdict
import argparse
import yaml
import random
import matplotlib.pyplot as plt
from torchvision import transforms

# 项目内部模块导入
from clip import clip
from dataset_utils import load_classes
from utils import imagenet_a_lt, imagenet_r_lt


from my_datasets import MyDataset
from torchvision import datasets
import traceback


def accuracy(output, target, n, dataset_name):
    """
    计算分类准确率
    
    对于ImageNet变体数据集（ImageNet-A, ImageNet-R）：
    - 这些数据集只包含ImageNet 1000类的子集（200类）
    - ImageFolder加载时标签为0-199（按文件夹字母顺序）
    - 需要使用映射列表(imagenet_a_lt/imagenet_r_lt)选择对应的logits
    
    Args:
        output: 模型输出的logits [N, num_classes]
        target: 真实标签 [N]（ImageNet-A/R为0-199，其他为原始标签）
        n: 样本数量
        dataset_name: 数据集名称
    
    Returns:
        float: Top-1准确率（百分比）
    """
    # 对于ImageNet-A：只从1000类中选出对应的200类
    if dataset_name in ['imagenet-a', 'imageneta']:
        _, pred = output[:, imagenet_a_lt].max(1)  # 在200个类别中预测
    # 对于ImageNet-R：只从1000类中选出对应的200类
    elif dataset_name in ['imagenet-r', 'imagenetr']:
        _, pred = output[:, imagenet_r_lt].max(1)  # 在200个类别中预测
    # 其他数据集：直接在所有类别中预测
    else:
        _, pred = output.max(1)
    
    # 比较预测与真实标签
    correct = pred.eq(target)
    
    # 计算Top-1准确率
    return float(correct.float().sum().cpu().numpy()) / n * 100


def visualize_preprocessed_vs_stn(preprocessed_images, transformed_images, dataset_name, config=None, save_dir="visualizations/test_stn_transforms", batch_idx=0, max_samples=None, theta_matrices=None, position_params=None):
    """
    可视化预处理后图片与STN变换后多视角图片的对比，包含变换矩阵信息
    
    Args:
        preprocessed_images: 预处理后的输入图像 [B, 3, 448, 448]
        transformed_images: STN变换后的多视角图像 [B, N, 3, 224, 224]
        dataset_name: 数据集名称
        config: 配置字典，包含STN和训练参数
        save_dir: 保存根目录
        batch_idx: 批次索引
        max_samples: 最大样本数，如果为None则处理整个批次
        theta_matrices: 变换矩阵 [B, N, 2, 3] (可选)
        position_params: 位置参数 [B, 2*N] (可选)
    """
    import os
    
    # 从配置中提取关键参数
    if config is not None:
        stn_config = config.get('stn_config', {})
        training_config = config.get('training', {})
        model_size = config.get('model_size', 'ViT-B_32')
        
        # 提取STN配置参数
        num_views = stn_config.get('num_views', 4)
        fusion_mode = stn_config.get('fusion_mode', 'concat')
        hidden_dim = stn_config.get('hidden_dim', 256)
        logits_temp = stn_config.get('logits_temp', None)  # STN配置中的温度参数
        
        # 提取损失权重参数
        classification_weight = stn_config.get('classification_weight', 1.0)
        contrastive_weight = stn_config.get('contrastive_weight', 0.0)
        decorrelation_weight = stn_config.get('decorrelation_weight', 0.0)
        adaptive_weight = stn_config.get('adaptive_weight', 0.0)
        
        # 提取训练配置参数（如果存在温度参数）
        temperature = training_config.get('temperature', logits_temp)  # 优先使用training配置，其次使用STN配置
        
        # 清理模型名称
        model_name_clean = model_size.replace('/', '_')
        
        # 构建详细的配置字符串
        config_parts = [
            f"model_{model_name_clean}",
            f"views{num_views}",
            f"{fusion_mode}",
            f"dim{hidden_dim}"
        ]
        
        # 添加温度参数
        if temperature is not None:
            config_parts.append(f"temp{temperature}")
        
        # 添加损失权重信息到路径中
        loss_parts = []
        if classification_weight > 0:
            loss_parts.append(f"cls{classification_weight}")
        if contrastive_weight > 0:
            loss_parts.append(f"con{contrastive_weight}")
        if decorrelation_weight > 0:
            loss_parts.append(f"dec{decorrelation_weight}")
        if adaptive_weight > 0:
            loss_parts.append(f"adp{adaptive_weight}")
        
        # 将损失权重部分添加到配置字符串中
        if loss_parts:
            config_parts.extend(loss_parts)
        
        config_str = "_".join(config_parts)
        
        # 根据数据集和配置参数组织保存路径
        dataset_save_dir = os.path.join(save_dir, dataset_name, config_str)
        
        print(f"🎨 保存配置: {config_str}")
        print(f"   📊 损失权重: cls={classification_weight}, con={contrastive_weight}, dec={decorrelation_weight}, adp={adaptive_weight}")
    else:
        # 如果没有配置信息，使用简单的数据集路径
        dataset_save_dir = os.path.join(save_dir, dataset_name, "default_config")
        print("⚠️ 未提供配置信息，使用默认路径")
    
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # 如果max_samples为None，则处理整个批次
    batch_size = preprocessed_images.size(0) if max_samples is None else min(preprocessed_images.size(0), max_samples)
    num_views = transformed_images.size(1)
    
    print(f"🎨 开始可视化数据集 {dataset_name} 的第 {batch_idx} 批次，共 {batch_size} 张图片")
    
    # 使用CLIP标准化参数进行反标准化
    def denormalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
        """将CLIP标准化的张量转换回[0,1]范围"""
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean
    
    # 张量转PIL图像
    def tensor_to_pil(tensor):
        """将张量转换为PIL图像"""
        tensor = torch.clamp(tensor, 0, 1)  # 确保在[0,1]范围内
        tensor = tensor.cpu()
        return transforms.ToPILImage()(tensor)
    
    for i in range(batch_size):
        # 创建子图：1个预处理后图像 + N个STN变换后的图像
        fig_height = 6 if (theta_matrices is not None or position_params is not None) else 4
        fig, axes = plt.subplots(1, 1 + num_views, figsize=(4 * (1 + num_views), fig_height))
        
        # 显示预处理后图像（448x448）
        preprocessed_denorm = denormalize(preprocessed_images[i])
        preprocessed_pil = tensor_to_pil(preprocessed_denorm)
        axes[0].imshow(preprocessed_pil)
        axes[0].set_title('Input_image 448x448', fontsize=10)
        axes[0].axis('off')
        
        # 显示STN变换后的图像（224x224）并添加变换矩阵信息
        for view_idx in range(num_views):
            transformed_denorm = denormalize(transformed_images[i, view_idx])
            transformed_pil = tensor_to_pil(transformed_denorm)
            axes[1 + view_idx].imshow(transformed_pil)
            
            # 构建标题，包含变换矩阵信息
            title = f'STN View {view_idx+1} 224x224'
            
            # 添加原始位置参数信息
            if position_params is not None:
                param_x = position_params[i, view_idx * 2].item()     # x坐标
                param_y = position_params[i, view_idx * 2 + 1].item() # y坐标
                title += f'\nPos: ({param_x:.3f}, {param_y:.3f})'
            
            axes[1 + view_idx].set_title(title, fontsize=9)
            axes[1 + view_idx].axis('off')
        
        plt.tight_layout()
        
        # 如果有变换矩阵信息，在底部添加详细信息
        if theta_matrices is not None:
            fig.text(0.5, 0.02, f'Sample {i}: Transformation matrices show scale and translation parameters', 
                     ha='center', fontsize=8, style='italic')
        
        # 保存图像到数据集专用目录
        save_path = os.path.join(dataset_save_dir, f'batch{batch_idx}_sample{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if i < 5 or (i + 1) % 10 == 0:  # 只显示前5张和每10张的保存信息，避免输出过多
            print(f"✅ 保存第 {i+1} 张图片: {save_path}")
    
    # 输出总结信息
    print(f"🎉 完成可视化！共保存 {batch_size} 张图片到目录: {dataset_save_dir}")
    print(f"📁 文件命名格式: batch{batch_idx}_sample{{i:03d}}.png")


def stn_precise_testing(stn_model, dataloader, device, dataset_name, 
                       precomputed_text_features, config=None, model_tag=None):
    """
    多视角STN精确测试
    
    测试流程：
    1. 对每张输入图片：
       a. 多视角STN模型学习多个空间变换并融合特征
       b. 计算融合后的图像特征与预计算文本特征的相似度矩阵
       c. 选择相似度最高的类别作为预测
    
    2. 统计预测准确率并返回详细结果
    
    核心优势：
    - 多视角空间变换，捕获目标的不同角度信息
    - 不需要动态计算文本特征，直接使用预计算结果
    - 与训练过程保持一致的数据流（448x448输入）
    - 不依赖WCA筛选，直接与所有类别比较
    
    Args:
        stn_model: 训练好的多视角STN模型 (MultiViewSTNModel)
        dataloader: 测试数据加载器 (images_448, labels)
        device: 计算设备
        dataset_name: 数据集名称  
        precomputed_text_features: 预计算文本特征张量 [feature_dim, num_classes]
        config: STN配置(可选)
        
    Returns:
        float: 百分比形式的Top-1准确率
    """
    print(f"=== 简化版STN精确测试阶段（WCA风格）===")
    print(f"📋 三阶段测试流程（与WCA一致）:")
    print(f"   🔄 阶段1：批量STN特征提取 - 收集所有变换后的图像特征")
    print(f"   🔄 阶段2：特征合并 - 将所有批次特征合并为完整数据集")
    print(f"   🔄 阶段3：一次性相似度计算 - 使用WCA的accuracy函数")
    
    stn_model.eval()
    
    # 第一步：收集所有STN变换后的图像特征（类似WCA的方式）
    print("🔄 第一阶段：收集所有STN变换后的图像特征...")
    all_image_features = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images_448, labels) in enumerate(tqdm(dataloader, desc="STN特征提取")):
            # 新数据格式：
            # images_448: 标准化后的图像张量 [B, 3, 448, 448]
            # labels: 标签张量 [B]
            images_448 = images_448.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # === 强制转换为float32精度 ===
            # 确保输入与模型权重的数据类型一致
            images_448 = images_448.float()
            labels = labels.long()
            
            
            # === 多视角STN变换和特征提取 ===
            try:
                # 多视角STN前向传播（使用448x448标准化图像）
                # 前10个批次获取中间结果用于可视化
                if batch_idx < 10:
                    result = stn_model(images_448, mode='test')  # 测试模式，返回可视化数据
                    
                    # STN模型返回元组: (fused_features, vis_data)
                    if isinstance(result, tuple) and len(result) == 2:
                        batch_transformed_features, vis_data = result
                        
                        # 可视化前10个批次的STN变换结果
                        print(f"🎨 开始可视化第 {batch_idx + 1}/10 个批次的STN变换结果...")
                        try:
                            # 确定保存路径（包含标签以避免覆盖）
                            save_base = "visualizations/test_stn_transforms"
                            if model_tag:
                                save_base = os.path.join(save_base, model_tag)

                            # 限制每个batch可视化的样本数，避免生成过多图片
                            # 如果想要可视化整个batch，可以将max_samples设为None
                            visualize_preprocessed_vs_stn(
                                preprocessed_images=images_448,  # [B, 3, 448, 448] 预处理后的输入图像
                                transformed_images=vis_data['view_images'],  # [B, N, 3, 224, 224] STN变换后的图像
                                dataset_name=dataset_name,
                                config=config,  # 传入完整配置信息
                                save_dir=save_base,
                                batch_idx=batch_idx,
                                max_samples=None, # 可视化整个batch
                                theta_matrices=vis_data.get('theta_matrices', None),  # [B, N, 2, 3] 变换矩阵
                                position_params=vis_data.get('position_params', None)  # [B, 2*N] 位置参数
                            )
                        except Exception as vis_e:
                            print(f"⚠️ 可视化失败，但继续测试: {vis_e}")
                    else:
                        # 如果返回格式不正确
                        print(f"⚠️ 第 {batch_idx} 批次未获取到正确的返回格式，跳过可视化")
                        batch_transformed_features = result  # 直接使用返回值
                else:
                    # 其他批次使用训练模式（不保存图像，只返回特征）
                    batch_transformed_features, _ = stn_model(images_448, mode='train')  # [B, 512]
                
                #print(f"🔧 多视角STN变换后的特征形状: {batch_transformed_features.shape}")
                
                # 收集特征和标签
                all_image_features.append(batch_transformed_features)
                all_targets.append(labels)
                
            except Exception as e:
                print(f"❌ 批量多视角STN测试失败 - batch {batch_idx}: {e}")
                print("程序终止，请检查模型配置和数据")
                raise e
    
    # 第二步：合并所有特征（类似WCA的image_features）
    print("🔄 第二阶段：合并所有特征...")
    all_image_features = torch.cat(all_image_features, dim=0)  # [total_samples, 512]
    print(f"🔧 合并后的特征形状: {all_image_features.shape}")

    all_targets = torch.cat(all_targets, dim=0)  # [total_samples]
    print(f"🔧 合并后的标签形状: {all_targets.shape}")

    total_samples = all_image_features.size(0)
    

    
    print(f"📊 收集完成：{total_samples} 张图片的STN特征")
    print(f"   图像特征形状: {all_image_features.shape}")
    print(f"   文本特征形状: {precomputed_text_features.shape}")
    
    # 调试：检查特征的数值范围
    print(f"\n🔍 特征数值检查:")
    print(f"   图像特征范围: [{all_image_features.min():.4f}, {all_image_features.max():.4f}]")
    print(f"   图像特征均值: {all_image_features.mean():.4f}, 标准差: {all_image_features.std():.4f}")
    print(f"   文本特征范围: [{precomputed_text_features.min():.4f}, {precomputed_text_features.max():.4f}]")
    print(f"   文本特征均值: {precomputed_text_features.mean():.4f}, 标准差: {precomputed_text_features.std():.4f}")
    
    # 检查前几个样本的图像特征是否相同
    print(f"\n🔍 检查图像特征多样性:")
    for i in range(min(3, total_samples)):
        feat_norm = all_image_features[i].norm().item()
        feat_mean = all_image_features[i].mean().item()
        print(f"   样本{i}: L2范数={feat_norm:.4f}, 均值={feat_mean:.4f}, 前5维={all_image_features[i, :5].tolist()}")
    
    # 第三步：一次性计算相似度（类似WCA的logits计算）
    print("\n🔄 第三阶段：计算最终相似度和准确率...")
    logits = all_image_features @ precomputed_text_features  # [total_samples, num_classes].
    print(f"🔧 计算后的logits形状: {logits.shape}")
    print(f"🔧 logits数值范围: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"🔧 logits均值: {logits.mean():.4f}, 标准差: {logits.std():.4f}")
    
    # 调试：查看前几个样本的预测
    print(f"\n🔍 前5个样本的预测检查:")
    for i in range(min(5, total_samples)):
        pred_class = logits[i].argmax().item()
        true_class = all_targets[i].item()
        max_logit = logits[i].max().item()
        top5_preds = logits[i].topk(5).indices.tolist()
        print(f"   样本{i}: 预测={pred_class}, 真实={true_class}, 最大logit={max_logit:.4f}, Top5={top5_preds}, 匹配={'✓' if pred_class == true_class else '✗'}")
    
    # 额外调试：检查所有预测的分布
    all_preds = logits.argmax(dim=1)
    unique_preds, counts = torch.unique(all_preds, return_counts=True)
    print(f"\n🔍 预测类别分布 (前10个最常见的):")
    sorted_indices = counts.argsort(descending=True)[:10]
    for idx in sorted_indices:
        pred_class = unique_preds[idx].item()
        count = counts[idx].item()
        print(f"   类别{pred_class}: {count}次 ({count/total_samples*100:.2f}%)")
    
    # 检查真实标签分布
    unique_targets, target_counts = torch.unique(all_targets, return_counts=True)
    print(f"\n🔍 真实标签分布:")
    print(f"   唯一类别数: {len(unique_targets)}")
    print(f"   标签范围: [{unique_targets.min().item()}, {unique_targets.max().item()}]")
    print(f"   前10个类别: {unique_targets[:10].tolist()}")
    
    # 使用WCA的accuracy函数计算准确率
    stn_accuracy = accuracy(logits, all_targets, total_samples, dataset_name)
    
    print(f"\n✅ STN测试完成，共测试 {total_samples} 张图片")
    print(f"🎯 STN Top-1 准确率: {stn_accuracy:.2f}%")
    
    return stn_accuracy


 




def main():
    """
    主测试函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='STN模型两阶段测试系统')
    parser.add_argument('--dataset_name', type=str, default='cub', 
                       choices=['cub', 'imagenet', 'food101', 'oxford_pets', 'dtd', 
                               'fgvc-aircraft', 'stanford_cars', 'stanford_dogs', 'flowers102',
                               'eurosat', 'places365', 'sun397',
                               'imagenetv2', 'imagenet-v2', 'imagenet-r', 'imagenetr',
                               'imagenet-s', 'imagenets', 'imagenet-sketch',
                               'imagenet-a', 'imageneta'],
                       help='数据集名称 (包括ImageNet领域泛化变体)')
    
    parser.add_argument('--source_dataset', type=str, default=None,
                       help='模型训练的源数据集名称（用于领域泛化测试）\n'
                            '例如：测试ImageNet-V2时使用ImageNet训练的模型\n'
                            '用法: --dataset_name imagenetv2 --source_dataset imagenet')

    parser.add_argument('--batch_size', type=int, default=32,
                       help='测试批次大小')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机数种子，用于确保测试结果可重复 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置默认设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 设置随机数种子，确保测试结果可重复
    def set_seed(seed):
        """设置所有相关库的随机数种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # 确保CuDNN的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed(args.seed)
    
    print("=== 多视角STN模型测试系统 ===")
    print(f"数据集: {args.dataset_name}")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"随机数种子: {args.seed}")

    
    try:
        # 步骤1: 加载STN训练配置文件
        print("\n📥 步骤1: 加载STN训练配置文件...")
        config_path = f"STN-Config/{args.dataset_name}.yaml"
        
        if not os.path.exists(config_path):
            print(f"⚠️ 配置文件不存在: {config_path}")
            
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 从配置文件获取model_size
        model_size = config['model_size']
        stn_config = config['stn_config']
        
        print(f"✅ STN配置加载完成: {config_path}")
        print(f"   🤖 CLIP模型: {model_size}")


        
        # 步骤2: 加载CLIP模型
        print("\n📥 步骤2: 加载CLIP模型...")
        clip_model, preprocess = clip.load(model_size, device=args.device)
        
        # === 转换CLIP模型为float32精度 ===
        print("🔧 转换CLIP模型精度: float16 → float32")
        clip_model = clip_model.float()  # 转换为float32
        
        # 验证转换结果
        clip_weight_dtype = clip_model.visual.conv1.weight.dtype
        print(f"✅ CLIP模型加载完成，当前精度: {clip_weight_dtype}")
        
        # 步骤3: 创建多视角STN模型
        print(f"\n📥 步骤3: 创建多视角STN模型...")
        from stn.multi_view_stn import MultiViewSTNModel
        
        num_views = stn_config.get('num_views', 4)  # 从配置中读取视角数量  
        #创建的模型是根据传入的配置文件等参数控制的
        stn_model = MultiViewSTNModel(clip_model, stn_config, num_views=num_views).to(args.device)
        print(f"✅ 多视角STN模型创建完成 ({num_views}个视角)")

        # 步骤4: 加载测试数据集
        print(f"\n📥 步骤4: 加载测试数据集...")
        
        # 从配置文件获取数据路径
        data_path = config['data_path']
        
        # 创建测试集数据加载器 - 使用新的统一接口
        from data_preprocess import load_multi_view_dataset
        
        print(f"\n🔥 加载测试集...")
        test_dataloader = load_multi_view_dataset(
            dataset_name=args.dataset_name,
            data_path=data_path,
            split='test',  # 测试集
            batch_size=args.batch_size,
            num_workers=4,
            target_size=448,  # 目标裁剪尺寸
            scale_short_edge=512,   # 短边缩放尺寸
            flip_prob=0.0,    # 测试时不使用翻转
            center_crop=True  # 测试时使用中心裁剪
        )
        
        print(f"✅ 测试数据集加载完成 - 新架构:")
        print(f"    📈 测试样本数: {len(test_dataloader.dataset)}")
        print(f"    📊 测试批次数: {len(test_dataloader)}")
        print(f"    🏗️ 架构: 基础数据集 + MultiViewDataPreprocessor transform")
        

        # 步骤5: 加载预计算的文本特征
        print(f"\n📥 步骤5: 加载预计算文本特征...")
        
        # 确定文本特征来源数据集（领域泛化模式）
        if args.source_dataset:
            text_feature_dataset = args.source_dataset  # 使用源数据集的文本特征
            print(f"🔄 领域泛化模式: 使用 {args.source_dataset} 的文本特征测试 {args.dataset_name}")
        else:
            text_feature_dataset = args.dataset_name
        
        # 构建预计算文件路径
        model_name = model_size.replace('/', '_')
        text_features_path = f"text_features/{text_feature_dataset}_{model_name}.pt"
        
        try:
            # 加载预计算的文本特征
            precomputed_text_features = torch.load(text_features_path, map_location=args.device)
            print(f"✅ 成功加载预计算文本特征: {text_features_path}")
            print(f"   特征形状: {precomputed_text_features.shape}")
            print(f"   数据类型: {precomputed_text_features.dtype}")
            
            # === 转换文本特征为float32精度 ===
            original_dtype = precomputed_text_features.dtype
            precomputed_text_features = precomputed_text_features.float()  # 转换为float32
            print(f"🔧 文本特征精度转换: {original_dtype} → {precomputed_text_features.dtype}")
            
            # 确保特征在正确的设备上
            precomputed_text_features = precomputed_text_features.to(args.device)
            print(f"   ✅ 文本特征已移动到指定设备")
            
        except FileNotFoundError:
            print(f"❌ 未找到预计算文本特征文件: {text_features_path}")
            return
        except Exception as e:
            print(f"❌ 加载文本特征时发生错误: {e}")
            return


        # 步骤6: 构建候选模型文件名（最佳损失与最佳准确率）
        print(f"\n📥 步骤6: 准备候选模型文件...")

        # 从配置文件中提取构建路径所需的参数
        num_views = stn_config.get('num_views', 4)
        fusion_mode = stn_config.get('fusion_mode', 'concat')
        hidden_dim = stn_config.get('hidden_dim', 256)
        
        # 读取损失权重配置（用于文件名生成）
        logits_temp = stn_config.get('logits_temp', 0.07)
        classification_weight = stn_config.get('classification_weight', 1.0)
        contrastive_weight = stn_config.get('contrastive_weight', 0.0)
        decorrelation_weight = stn_config.get('decorrelation_weight', 0.0)
        adaptive_weight = stn_config.get('adaptive_weight', 0.0)
        
        # 构建完整的模型文件路径（与训练时保持一致），仅使用数据集二级目录
        model_name_clean = model_size.replace('/', '_')  # ViT-B/32 -> ViT-B_32
        
        # 构建基础架构部分
        model_name_suffix = f"views{num_views}_{fusion_mode}_dim{hidden_dim}"
        
        # 构建损失权重部分
        loss_parts = []
        if logits_temp is not None:
            loss_parts.append(f"temp{logits_temp}")
        if classification_weight > 0:
            loss_parts.append(f"cls{classification_weight}")
        if contrastive_weight > 0:
            loss_parts.append(f"con{contrastive_weight}")
        if decorrelation_weight > 0:
            loss_parts.append(f"dec{decorrelation_weight}")
        if adaptive_weight > 0:
            loss_parts.append(f"adp{adaptive_weight}")
        
        loss_suffix = "_".join(loss_parts)

        # 确定模型检查点目录
        # 如果指定了source_dataset，则从源数据集目录加载模型（用于领域泛化测试）
        if args.source_dataset:
            ckpt_dir_new = os.path.join('checkpoints', args.source_dataset)
            model_dataset_name = args.source_dataset  # 文件名中使用源数据集名称
            print(f"\n🔄 领域泛化测试模式:")
            print(f"   源数据集（模型训练）: {args.source_dataset}")
            print(f"   目标数据集（测试）: {args.dataset_name}")
        else:
            ckpt_dir_new = os.path.join('checkpoints', args.dataset_name)
            model_dataset_name = args.dataset_name
        
        # 组合所有部分（使用模型训练时的数据集名称）
        base_filename = f"multi_view_stn_{model_dataset_name}_{model_name_clean}_{model_name_suffix}"
        if loss_suffix:
            base_filename = f"{base_filename}_{loss_suffix}"

        # 新命名（仅保留带后缀的最佳模型）
        best_loss_path_new = os.path.join(ckpt_dir_new, f"{base_filename}_best_loss.pth")
        best_acc_path_new = os.path.join(ckpt_dir_new, f"{base_filename}_best_acc.pth")
        
        print("📁 模型文件（仅新目录，且仅带后缀）：")
        print(f"   最佳损失：{best_loss_path_new}")
        print(f"   最佳准确：{best_acc_path_new}")
        print(f"   📊 架构参数: views={num_views}, fusion={fusion_mode}, dim={hidden_dim}")
        print(f"   🎯 损失权重: temp={logits_temp}, cls={classification_weight}, con={contrastive_weight}, dec={decorrelation_weight}, adp={adaptive_weight}")

        def eval_one(ckpt_path: str, tag: str):
            try:
                state_dict = torch.load(ckpt_path, map_location=args.device)
                print(f"✅ 成功加载模型文件({tag}): {ckpt_path}")
                stn_model.load_state_dict(state_dict, strict=True)
            except FileNotFoundError:
                print(f"❌ 未找到模型文件({tag}): {ckpt_path}")
                return None
            except Exception as e:
                print(f"❌ 模型权重加载失败({tag}): {e}")
                return None

            stn_model.to(args.device)
            model_float = stn_model.float()

            print(f"\n🎯 开始测试（{tag}）...")
            
            # 使用一个标志或状态来管理可视化，确保每个模型测试时都独立进行可视化
            # 将 tag 转为文件夹名称 (best_loss 或 best_acc)
            folder_tag = 'best_loss' if '损失' in tag or 'loss' in tag.lower() else 'best_acc'
            
            acc = stn_precise_testing(
                model_float, test_dataloader, args.device,
                args.dataset_name, precomputed_text_features, config,
                model_tag=folder_tag
            )
            print(f"\n🎉 {tag} 模型测试完成！Top-1 准确率: {acc:.2f}%")
            return acc

        results = {}
        # 仅在新目录查找，且仅接受带 _best_loss 后缀的文件
        if os.path.exists(best_loss_path_new):
            results['best_loss'] = eval_one(best_loss_path_new, '最佳损失')
        else:
            print("❌ 未找到'最佳损失'模型文件（仅查找 checkpoints/{dataset}/..._best_loss.pth）")

        if os.path.exists(best_acc_path_new):
            results['best_acc'] = eval_one(best_acc_path_new, '最佳准确')
        else:
            print("❌ 未找到'最佳准确'模型文件（仅查找新目录）")

        print("\n=== 测试汇总 ===")
        for k, v in results.items():
            if v is not None:
                print(f"{k}: {v:.2f}%")
            else:
                print(f"{k}: 未测试（文件缺失或加载失败）")
        print(f"   📊 架构参数: views={num_views}, fusion={fusion_mode}, dim={hidden_dim}")
        print(f"   🎯 损失权重: temp={logits_temp}, cls={classification_weight}, con={contrastive_weight}, dec={decorrelation_weight}, adp={adaptive_weight}")
        return
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()