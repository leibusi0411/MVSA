"""
多视角STN训练模块

Multi-View STN Training Module

功能特点：
1. 支持多视角STN模型训练
2. 综合损失函数（分类+多样性+一致性）
3. 动态视角数量配置
4. 可视化多视角变换结果
5. 与原有train_simple.py兼容的接口

Multi-view STN training with comprehensive loss functions.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler  # 禁用混合精度训练
from tqdm import tqdm
import os
import json

# === 启用梯度异常检测 ===
torch.autograd.set_detect_anomaly(True)
print("🔍 多视角STN训练模块: 梯度异常检测已启用")

# 项目内部模块导入
from clip import clip
from stn.loss_multi import MultiViewSTNLoss
from test_multi_stn import visualize_preprocessed_vs_stn
from text_aggregation import WeightedTextAggregator

# 导入数据集工具函数
from dataset_utils import load_classes, load_text_prompts


def compute_and_save_text_features(clip_model, dataset_name, model_size, device, 
                                 text_scale=7.39, use_weighted_aggregation=True):
    """
    计算并保存文本特征
    
    Args:
        clip_model: CLIP模型
        dataset_name: 数据集名称
        model_size: 模型大小
        device: 计算设备
        text_scale: 文本聚合温度参数
        use_weighted_aggregation: 是否使用加权聚合
        
    Returns:
        torch.Tensor: 文本特征张量 [feature_dim, num_classes]
    """
    print(f"\n🔄 开始计算文本特征...")
    
    # 1. 加载类别名称
    class_names = load_classes(dataset_name)
    if class_names is None:
        return None
    
    # 2. 加载文本描述
    text_prompts_dict = load_text_prompts(dataset_name)
    if text_prompts_dict is None:
        print(f"❌ 无法加载文本描述，使用简单模板")
        # 使用简单模板作为回退
        text_prompts_dict = {name: [f"a photo of a {name}."] for name in class_names}
    
    # 3. 创建文本聚合器
    aggregator = WeightedTextAggregator(text_scale=text_scale, device=device)
    
    # 4. 计算聚合后的文本特征
    text_features = aggregator.aggregate_class_descriptions(
        model=clip_model,
        class_names=class_names,
        descriptions_dict=text_prompts_dict,
        use_weighted_aggregation=use_weighted_aggregation,
        show_progress=True
    )
    
    # 5. 转置以匹配预期格式 [feature_dim, num_classes]
    text_features = text_features.T
    
    # 6. 保存到文件
    model_name = model_size.replace('/', '_')
    text_features_path = f"text_features/{dataset_name}_{model_name}.pt"
    
    # 确保目录存在
    os.makedirs("text_features", exist_ok=True)
    
    try:
        torch.save(text_features, text_features_path)
        print(f"✅ 文本特征已保存: {text_features_path}")
        print(f"   特征形状: {text_features.shape}")
        print(f"   数据类型: {text_features.dtype}")
    except Exception as e:
        print(f"❌ 保存文本特征失败: {e}")
        return None
    
    return text_features


class WarmupScheduler:
    """
    学习率预热调度器（专用于CosineAnnealingLR）
    
    在训练初期使用较小的学习率，逐渐增加到目标学习率，
    然后切换到CosineAnnealingLR调度器进行后续调度。
    """
    
    def __init__(self, optimizer, warmup_epochs, base_scheduler, warmup_factor=0.1):
        """
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: 预热轮数
            base_scheduler: CosineAnnealingLR调度器（预热后使用）
            warmup_factor: 预热起始学习率因子（相对于基础学习率）
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_factor = warmup_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
        print(f"🔥 学习率预热调度器初始化:")
        print(f"   预热轮数: {warmup_epochs}")
        print(f"   预热因子: {warmup_factor}")
        print(f"   基础学习率: {self.base_lrs}")
        print(f"   主调度器: CosineAnnealingLR")
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            warmup_progress = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.base_lrs[i] * (self.warmup_factor + (1 - self.warmup_factor) * warmup_progress)
                param_group['lr'] = lr
            
            if self.current_epoch == 0 or (self.current_epoch + 1) % 5 == 0:
                print(f"🔥 预热进度: {self.current_epoch+1}/{self.warmup_epochs}, "
                      f"当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
        else:
            # 预热结束，使用CosineAnnealingLR调度器
            if self.base_scheduler:
                # 调整CosineAnnealingLR的epoch计数
                adjusted_epoch = self.current_epoch - self.warmup_epochs
                self.base_scheduler.last_epoch = adjusted_epoch - 1  # 设置正确的epoch计数
                self.base_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """保存调度器状态"""
        state = {
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'base_lrs': self.base_lrs,
            'current_epoch': self.current_epoch,
        }
        # 保存base_scheduler的状态
        if self.base_scheduler is not None:
            state['base_scheduler'] = self.base_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """恢复调度器状态"""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.base_lrs = state_dict['base_lrs']
        self.current_epoch = state_dict['current_epoch']
        
        # 恢复base_scheduler的状态
        if self.base_scheduler is not None and 'base_scheduler' in state_dict:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def train_multi_view_stn_model(model, dataloader, optimizer, criterion, device, epoch, 
                              dataset_name, all_classes_text_features=None, 
                              max_grad_norm=1.0, grad_norm_type=2.0, config=None):
    """
    训练多视角STN模型一个epoch
    
    核心流程：
    1. 多个STN分支并行预测不同视角变换
    2. 融合多视角特征
    3. 计算综合损失（分类+多样性）
    4. 只训练STN组件，CLIP保持冻结
    
    Args:
        model: 多视角STN模型实例
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 多视角损失函数
        device: 计算设备
        epoch: 当前轮数
        dataset_name: 数据集名称
        all_classes_text_features: 预计算的文本特征 [D, C]
        max_grad_norm: 梯度裁剪阈值 (默认1.0)
        grad_norm_type: 梯度范数类型 (默认2.0，即L2范数)
        
    Returns:
        tuple: (平均损失, 平均准确率)
    """
    model.train()
    
    # === Epoch开始前清理GPU缓存 ===
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 统计变量
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # === 数据预处理 ===
        
        #将cpu数据转移到gpu
        images_448 = images.to(device, non_blocking=True)  # [B, 3, 448, 448]
        
        labels = labels.to(device, non_blocking=True)
        
        # === 前向传播 ===
        optimizer.zero_grad()
        
        # === Float32精度管理 ===
        # 强制使用float32精度，确保数值稳定性
        images_448 = images_448.float()
        labels = labels.long()
        

        # 获取中间结果用于新增损失函数
        # 训练和验证阶段统一使用 train 模式
        fused_features, view_features = model(images_448, mode='train')
        
        # 如果不需要特殊损失，view_features 会被忽略（但仍然会计算）
        if not ((hasattr(criterion, 'decorrelation_weight') and criterion.decorrelation_weight > 0) or \
                (hasattr(criterion, 'adaptive_weight') and criterion.adaptive_weight > 0)):
            view_features = None  # 不传递给损失函数

       
        # === 多视角STN前向传播 (Float32) ===   训练时检查初始化是否正确
        # 第一个批次时获取中间结果用于可视化
        # if batch_idx == 0 and epoch == 0:
        #     # 获取详细的中间结果用于可视化
        #     fused_features, intermediate_results = model(images_448, return_intermediate=True)
        #     view_params = intermediate_results.get('position_params', None)
        #     transformed_images = intermediate_results.get('view_images', None)
        #     theta_matrices = intermediate_results.get('theta_matrices', None)
            
        #     # 调用可视化函数
        #     if transformed_images is not None:
        #         print(f"🎨 第一个批次可视化: 输入图像 {images_448.shape}, 变换后图像 {transformed_images.shape}")
        #         try:
        #             visualize_preprocessed_vs_stn(
        #                 preprocessed_images=images_448[:8],  # 只可视化前8个样本
        #                 transformed_images=transformed_images[:8],
        #                 dataset_name=dataset_name,
        #                 config=config,
        #                 save_dir="visualizations/training_init_transforms",
        #                 batch_idx=0,
        #                 max_samples=8,
        #                 theta_matrices=theta_matrices[:8] if theta_matrices is not None else None,
        #                 position_params=view_params[:8] if view_params is not None else None
        #             )
        #             print("✅ 初始化变换可视化完成")
        #         except Exception as e:
        #             print(f"⚠️ 可视化失败: {e}")
        # else:
        #     # 常规训练时不返回中间结果
        #     fused_features = model(images_448, return_intermediate=False)
        #     view_params = None  # 训练时不需要中间结果
        
        # === 损失计算 ===
        if all_classes_text_features is None:
            raise RuntimeError("all_classes_text_features 不能为 None")
        
        # 使用预计算的文本特征
        text_features = all_classes_text_features  # [D, C]
        
        # 计算相似度矩阵（使用温度参数进行数值稳定化）
        # 从配置文件读取温度参数，如果没有配置则使用默认值0.07
        temperature = config['stn_config'].get('logits_temp', 0.07) if config else 0.07
        similarity = (fused_features @ text_features) / temperature  # [B, C]
        
        if batch_idx == 0:
            print(f"🎯 多视角前向: fused_features {fused_features.shape}, similarity {similarity.shape}")
            print(f"📊 相似度统计: min={similarity.min().item():.3f}, max={similarity.max().item():.3f}, mean={similarity.mean().item():.3f}")
            print(f"🔧 数据类型: similarity={similarity.dtype}, fused_features={fused_features.dtype}")
            if view_features is not None:
                print(f"🔍 多视角特征: {view_features.shape}")
        
        # 计算多视角损失（损失函数内部自动处理FP32精度切换）
        # 传递所有必要参数给损失函数，并获取详细损失信息
        loss, loss_details = criterion(
            labels=labels, 
            similarity_or_logits=similarity,  #融合特征与文本特征的相似度
            view_features=view_features,  #多视角特征
            text_features=text_features
        )
        
        # 显示损失精度信息（仅第一个batch）
        if batch_idx == 0:
            print(f"🔧 损失计算: loss={loss.item():.6f}, dtype={loss.dtype}")
            print(f"📊 损失详情: {loss_details}")
        
        # === 数值稳定性检查 ===
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"⚠️  警告: 损失值异常 - loss: {loss.item()}")
            print(f"📊 相似度矩阵统计: min={similarity.min().item():.6f}, max={similarity.max().item():.6f}")
            print(f"📊 融合特征统计: min={fused_features.min().item():.6f}, max={fused_features.max().item():.6f}")
        
        # === 相似度数值稳定性检查 ===
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            print(f"❌ 相似度矩阵包含NaN/Inf值")
            print(f"📊 融合特征范数: {fused_features.norm(dim=1).mean().item():.6f}")
            print(f"📊 文本特征范数: {text_features.norm(dim=0).mean().item():.6f}")
            print(f"⚠️  跳过当前batch，继续训练")
            # 应急处理：跳过这个batch（优化器不更新）
            continue
        
        # === 反向传播 (Float32) ===
        try:
            # 标准float32精度反向传播
            loss.backward()
        except RuntimeError as e:
            print(f"❌ 反向传播异常: {e}")
            print(f"📊 损失值: {loss.item()}")
            print(f"📊 损失梯度: {loss.requires_grad}")
            raise
        
        # === 计算准确率 (在删除similarity之前) ===
        with torch.no_grad():
            predictions = similarity.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            batch_accuracy = correct / labels.size(0)
        
        # 内存优化：反向传播后立即释放不需要的中间变量
        del fused_features, similarity
        if view_features is not None:
            del view_features
        
        # === 增强梯度裁剪 ===
        
        # 收集所有可训练参数
        all_params = []
        if hasattr(model, 'localization_network'):
            all_params.extend(list(model.localization_network.parameters()))
        if hasattr(model, 'fusion_module'):
            fusion_params = list(model.fusion_module.parameters())
            if fusion_params:
                all_params.extend(fusion_params)
        
        if all_params:
            # Float32标准精度下的梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params, 
                max_norm=max_grad_norm, 
                norm_type=grad_norm_type
            )
            
            # 记录梯度范数用于监控
            if batch_idx == 0 or torch.isnan(grad_norm) or grad_norm > max_grad_norm * 2:
                print(f"🔧 批次{batch_idx}: 梯度范数={grad_norm:.6f}, 裁剪阈值={max_grad_norm}")
        else:
            grad_norm = 0.0
        
        # 梯度监控（每50个批次，增强异常检测）
        if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
            total_grad_norm = 0
            param_count = 0
            nan_params = []
            max_param_grad_norm = 0  # 重命名以避免与函数参数冲突
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    total_grad_norm += param_grad_norm
                    param_count += 1
                    max_param_grad_norm = max(max_param_grad_norm, param_grad_norm)
                    
                    # 检查NaN/Inf梯度
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_params.append(name)
            
            avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
            print(f"🔍 梯度监控 [批次{batch_idx+1}]: 平均={avg_grad_norm:.6f}, 最大={max_param_grad_norm:.6f}, 参数数={param_count}")
            
            # 异常检测
            if max_param_grad_norm > 10.0:
                print(f"⚠️  梯度爆炸警告: 最大梯度范数 {max_param_grad_norm:.6f}")
            elif avg_grad_norm < 1e-8:
                print(f"⚠️  梯度消失警告: 平均梯度范数 {avg_grad_norm:.6f}")
            
            if nan_params:
                print(f"❌ 检测到异常梯度的参数: {nan_params[:3]}{'...' if len(nan_params) > 3 else ''}")
        
        # === 优化器步骤 (Float32) ===
        optimizer.step()
        
        # === 统计更新 ===
        total_loss += loss.item()
        
        # 更新进度条 - 显示详细损失信息
        current_lr = optimizer.param_groups[0]['lr']
        progress_dict = {
            'Total': f'{loss_details["total"]:.3f}',
            'Acc': f'{batch_accuracy:.3f}',
            'LR': f'{current_lr:.1e}'
        }
        
        # 添加启用的损失项（只显示加权后的最终值）
        if loss_details.get("classification_weighted", 0) > 0:
            progress_dict['Cls'] = f'{loss_details["classification_weighted"]:.3f}'
        
        if loss_details.get("contrastive_weighted", 0) > 0:
            progress_dict['Con'] = f'{loss_details["contrastive_weighted"]:.3f}'
        
        if loss_details.get("decorrelation_weighted", 0) > 0:
            progress_dict['Dec'] = f'{loss_details["decorrelation_weighted"]:.3f}'
        
        if loss_details.get("adaptive_weighted", 0) > 0:
            progress_dict['Adp'] = f'{loss_details["adaptive_weighted"]:.3f}'
        
        #将准备好的信息字典设置到进度条的后缀显示区域
        pbar.set_postfix(progress_dict)
        
        # 内存优化：每个batch结束后清理
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算平均值
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy

#验证函数
def validate_multi_view_stn_model(model, val_dataloader, criterion, device, all_classes_text_features, config=None):
    """
    验证多视角STN模型
    
    计算完整的验证损失，包括：
    - 分类损失（基础交叉熵）
    - 特征去相关损失（如果启用）
    - 自适应分类损失（如果启用）
    
    Args:
        model: 多视角STN模型实例
        val_dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        all_classes_text_features: 预计算的文本特征
        config: 配置字典（用于获取温度参数等）
        
    Returns:
        tuple: (平均验证损失, 平均验证准确率)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            # 数据预处理
            images_448 = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # === Float32精度管理 ===
            images_448 = images_448.float()
            labels = labels.long()
            
            # === 前向传播 (Float32) ===
            # 获取融合特征和多视角特征（用于损失计算）
            fused_features, view_features = model(images_448, mode='train')
            
            # 内存优化：立即释放输入图像
            del images_448
            
            text_features = all_classes_text_features
            # 使用温度参数进行数值稳定化
            # 从配置文件读取温度参数，如果没有配置则使用默认值0.07
            temperature = config['stn_config'].get('logits_temp', 0.07) if config else 0.07
            similarity = (fused_features @ text_features) / temperature
            
            # === 损失计算 (Float32) ===
            # 验证时计算完整损失（包括所有启用的损失项）
            loss, loss_details = criterion(
                labels=labels, 
                similarity_or_logits=similarity, 
                view_features=view_features,  # 传递多视角特征
                text_features=text_features
            )
            
            # 统计
            total_loss += loss.item()
            predictions = similarity.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            # 内存优化：释放中间变量
            del fused_features, view_features, similarity, loss
    
    avg_val_loss = total_loss / len(val_dataloader)
    avg_val_accuracy = total_correct / total_samples
    
    return avg_val_loss, avg_val_accuracy


def train_multi_view_stn_full(stn_model, train_dataloader, val_dataloader, config, device, dataset_name, model_size):
    """
    完整的多视角STN训练流程
    
    Args:
        stn_model: 多视角STN模型实例
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        config: 配置字典
        device: 计算设备
        dataset_name: 数据集名称
        model_size: 模型大小标识
        
    Returns:
        str: 保存的模型路径
    """
    print("=== 开始多视角STN训练 ===")
    
    # === GPU缓存清理 ===
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 训练前GPU缓存已清理")
    
    # 从配置中提取STN参数用于文件名
    # 读取STN配置参数
    stn_config = config.get('stn_config', {})
    num_views = stn_config.get('num_views', 4)
    fusion_mode = stn_config.get('fusion_mode', 'simple')
    hidden_dim = stn_config.get('hidden_dim', 512)
    
    # 读取损失权重配置（用于文件名生成）
    logits_temp = stn_config.get('logits_temp', 0.07)
    classification_weight = stn_config.get('classification_weight', 1.0)
    contrastive_weight = stn_config.get('contrastive_weight', 0.0)
    decorrelation_weight = stn_config.get('decorrelation_weight', 0.0)
    adaptive_weight = stn_config.get('adaptive_weight', 0.0)
    
    # 生成包含STN参数和损失权重的模型文件名
    model_name_suffix = f"views{num_views}_{fusion_mode}_dim{hidden_dim}"
    
    # 添加损失权重信息到文件名
    loss_suffix = f"temp{logits_temp}"
    if classification_weight > 0:
        loss_suffix += f"_cls{classification_weight}"
    if contrastive_weight > 0:
        loss_suffix += f"_con{contrastive_weight}"
    if decorrelation_weight > 0:
        loss_suffix += f"_dec{decorrelation_weight}"
    if adaptive_weight > 0:
        loss_suffix += f"_adp{adaptive_weight}"
    
    base_model_name = f'multi_view_stn_{dataset_name}_{model_size.replace("/", "_")}_{model_name_suffix}_{loss_suffix}'
    print(f"📁 模型保存前缀: {base_model_name}")
    
    # 设置训练环境
    criterion, optimizer, scheduler = setup_multi_view_training(stn_model, config, device)
    
    # === 禁用混合精度训练，使用float32 ===
    print("🔧 使用标准float32精度训练 (混合精度已禁用)")
    
    # 加载或计算文本特征
    print("🔄 加载预计算文本特征...")
    model_name = model_size.replace('/', '_')
    text_features_path = f"text_features/{dataset_name}_{model_name}.pt"
    
    try:
        all_classes_text_features = torch.load(text_features_path, map_location=device)
        print(f"✅ 成功加载预计算文本特征: {text_features_path}")
        print(f"   特征形状: {all_classes_text_features.shape}")
        
        # === 转换文本特征为float32精度 ===
        original_dtype = all_classes_text_features.dtype
        all_classes_text_features = all_classes_text_features.float()  # 转换为float32
        print(f"🔧 文本特征精度转换: {original_dtype} → {all_classes_text_features.dtype}")
        
    except FileNotFoundError:
        print(f"⚠️ 未找到预计算文本特征: {text_features_path}")
        print("🔄 开始实时计算文本特征...")
        
        # 实时计算文本特征
        all_classes_text_features = compute_and_save_text_features(
            clip_model=stn_model.clip_model,  # 使用STN模型中的CLIP
            dataset_name=dataset_name,
            model_size=model_size,
            device=device,
            text_scale=7.39,  # WCA标准温度参数
            use_weighted_aggregation=True
        )
        
        if all_classes_text_features is None:
            print("❌ 文本特征计算失败")
            return None
            
        print(f"✅ 文本特征计算完成")
    
    except Exception as e:
        print(f"❌ 文本特征处理失败: {e}")
        return None
    
    # 确保文本特征在正确设备上且不需要梯度
    all_classes_text_features = all_classes_text_features.to(device)
    all_classes_text_features.requires_grad_(False)
    
    # 训练循环
    print(f"开始多视角训练，总轮数: {config['training']['epochs']}")
    
    # 早停机制 - 基于验证损失
    # === 早停机制配置 ===
    best_val_loss = float('inf')
    best_loss_acc = 0.0
    best_loss_epoch = 0
    best_val_accuracy = 0.0
    best_acc_epoch = 0
    patience_counter = 0
    # 从配置文件读取耐心值，默认5
    patience = config['training'].get('patience', 5)
    best_model_state = None
    best_epoch = 0
    
    print(f"📊 早停机制配置:")
    print(f"   观察指标: 验证集损失")
    print(f"   耐心值: {patience} 个epoch (来自配置文件)")
    print(f"   触发条件: 验证损失连续{patience}个epoch不下降")
    
    print(f"🔧 梯度裁剪配置:")
    print(f"   最大梯度范数: {config['training'].get('max_grad_norm', 1.0)}")
    print(f"   范数类型: L{int(config['training'].get('grad_norm_type', 2.0))} 范数")
    print(f"   位置: loss.backward() 之后, optimizer.step() 之前")
    
    # 将检查点按数据集分类保存到二级目录，例如 checkpoints/cub/
    ckpt_dir = os.path.join('checkpoints', dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_best_loss = os.path.join(ckpt_dir, f"{base_model_name}_best_loss.pth")
    ckpt_best_acc = os.path.join(ckpt_dir, f"{base_model_name}_best_acc.pth")
    ckpt_latest = os.path.join(ckpt_dir, f"{base_model_name}_latest.pth")  # 最新检查点
    
    start_epoch = 0  # 起始epoch
    
    # === 断点续训：检查是否存在最新检查点 ===
    if os.path.exists(ckpt_latest):
        print(f"\n🔄 发现检查点，尝试恢复训练: {ckpt_latest}")
        
        try:
            checkpoint = torch.load(ckpt_latest, map_location=device)
            
            # 恢复模型权重
            stn_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 恢复优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复调度器状态
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练进度
            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_loss_acc = checkpoint.get('best_loss_acc', 0.0)
            best_loss_epoch = checkpoint.get('best_loss_epoch', 0)
            best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            best_acc_epoch = checkpoint.get('best_acc_epoch', 0)
            patience_counter = checkpoint.get('patience_counter', 0)
            best_epoch = checkpoint.get('best_epoch', 0)
            best_model_state = checkpoint.get('best_model_state', None)
            
            print(f"✅ 成功恢复训练状态:")
            print(f"   - 起始Epoch: {start_epoch}/{config['training']['epochs']}")
            print(f"   - 最佳Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}, Acc={best_loss_acc:.3f})")
            print(f"   - 最佳Acc: {best_val_accuracy:.3f} (Epoch {best_acc_epoch})")
            print(f"   - 早停计数: {patience_counter}/{patience}")
        
        except Exception as e:
            print(f"⚠️  恢复检查点失败: {e}")
            print(f"   从头开始训练...")
            start_epoch = 0
    else:
        print(f"\n🆕 未找到检查点，从头开始训练")

    for epoch in range(start_epoch, config['training']['epochs']):
        # 训练一个epoch
        # 从配置中读取梯度裁剪参数
        max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        grad_norm_type = config['training'].get('grad_norm_type', 2.0)
        
        train_loss, train_accuracy = train_multi_view_stn_model(
            stn_model, train_dataloader, optimizer, criterion, device, epoch,
            dataset_name, all_classes_text_features, max_grad_norm, grad_norm_type, config
        )
        
        # 验证模型
        val_loss, val_accuracy = validate_multi_view_stn_model(
            stn_model, val_dataloader, criterion, device, all_classes_text_features, config
        )
        
        # === 早停检查：基于验证损失 ===
        if val_loss < best_val_loss:
            # 验证损失改善
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_loss_acc = val_accuracy
            patience_counter = 0
            best_epoch = epoch + 1
            best_loss_epoch = best_epoch
            best_model_state = stn_model.state_dict().copy()

            torch.save(best_model_state, ckpt_best_loss)
            print(f"  🎯 验证损失改善 -{improvement:.6f}! 新最佳Loss={best_val_loss:.6f} (第{best_loss_epoch}轮), 当次Acc={best_loss_acc:.3f}")
            print(f"  💾 已保存最佳损失模型: {ckpt_best_loss}")
        else:
            # 验证损失没有改善
            patience_counter += 1
            print(f"  ⏳ 验证损失未改善 {patience_counter}/{patience} 轮 (当前: {val_loss:.6f}, 最佳: {best_val_loss:.6f})")

        # 独立检查最佳准确率
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_acc_epoch = epoch + 1
            torch.save(stn_model.state_dict(), ckpt_best_acc)
            print(f"  🏅 验证准确率提升! 新最佳Acc={best_val_accuracy:.3f} (第{best_acc_epoch}轮)")
            print(f"  💾 已保存最佳准确率模型: {ckpt_best_acc}")
        
        # === 每个epoch保存最新检查点（用于断点续训）===
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': stn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_val_loss': best_val_loss,
            'best_loss_acc': best_loss_acc,
            'best_loss_epoch': best_loss_epoch,
            'best_val_accuracy': best_val_accuracy,
            'best_acc_epoch': best_acc_epoch,
            'patience_counter': patience_counter,
            'best_epoch': best_epoch,
            'best_model_state': best_model_state,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }
        torch.save(checkpoint, ckpt_latest)
        print(f"  💾 已保存最新检查点: {ckpt_latest} (Epoch {epoch+1})")
        
        # 打印训练信息
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"训练[损失={train_loss:.4f}, 准确率={train_accuracy:.3f}] "
              f"验证[损失={val_loss:.4f}, 准确率={val_accuracy:.3f}] "
              f"学习率={current_lr:.2e}")
        
        # 过拟合检测
        if val_loss > train_loss * 1.5:
            print(f"  ⚠️  可能过拟合: 验证损失({val_loss:.4f}) >> 训练损失({train_loss:.4f})")
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # === 每个epoch后清理GPU缓存 ===
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # === 早停检查 ===
        if patience_counter >= patience:
            print(f"\n🛑 早停机制触发!")
            print(f"   📊 验证损失连续 {patience} 个epoch未改善")
            print(f"   🏆 最佳Loss (第{best_loss_epoch}轮): 损失={best_val_loss:.6f}, Acc@bestLoss={best_loss_acc:.3f}")
            print(f"   🏅 最佳Acc  (第{best_acc_epoch}轮): 准确率={best_val_accuracy:.3f}")
            print(f"   📁 已保存: \n      - 最佳损失: {ckpt_best_loss}\n      - 最佳准确: {ckpt_best_acc}")
            print(f"   ⏹️  停止训练，避免过拟合")
            break
        

    # === 训练完成总结 ===
    final_model_path = ckpt_best_loss
    total_epochs = epoch + 1
    
    if best_model_state is not None:
        print(f"\n🎉 多视角STN训练完成！")
        print(f"   📊 训练统计:")
        print(f"      - 总训练轮数: {total_epochs}")
        print(f"      - 最佳模型轮数: {best_epoch}")
        print(f"      - 早停耐心值: {patience}")
        print(f"   🏆 最佳Loss: 损失={best_val_loss:.6f}, Acc@bestLoss={best_loss_acc:.3f} (第{best_loss_epoch}轮)")
        print(f"   🏅 最佳Acc:  准确率={best_val_accuracy:.3f} (第{best_acc_epoch}轮)")
        print(f"   💾 已保存: \n      - 最佳损失: {ckpt_best_loss}\n      - 最佳准确: {ckpt_best_acc}")
        
        # 判断停止原因
        if patience_counter >= patience:
            print(f"   ⏹️  停止原因: 早停机制触发 (验证损失{patience}轮未改善)")
        else:
            print(f"   ⏹️  停止原因: 达到最大训练轮数")
    else:
        # 备用保存（同样按数据集分类）
        os.makedirs(os.path.join('checkpoints', dataset_name), exist_ok=True)
        torch.save(stn_model.state_dict(), final_model_path)
        print(f"\n⚠️  训练异常结束，保存当前模型到(按最佳损失命名): {final_model_path}")
    
    return final_model_path




def setup_multi_view_training(stn_model, config, device):
    """
    设置多视角STN训练环境
    
    Args:
        stn_model: 多视角STN模型
        config: 配置字典
        device: 计算设备
        
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    print("=== 多视角STN训练环境设置 ===")
    
    # 损失函数配置  损失权重设置  读取配置文件中的损失权重
    stn_config = config.get('stn_config', {})
    logits_temp = stn_config.get('logits_temp', 0.07)
    classification_weight = stn_config.get('classification_weight', 1.0)
    contrastive_weight = stn_config.get('contrastive_weight', 0.0)
    decorrelation_weight = stn_config.get('decorrelation_weight', 0.0)
    adaptive_weight = stn_config.get('adaptive_weight', 0.0)

    print(f"📋 多视角损失配置:")
    print(f"   温度参数: {logits_temp}")
    print(f"   标准分类权重: {classification_weight}")
    print(f"   对比损失权重: {contrastive_weight}")
    print(f"   去相关权重: {decorrelation_weight}")
    print(f"   自适应权重: {adaptive_weight}")
    
    # 创建多视角损失函数 - 包含新增的损失类型
    criterion = MultiViewSTNLoss(
        logits_temp=logits_temp,
        classification_weight=classification_weight,
        contrastive_weight=contrastive_weight,
        decorrelation_weight=decorrelation_weight,
        adaptive_weight=adaptive_weight
    )
    print(f"✅ 使用多视角损失函数")
    

    # 优化器配置
    try:
        learning_rate = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"❌ 错误：训练配置参数无效 - {e}")
    
    optimizer_type = 'adamw'
    

    
    print(f"📊 优化器配置:")
    print(f"   学习率: {learning_rate}")
    print(f"   权重衰减: {weight_decay}")
    print(f"   类型: AdamW (统一优化器)")
    
    # === 智能参数分组：为不同类型参数设置不同的权重衰减策略 ===
    #创建两个列表，分别存储需要权重衰减的参数和不需要权重衰减的参数
    decay_params = []       # 需要权重衰减的参数（主要是权重矩阵）
    no_decay_params = []    # 不需要权重衰减的参数（偏置、归一化层等）
    
    print("📊 智能参数分组配置:")
    print("   🔸 权重衰减组: 线性层权重矩阵、卷积层权重")
    print("   🔹 无权重衰减组: 偏置参数、LayerNorm参数、BatchNorm参数")
    
    # 处理定位网络参数
    if hasattr(stn_model, 'localization_network'):
        loc_decay_count = 0
        loc_no_decay_count = 0
        
        # === 收集并输出定位网络的所有可训练参数组件名称 ===
        print(f"   📍 定位网络可训练参数组件详情:")
        all_param_names = []
        decay_param_names = []
        no_decay_param_names = []
        
        for name, param in stn_model.localization_network.named_parameters():  #使用 named_parameters() 获取参数名和参数对象的配对
            if param.requires_grad:  #检查参数是否需要梯度更新（即是否可训练）
                all_param_names.append(name)
                param_shape = tuple(param.shape)
                param_count = param.numel()
                
                # 偏置参数不使用权重衰减
                if 'bias' in name:
                    no_decay_params.append(param)
                    no_decay_param_names.append(name)
                    loc_no_decay_count += 1
                    print(f"      🔹 {name:30} | 形状: {str(param_shape):15} | 参数量: {param_count:6} | 类型: 偏置参数")
                # 归一化层参数不使用权重衰减
                elif any(norm_type in name.lower() for norm_type in ['layernorm', 'batchnorm', 'groupnorm', 'instancenorm']):
                    no_decay_params.append(param)
                    no_decay_param_names.append(name)
                    loc_no_decay_count += 1
                    print(f"      🔹 {name:30} | 形状: {str(param_shape):15} | 参数量: {param_count:6} | 类型: 归一化层参数")
                # 权重矩阵 （主要是线性层和卷积层的权重矩阵）使用权重衰减 
                else:
                    decay_params.append(param)
                    decay_param_names.append(name)
                    loc_decay_count += 1
                    print(f"      🔸 {name:30} | 形状: {str(param_shape):15} | 参数量: {param_count:6} | 类型: 权重矩阵")
        
        print(f"   📊 定位网络参数统计: {loc_decay_count}个权重参数 + {loc_no_decay_count}个偏置/归一化参数")
        print(f"      - 总可训练参数: {len(all_param_names)} 个")
        print(f"      - 权重衰减参数: {len(decay_param_names)} 个")
        print(f"      - 无权重衰减参数: {len(no_decay_param_names)} 个")
        
        # 详细显示定位网络的参数分组（调试用）
        if loc_no_decay_count > 0:
            no_decay_names = []
            for name, param in stn_model.localization_network.named_parameters():
                if param.requires_grad and ('bias' in name or 
                    any(norm_type in name.lower() for norm_type in ['layernorm', 'batchnorm', 'groupnorm', 'instancenorm'])):
                    no_decay_names.append(name)
            print(f"      🔹 无权重衰减参数: {no_decay_names[:3]}{'...' if len(no_decay_names) > 3 else ''}")
    

    # 处理融合模块参数
    if hasattr(stn_model, 'fusion_module'):   #检查融合模块是否存在
        fusion_params = list(stn_model.fusion_module.parameters())
        if fusion_params:
            fusion_decay_count = 0
            fusion_no_decay_count = 0
            
            # === 收集并输出融合模块的所有可训练参数组件名称 ===
            print(f"   🔗 融合模块可训练参数组件详情:")
            fusion_all_param_names = []
            fusion_decay_param_names = []
            fusion_no_decay_param_names = []
            
            for name, param in stn_model.fusion_module.named_parameters():
                if param.requires_grad:
                    fusion_all_param_names.append(name)
                    param_shape = tuple(param.shape)
                    param_count = param.numel()
                    
                    # 偏置参数和归一化层参数不使用权重衰减
                    if ('bias' in name or 
                        any(norm_type in name.lower() for norm_type in ['layernorm', 'batchnorm', 'groupnorm', 'instancenorm'])):
                        no_decay_params.append(param)
                        fusion_no_decay_param_names.append(name)
                        fusion_no_decay_count += 1
                        param_type = "偏置参数" if 'bias' in name else "归一化层参数"
                        print(f"      🔹 {name:30} | 形状: {str(param_shape):15} | 参数量: {param_count:6} | 类型: {param_type}")
                    # 权重矩阵使用权重衰减
                    else:
                        decay_params.append(param)
                        fusion_decay_param_names.append(name)
                        fusion_decay_count += 1
                        print(f"      🔸 {name:30} | 形状: {str(param_shape):15} | 参数量: {param_count:6} | 类型: 权重矩阵")
            
            print(f"   📊 融合模块参数统计: {fusion_decay_count}个权重参数 + {fusion_no_decay_count}个偏置/归一化参数")
            print(f"      - 总可训练参数: {len(fusion_all_param_names)} 个")
            print(f"      - 权重衰减参数: {len(fusion_decay_param_names)} 个")
            print(f"      - 无权重衰减参数: {len(fusion_no_decay_param_names)} 个")
            
            # 详细显示融合模块的参数分组（调试用）
            if fusion_no_decay_count > 0:
                fusion_no_decay_names = []
                for name, param in stn_model.fusion_module.named_parameters():
                    if param.requires_grad and ('bias' in name or 
                        any(norm_type in name.lower() for norm_type in ['layernorm', 'batchnorm', 'groupnorm', 'instancenorm'])):
                        fusion_no_decay_names.append(name)
                print(f"      🔹 无权重衰减参数: {fusion_no_decay_names[:3]}{'...' if len(fusion_no_decay_names) > 3 else ''}")
        else:
            print(f"   🔗 融合模块: 无可训练参数（简单平均融合）")
    

    
    # 验证参数分组结果
    total_params = len(decay_params) + len(no_decay_params)
    if total_params == 0:
        raise RuntimeError("❌ 错误：没有找到任何可训练参数！请检查模型配置。")
    
    print(f"   📈 参数分组结果:")
    print(f"      - 权重衰减组: {len(decay_params)} 个参数")
    print(f"      - 无权重衰减组: {len(no_decay_params)} 个参数")
    print(f"      - 总计: {total_params} 个可训练参数")
    
    # === 创建参数分组优化器 ===
    # 构建参数组：不同组使用不同的权重衰减策略
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
            'name': 'decay_group'
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'name': 'no_decay_group'
        }
    ]
    
    # === 统一使用AdamW优化器 ===
    # 从配置文件读取AdamW参数，确保类型转换
    beta1 = float(config['training'].get('beta1', 0.9))
    beta2 = float(config['training'].get('beta2', 0.999))
    eps = float(config['training'].get('eps', 1e-8))
    
    optimizer = optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps
    )
    print(f"🚀 使用 AdamW 优化器 (参数分组) - beta1={beta1}, beta2={beta2}, eps={eps}")
    print(f"   🔸 权重衰减组: {len(decay_params)}个参数, weight_decay={weight_decay}")
    print(f"   🔹 无权重衰减组: {len(no_decay_params)}个参数, weight_decay=0.0")
    
    # === 学习率调度器（统一使用CosineAnnealingLR + 预热） ===
    warmup_epochs = config['training'].get('warmup_epochs', 5)  # 默认预热5个epoch
    warmup_factor = config['training'].get('warmup_factor', 0.1)  # 预热起始学习率为10%
    
    # 统一使用CosineAnnealingLR调度器
    total_epochs = config['training']['epochs']
    eta_min = float(config['training'].get('eta_min', 1e-6))  # 最小学习率，默认1e-6，确保为float类型
    
    base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=eta_min
    )
    print(f"📈 基础调度器: CosineAnnealingLR (T_max={total_epochs}, eta_min={eta_min:.2e})")
    
    # 包装预热调度器
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=base_scheduler,
            warmup_factor=warmup_factor
        )
        print(f"🔥 启用学习率预热: {warmup_epochs}个epoch, 起始因子: {warmup_factor}")
    else:
        scheduler = base_scheduler
        print("🔥 跳过学习率预热")
    
    return criterion, optimizer, scheduler
