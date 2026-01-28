"""

Multi-View Spatial Transformer Network (MV-STN)

核心设计理念：
1. 多个并联的STN分支，每个分支学习不同的视角变换
2. 共享CLIP编码器，降低参数量和内存占用  
3. 多样性约束，确保不同分支关注不同区域
4. 自适应特征融合，智能聚合多视角特征

架构流程：
输入图像 → N个STN分支 → N个变换后视角 + 1个原始全局视角 → (N+1)个CLIP特征 → 特征融合 → 最终特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import math


#最高的封装类，包含全部流程以及最后的特征融合
class MultiViewSTNModel(nn.Module):
    """
    多视角空间变换网络模型
    
    核心功能：
    1. 多个并联STN分支，每个学习不同的空间变换
    2. 多样性正则化，防止分支学习相同变换
    3. 自适应特征融合，智能聚合多视角特征
    4. 端到端训练，统一优化所有分支
    
    Multi-view STN model with N parallel transformation branches.
    """
    
    def __init__(self, clip_model, config, num_views=4):
        """
        初始化多视角STN模型
        
        Args:
            clip_model: 预训练的CLIP模型实例
            config (dict): STN配置字典
            num_views (int): 视角数量 (支持2、4、5、6、8个视角)
        """
        super().__init__()
        
        self.num_views = num_views
        self.clip_model = clip_model
        self.config = config
        
        # 从配置中获取参数
        hidden_dim = config.get('hidden_dim',512)
        dropout = config.get('dropout', 0.1)
        #配置文件中决定
        fusion_mode = config.get('fusion_mode', 'concat')  # 融合模式
        
        print(f"🔄 初始化多视角STN模型: {num_views}个视角")
        
        # === 共享定位网络 ===
        # 只需要CLIP模型和视角数量，因为编码器是冻结的ViT
        #定位网络编码器，预测变换参数
        self.localization_network = SharedLocalizationNetwork(
            clip_model=clip_model,
            num_views=num_views,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # === 多视角STN处理器 === 执行空间变换操作的模块，网格生成器和采样网络，并对变换后的图片提取特征
        self.stn_processor = MultiViewSTNProcessor(
            clip_model=clip_model,
            num_views=num_views
        )
        
        # === 特征融合模块 ===   根据配置文件选择融合方式，初始化融合模块以及相关参数
        feature_dim = clip_model.visual.output_dim  # CLIP特征维度
        # 实际视角数量：STN视角数 + 1个原始全局视角
        actual_num_views = num_views + 1
        
        print(f"🔄 融合模块初始化: {num_views}个STN视角 + 1个全局视角 = {actual_num_views}个总视角")

        if fusion_mode == 'weighted':
            self.fusion_module = WeightedFusion(
                feature_dim=feature_dim,
                num_views=actual_num_views  # 使用实际视角数量
            )
        elif fusion_mode == 'concat':
            # 拼接融合：将多视角特征拼接后通过线性层降维
            self.fusion_module = ConcatFusion(
                feature_dim=feature_dim,
                num_views=actual_num_views,  # 使用实际视角数量
                dropout=dropout  # 使用配置文件的dropout参数
            )
        elif fusion_mode == 'transformer':
            # Transformer融合：使用CLS Token和Transformer编码器
            transformer_heads = config.get('transformer_heads', 8)
            transformer_layers = config.get('transformer_layers', 2)
            transformer_dropout = config.get('transformer_dropout', 0.1)
            self.fusion_module = TransformerFusion(
                feature_dim=feature_dim,
                num_views=actual_num_views,  # 使用实际视角数量
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=transformer_dropout
            )
        else:
            self.fusion_module = SimpleFusion()  # 简单平均
            
        print(f"  🎯 特征融合模式: {fusion_mode}")
        



        # === 冻结CLIP参数（在所有STN组件创建完成后） ===
        self._freeze_clip_parameters()
        
        print(f"✅ 多视角STN模型初始化完成")
    

    def _freeze_clip_parameters(self):
        """冻结CLIP模型的所有参数，只训练STN组件"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"    🔒 CLIP模型参数已冻结: {frozen_params:,} 个参数")
        print(f"    🎯 可训练STN参数: {trainable_params:,} 个参数")
    
    def forward(self, images_448, mode='train'):
        """
        多视角STN前向传播 
        
        Args:
            images_448 (torch.Tensor): 输入图像 [B, 3, 448, 448]
            mode (str): 运行模式
                - 'train': 训练/验证模式，返回 (fused_features, view_features)
                           用于训练和验证阶段，需要 view_features 计算损失
                - 'test': 测试模式，返回 (fused_features, vis_data)
                          用于测试阶段，需要可视化数据
            
        Returns:
            根据 mode 返回不同内容：
            - mode='train': (fused_features [B, D], view_features [B, N, D])
            - mode='test': (fused_features [B, D], vis_data dict)
        """
        batch_size = images_448.size(0)
        
        # === 步骤1：生成CLIP预处理图像用于定位网络 ===
        from data_preprocess import prepare_clip_input
        #双三次插值匹配CLIP官方预处理
        preprocessed_images = prepare_clip_input(images_448, clip_size=224)  # [B, 3, 224, 224]
        
        # === 步骤2：使用ViT提取特征并预测所有视角位置 ===
        # 在 localization_network 内部会：
        # 1. 使用CLIP ViT提取patch tokens特征
        # 2. 对patch tokens进行平均池化得到空间特征
        # 3. 通过MLP预测所有视角的位置参数
        # 同时返回CLS特征用于后续复用
        position_params, cls_features = self.localization_network(preprocessed_images)  # [B, 2*N], [B, D]
        
        # 内存优化：立即释放预处理图像
        del preprocessed_images
        
        # === 步骤3：生成变换矩阵 ===
        theta_matrices = self.localization_network.get_transformation_matrices(position_params)  # [B, N, 2, 3]
        
        # === 步骤4：并行执行所有视角的空间变换和特征提取 ===
        # 注意：这里使用448×448原始图像进行空间变换，而不是224×224预处理图像，并进行特征提取
        # 只在测试模式下保存图像用于可视化，避免训练和验证阶段的内存累积
        save_images = (mode == 'test')
        multi_view_features, view_images = self.stn_processor(images_448, theta_matrices, save_images=save_images)
        # multi_view_features: [B, N, D]
        # view_images: [B, N, 3, 224, 224] 或 None
        
        # === 步骤4.5：添加原始图片特征作为全局视角 ===
        # 复用步骤1中已经计算的cls_features，避免重复编码
        original_features = F.normalize(cls_features, dim=-1)  # [B, D]
        
        # 将原始特征添加到多视角特征中：[B, N, D] + [B, 1, D] -> [B, N+1, D]
        original_features_expanded = original_features.unsqueeze(1)  # [B, 1, D]
        multi_view_features_with_original = torch.cat([original_features_expanded, multi_view_features], dim=1)  # [B, N+1, D]

        # === 步骤5：特征融合 ===  
        #  所有融合模块在forward中只要传入多视角特征和原始图片特征即可
        
        fused_features = self.fusion_module(multi_view_features_with_original)  # 使用包含原始特征的多视角特征
        
        # === 步骤6：根据模式返回不同结果 ===
        if mode == 'train':
            # 训练/验证模式：返回融合特征和多视角特征（用于损失计算）
            return fused_features, multi_view_features
        elif mode == 'test':
            # 测试模式：返回融合特征和可视化数据
            vis_data = {
                'view_images': view_images,           # [B, N, 3, 224, 224] 变换后的图像
                'position_params': position_params,   # [B, 2*N] 位置参数
                'theta_matrices': theta_matrices,     # [B, N, 2, 3] 变换矩阵
                'num_views': self.num_views          # 视角数量
            }
            return fused_features, vis_data
        else:
            raise ValueError(f"不支持的模式: {mode}，请使用 'train' 或 'test'")
    



class SharedLocalizationNetwork(nn.Module):
    """
    共享定位网络：基于CLIP ViT patch tokens预测所有视角的位置参数
    
    设计理念：
    1. 使用CLIP ViT的patch tokens作为空间特征输入
    2. 一个MLP同时预测所有N个视角的中心位置(x,y)
    3. 尺度固定为50%，无旋转和剪切
    4. 输出: [B, 2*N] -> N个视角的(x,y)坐标
    """
    
    def __init__(self, clip_model, num_views, hidden_dim=512, dropout=0.1):
        """
        Args:
            clip_model: CLIP模型实例（冻结的ViT编码器）
            num_views (int): 视角数量
            hidden_dim (int): MLP隐藏层维度（可选，使用默认值）
            dropout (float): Dropout概率（可选，使用默认值）
        """
        super().__init__()
        
        self.clip_model = clip_model
        self.num_views = num_views
        
        # 获取ViT patch tokens的维度（只支持ViT模型）
        assert hasattr(clip_model.visual, 'transformer'), "只支持ViT模型作为图像编码器"
        self.patch_dim = clip_model.visual.transformer.width
        print(f"      - Patch维度: {self.patch_dim}")
        
        # === 共享定位网络（改进架构） ===
        # 输入：patch tokens [B, num_patches, patch_dim]，输出：2*num_views个位置参数
        
        # 步骤1: 1x1卷积降维 patch_dim -> hidden_dim (保持空间维度)
        self.channel_reduction = nn.Conv2d(
            in_channels=self.patch_dim, 
            out_channels=hidden_dim, 
            kernel_size=1
        )
        
        # === 新增：空间信息聚合卷积块（带瓶颈设计） ===  再降到hidden_dim的一半
        conv_output_dim = hidden_dim // 2  # 瓶颈层输出维度
        self.spatial_aggregation_convs = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, conv_output_dim, kernel_size=3, padding=1, bias=False), # 降维
            nn.BatchNorm2d(conv_output_dim),
            nn.ReLU(inplace=True)
        )

        # === 自适应池化层，统一空间维度 ===
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # 步骤2: 降维层（使用固定输入维度）
        # 计算展平后的特征维度: conv_output_dim * 5 * 5
        flattened_dim = conv_output_dim * 5 * 5
        print(f"      - 展平维度: {flattened_dim} ({conv_output_dim} x 5^2)")

        self.dimension_reduction = nn.Linear(flattened_dim, 128)
        
        # 步骤3: 添加dropout层（在ReLU激活函数之后）
        self.dropout = nn.Dropout(dropout)
        
        # 步骤4: 最终位置预测层
        self.position_predictor = nn.Linear(128, 2 * num_views)
        
        print(f"    🏗️  位置预测器架构:")
        print(f"      - 1x1卷积: {self.patch_dim} -> {hidden_dim}")
        print(f"      - 空间聚合卷积: 2x[3x3 Conv -> BN -> ReLU], 输出通道: {conv_output_dim}")
        print(f"      - 自适应池化: -> 5x5")
        print(f"      - 展平维度: {flattened_dim} ({conv_output_dim} x 5^2)")
        print(f"      - 降维层: {flattened_dim} -> 128")
        print(f"      - Dropout: {dropout}")
        print(f"      - 输出层: 128 -> {2 * num_views}")
        
        # === 固定变换参数 ===
        self.fixed_scale = 0.5  # 固定尺度为50%
        
        # === 初始化 初始化位置预测器，使裁剪块平铺整个图像平面 ===
        self._initialize_position_predictor()
        
        print(f"  🎯 共享定位网络初始化完成")
        print(f"    - Patch维度: {self.patch_dim}")
        print(f"    - 视角数量: {num_views}")
        print(f"    - 输出参数: {2 * num_views} (每个视角2个位置参数)")
    
    def forward(self, preprocessed_images):
        """
        预测所有视角的位置参数，同时返回CLIP的CLS特征
        
        Args:
            preprocessed_images (torch.Tensor): CLIP预处理图像 [B, 3, 224, 224]
            
        Returns:
            tuple: (position_params, cls_features)
                - position_params (torch.Tensor): 位置参数 [B, 2*N]
                - cls_features (torch.Tensor): CLIP CLS特征 [B, D]
        """
        batch_size = preprocessed_images.size(0)
        
        # === 步骤1：提取ViT patch tokens特征 ===
        # CLIP参数已经冻结（requires_grad=False），但需要保持梯度计算图用于STN反向传播
        cls_features, patch_features = self.clip_model.encode_image(preprocessed_images)
        # patch_features: [B, num_patches, patch_dim]
        

        
        # === 步骤2：改进的位置预测流程（保持空间维度） ===
        # 2.1: 重塑patch tokens为4D特征图 [B, num_patches, patch_dim] -> [B, patch_dim, H, W]
        batch_size = patch_features.shape[0]
        spatial_size = int(patch_features.shape[1] ** 0.5)  # 计算空间维度 sqrt(num_patches)
        
        # 重塑为4D特征图
        feature_map = patch_features.transpose(1, 2).view(
            batch_size, self.patch_dim, spatial_size, spatial_size
        )  # [B, patch_dim, H, W]  H=W
        
        # 2.2: 1x1卷积降维 [B, patch_dim, H, W] -> [B, hidden_dim, H, W]
        reduced_feature_map = self.channel_reduction(feature_map)  # [B, hidden_dim, H, W]
        reduced_feature_map = F.relu(reduced_feature_map)

        # === 新增：空间信息聚合 ===
        aggregated_feature_map = self.spatial_aggregation_convs(reduced_feature_map)

        # === 自适应池化，统一空间维度 ===
        pooled_feature_map = self.adaptive_pool(aggregated_feature_map) # [B, hidden_dim, 5, 5]

        # 2.3: 展平空间特征 [B, hidden_dim, 5, 5] -> [B, hidden_dim * 5 * 5]
        flattened_features = pooled_feature_map.flatten(start_dim=1)  # [B, flattened_dim]
        
        # 2.4: 降维到固定大小 [B, flattened_dim] -> [B, 128] 
        compressed_features = self.dimension_reduction(flattened_features)  # [B, 128]
        compressed_features = F.relu(compressed_features)  # 添加激活函数
        compressed_features = self.dropout(compressed_features)  # 添加dropout
        
        # 2.5: 预测位置参数 [B, 128] -> [B, 2*N]
        position_params = self.position_predictor(compressed_features)  # [B, 2*N]
        
        # === 步骤3：限制位置参数范围 ===
        # 使用tanh将位置限制在[-1, 1]范围内，然后缩放到[-0.5, 0.5]
        position_params = torch.tanh(position_params)
        # 将范围从[-1, 1]缩放到[-0.5, 0.5]，防止视角超出图片边界
        # position_params = position_params * 0.5
        
        return position_params, cls_features

    def get_transformation_matrices(self, position_params):
        """
        根据位置参数生成所有视角的变换矩阵
        
        Args:
            position_params (torch.Tensor): 位置参数 [B, 2*N]
            
        Returns:
            torch.Tensor: 变换矩阵 [B, N, 2, 3]
        """
        batch_size = position_params.size(0)
        
        # 重塑位置参数
        positions = position_params.view(batch_size, self.num_views, 2)  # [B, N, 2]
        
        # === 调试输出：显示位置参数（仅第一个batch和前几次调用） ===
        # if not hasattr(self, '_debug_call_count'):
        #     self._debug_call_count = 0
        
        # if self._debug_call_count < 3 and batch_size > 0:  # 只在前3次调用时输出
        #     print(f"\n🔍 变换矩阵生成调试 (第{self._debug_call_count+1}次调用):")
        #     print(f"   批次大小: {batch_size}, 视角数量: {self.num_views}")
        #     print(f"   位置参数形状: {position_params.shape}")
            
        #     # 显示第一个样本的位置参数
        #     first_sample_positions = positions[0]  # [N, 2]
        #     print(f"   第一个样本的位置参数:")
        #     for view_idx in range(self.num_views):
        #         tx = first_sample_positions[view_idx, 0].item()
        #         ty = first_sample_positions[view_idx, 1].item()
        #         print(f"     视角{view_idx}: tx={tx:+.3f}, ty={ty:+.3f}")
        
        # 创建变换矩阵
        theta_matrices = []
        
        for view_idx in range(self.num_views):
            # 提取当前视角的位置参数（已限制在[-0.5, 0.5]范围内）
            tx = positions[:, view_idx, 0]  # [B] tx范围: [-0.5, 0.5]
            ty = positions[:, view_idx, 1]  # [B] ty范围: [-0.5, 0.5]
            
            # 构建仿射变换矩阵 [B, 2, 3]
            # [scale, 0,     tx]
            # [0,     scale, ty]
            # 由于scale=0.5，tx/ty在[-0.5,0.5]，所以变换后的视角不会超出图片边界
            theta = torch.zeros(batch_size, 2, 3, device=position_params.device)
            theta[:, 0, 0] = self.fixed_scale  # x方向缩放 (0.5)
            theta[:, 1, 1] = self.fixed_scale  # y方向缩放 (0.5)
            theta[:, 0, 2] = tx  # x方向平移 (范围: [-0.5, 0.5])
            theta[:, 1, 2] = ty  # y方向平移 (范围: [-0.5, 0.5])
            
            theta_matrices.append(theta)
        
        # 堆叠所有视角的变换矩阵
        theta_matrices = torch.stack(theta_matrices, dim=1)  # [B, N, 2, 3]
        
        # === 调试输出：显示变换矩阵（仅第一个batch和前几次调用） ===
        # if self._debug_call_count < 3 and batch_size > 0:
        #     print(f"   生成的变换矩阵 (第一个样本):")
        #     first_sample_matrices = theta_matrices[0]  # [N, 2, 3]
        #     for view_idx in range(self.num_views):
        #         matrix = first_sample_matrices[view_idx]  # [2, 3]
        #         print(f"     视角{view_idx}:")
        #         print(f"       [[{matrix[0,0]:+.3f}, {matrix[0,1]:+.3f}, {matrix[0,2]:+.3f}],")
        #         print(f"        [{matrix[1,0]:+.3f}, {matrix[1,1]:+.3f}, {matrix[1,2]:+.3f}]]")
        #         print(f"       解释: 缩放={self.fixed_scale}, 平移=({matrix[0,2]:+.3f}, {matrix[1,2]:+.3f})")
            
        #     self._debug_call_count += 1
        #     print()  # 空行分隔
        

        return theta_matrices


    # 初始化位置预测器
    def _initialize_position_predictor(self):
        """
        初始化位置预测器，使裁剪块平铺整个图像平面
        
        策略：
        1. 前面层使用Kaiming Normal初始化（适合ReLU）
        2. 最后一层权重初始化为全零，确保输出完全由偏置决定
        3. 偏置设置为网格中心坐标，实现平铺覆盖
        4. 每个50%尺度的裁剪块覆盖不同的网格区域

        """
        
        # === 步骤0: 初始化前面的层 ===
        print(f"    🔧 初始化定位网络各层权重:")
        
        # 0.1: 初始化1x1卷积层（channel_reduction）
        nn_init.kaiming_normal_(self.channel_reduction.weight, mode='fan_out', nonlinearity='relu')
        if self.channel_reduction.bias is not None:
            nn_init.zeros_(self.channel_reduction.bias)
        # print(f"      - Conv2d层: Kaiming Normal初始化 (fan_out, ReLU), 偏置置零")
        # print(f"        权重形状: {self.channel_reduction.weight.shape}")
        
        # 0.2: 初始化降维层（dimension_reduction）
        nn_init.kaiming_normal_(self.dimension_reduction.weight, mode='fan_in', nonlinearity='relu')
        if self.dimension_reduction.bias is not None:
            nn_init.zeros_(self.dimension_reduction.bias)
        # print(f"      - Linear层: Kaiming Normal初始化 (fan_in, ReLU), 偏置置零")
        # print(f"        权重形状: {self.dimension_reduction.weight.shape}")
        
        # === 步骤1: 最后一层特殊初始化 ===
        # 获取最后一层（现在是独立的Linear层）
        final_layer = self.position_predictor
        
        # 权重初始化为全零，确保网络输出不受输入影响，完全由偏置决定
        final_layer.weight.data.zero_()
        # print(f"      - 位置预测层: 权重全零初始化，形状: {final_layer.weight.shape}")
        
        # === 步骤2: 计算网格布局 ===
        # 为不同视角设置网格化的初始位置偏置    创建一个全零张量，用于存储所有视角的位置参数偏置
        bias_init = torch.zeros(2 * self.num_views)  ## [0, 0, 0, 0, 0, 0, 0, 0]   N=4
        
        # === 步骤3: 硬编码视角位置初始化 ===
        # 根据视角数量直接设置特定的初始化位置
        # 计算反双曲正切值，使得tanh(init_value) = target_value
        import math
        
        def inverse_tanh(target_value):
            """计算反双曲正切值，使得tanh(result) = target_value"""
            if abs(target_value) >= 1.0:
                raise ValueError(f"目标值必须在(-1, 1)范围内，得到: {target_value}")
            return math.atanh(target_value)
        
        # 目标值（经过tanh后的期望值）
        target_offset = 0.5  # 期望的tanh后的偏移值
        init_offset = inverse_tanh(target_offset)  # 初始化时需要的值
        
        if self.num_views == 2:
            # 2个视角：水平并排
            positions = [
                (-init_offset, 0.0),  # 左侧 (tanh后为-0.5)
                (init_offset, 0.0)    # 右侧 (tanh后为0.5)
            ]
            print(f"    📍 2个视角: 水平并排 (初始化值: ±{init_offset:.4f}, tanh后: ±{target_offset})")
            
        elif self.num_views == 4:
            # 4个视角：四个象限（保持原有设计）
            positions = [
                (-init_offset, -init_offset),  # 左上角 (tanh后为[-0.5, -0.5])
                (init_offset, -init_offset),   # 右上角 (tanh后为[0.5, -0.5])
                (-init_offset, init_offset),   # 左下角 (tanh后为[-0.5, 0.5])
                (init_offset, init_offset)     # 右下角 (tanh后为[0.5, 0.5])
            ]
            print(f"    📍 4个视角: 四象限布局 (初始化值: ±{init_offset:.4f}, tanh后: ±{target_offset})")
            
        elif self.num_views == 5:
            # 5个视角：四象限 + 中心
            positions = [
                (-init_offset, -init_offset),  # 左上角
                (init_offset, -init_offset),   # 右上角
                (-init_offset, init_offset),   # 左下角
                (init_offset, init_offset),    # 右下角
                (0.0, 0.0)                     # 中心 (tanh后仍为[0.0, 0.0])
            ]
            print(f"    📍 5个视角: 四象限 + 中心 (初始化值: ±{init_offset:.4f}, tanh后: ±{target_offset})")
            
        elif self.num_views == 6:
            # 6个视角：四象限 + 中心圆形分布的上下两个点
            # 中心圆形半径目标值为0.5（tanh后）
            radius_init = init_offset  # 使用相同的初始化值
            positions = [
                # 四象限
                (-init_offset, -init_offset),  # 左上角
                (init_offset, -init_offset),   # 右上角
                (-init_offset, init_offset),   # 左下角
                (init_offset, init_offset),    # 右下角
                # 中心圆形分布：上下
                (0.0, -radius_init),          # 上方 (tanh后为[0.0, -0.5])
                (0.0, radius_init),           # 下方 (tanh后为[0.0, 0.5])
            ]
            print(f"    📍 6个视角: 四象限 + 中心圆形分布（上下）(初始化值: ±{init_offset:.4f}, tanh后半径: {target_offset})")
            
        elif self.num_views == 8:
            # 8个视角：四象限 + 中心圆形分布的上下左右四个点
            # 中心圆形半径目标值为0.5（tanh后）
            radius_init = init_offset  # 使用相同的初始化值
            positions = [
                # 四象限
                (-init_offset, -init_offset),  # 左上角
                (init_offset, -init_offset),   # 右上角
                (-init_offset, init_offset),   # 左下角
                (init_offset, init_offset),    # 右下角
                # 中心圆形分布：上下左右
                (0.0, -radius_init),          # 上方 (tanh后为[0.0, -0.5])
                (radius_init, 0.0),           # 右方 (tanh后为[0.5, 0.0])
                (0.0, radius_init),           # 下方 (tanh后为[0.0, 0.5])
                (-radius_init, 0.0)           # 左方 (tanh后为[-0.5, 0.0])
            ]
            print(f"    📍 8个视角: 四象限 + 中心圆形分布 (初始化值: ±{init_offset:.4f}, tanh后半径: {target_offset})")
            
        else:
            # 只支持2、4、5、8个视角
            raise ValueError(f"不支持的视角数量: {self.num_views}。仅支持2、4、5、6、8个视角。")
        
        # === 步骤4: 设置偏置为目标坐标 ===
        for i, (x, y) in enumerate(positions):
            bias_init[2*i] = x      # x坐标
            bias_init[2*i+1] = y    # y坐标
        
        # 使用copy_()进行in-place操作，符合PyTorch惯例
        final_layer.bias.data.copy_(bias_init)
        
        
        # === 步骤5: 输出初始化信息 ===
        print(f"    🎯 位置初始化完成，{self.num_views}个视角:")
        for i, (x, y) in enumerate(positions):
            print(f"      视角{i}: 中心({x:+.1f}, {y:+.1f})")
        
        print(f"    ✅ 权重零化+偏置初始化完成，训练开始时输出完全由偏置决定")
    

    
    

#包含STN的网格生成器和采样，以及对变换后的图片提取特征
class MultiViewSTNProcessor(nn.Module):
    """
    多视角STN处理器：统一处理所有视角的空间变换和特征提取
    """
    
    def __init__(self, clip_model, num_views):
        super().__init__()
        
        self.clip_model = clip_model
        self.num_views = num_views
        
        # 空间变换参数
        self.output_size = (224, 224)
        self.mode = 'bilinear'
        self.padding_mode = 'zeros'
        
        print(f"  🔄 多视角STN处理器初始化完成，支持{num_views}个并行变换（使用PyTorch内置函数）")
    
    def forward(self, original_images, theta_matrices, save_images=False):
        """
        执行多视角空间变换和特征提取
        
        Args:
            original_images (torch.Tensor): 原始图像 [B, 3, H, W]
            theta_matrices (torch.Tensor): 变换矩阵 [B, N, 2, 3]
            save_images (bool): 是否保存变换后的图像（用于可视化），默认False
        
        Returns:
            tuple: (多视角特征, 变换后图像)
                - view_features: [B, N, D]
                - transformed_images: [B, N, 3, 224, 224] 或 None
        """
        batch_size, num_views = theta_matrices.shape[:2]
        
        view_features = []
        # 只在明确需要可视化时才保存图像，避免验证阶段的内存累积
        transformed_images = [] if save_images else None
        
        # === 对每个视角执行变换 ===
        for view_idx in range(num_views):
            theta = theta_matrices[:, view_idx]  # [B, 2, 3]
            
            # === 直接使用PyTorch内置函数进行空间变换 ===
            # 步骤1: 生成采样网格
            output_size = torch.Size([batch_size, original_images.size(1)] + list(self.output_size))
            grid = F.affine_grid(theta, output_size, align_corners=False)  # [B, H, W, 2]
            
            # 步骤2: 执行网格采样
            transformed = F.grid_sample(
                original_images, 
                grid, 
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=False
            )   # [B, 3, 224, 224]  返回变换后的图片
            
            # 内存优化：立即释放grid
            del grid
            
            # CLIP特征提取
            # 注意：transformed图像已经是标准化后的，可以直接输入CLIP
            
            # === CLIP ViT特征提取 ===
            # CLIP参数已经冻结（requires_grad=False），但需要保持梯度计算图用于STN反向传播
            cls_features, _ = self.clip_model.encode_image(transformed)
            features = cls_features
            
            features = F.normalize(features, dim=-1)  # 提取的特征L2归一化
            
            view_features.append(features)

            # 仅在 eval/test 阶段用于可视化时收集图像；训练阶段不收集，避免大张量堆叠
            if transformed_images is not None:
                transformed_images.append(transformed)
            else:
                # 训练阶段立即释放变换后的图像
                del transformed
        
        # 堆叠结果
        view_features = torch.stack(view_features, dim=1)      # [B, N, D]
        if transformed_images is not None:
            transformed_images = torch.stack(transformed_images, dim=1)  # [B, N, 3, 224, 224] 张量
        else:
            transformed_images = None
        
        return view_features, transformed_images






#特征融合模块

class WeightedFusion(nn.Module):
    """
    基于相似度的加权融合 (改进版)
    
    改进点：
    1. 使用传入的原始全局特征作为基准，而不是计算平均值
    2. 计算局部视角与全局特征的相似度进行加权
    3. 引入门控机制融合加权后的局部特征与全局特征
    """
    
    def __init__(self, feature_dim, num_views, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim
        
        # 门控网络：学习如何融合局部特征和全局特征
        # 输入: [weighted_local_feature, global_feature] -> 2 * feature_dim
        # 输出: 门控值 [B, D] (通道级门控，类似于SE Block)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),  # 输出维度改为 feature_dim
            nn.Sigmoid()
        )
        
        # 初始化门控网络权重
        self._init_gate_weights()
        
        print(f"    ⚖️ 加权融合模块初始化 (改进版):")
        print(f"      - 温度参数: {temperature}")
        print(f"      - 门控机制: 启用 (通道级门控 [B, D] + LayerNorm)")
        print(f"      - 初始化: 中性起步 (Gate ≈ 0.5)")
    
    def _init_gate_weights(self):
        """
        初始化门控网络权重
        策略：
        1. 第一层：Kaiming初始化，适合ReLU
        2. 第二层：权重和偏置全零初始化
           这样初始输出logit为0，Sigmoid(0)=0.5
           实现"中性融合起步"，即初始时局部和全局特征各占50%
        """
        # 第一层线性层 (index 0)
        first_layer = self.gate_net[0]
        nn.init.kaiming_normal_(first_layer.weight, mode='fan_out', nonlinearity='relu')
        if first_layer.bias is not None:
            nn.init.constant_(first_layer.bias, 0)
            
        # 第二层线性层 (index 3) - 注意：由于通过LayerNorm插入，索引变为3
        last_layer = self.gate_net[3]
        nn.init.constant_(last_layer.weight, 0)
        if last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0)

    def forward(self, multi_view_features):
        """
        Args:
            multi_view_features (torch.Tensor): [B, N+1, D]
            其中 index 0 是原始全局特征，index 1:N+1 是 STN 局部特征
        """
        batch_size, num_total_views, feature_dim = multi_view_features.shape
        
        # 分离全局特征和局部特征
        # index 0: 原始全局特征 [B, 1, D] -> [B, D]
        global_feature = multi_view_features[:, 0, :]
        
        # index 1+: STN局部特征 [B, N, D]
        local_features = multi_view_features[:, 1:, :]
        
        # === 步骤1: 计算局部特征的加权组合 ===
        # 计算每个局部视角与全局特征的相似度
        # [B, N, D] @ [B, D, 1] -> [B, N, 1] -> [B, N]
        similarities = torch.bmm(
            local_features, 
            global_feature.unsqueeze(-1)
        ).squeeze(-1)
        
        # 温度缩放和softmax得到权重
        weights = F.softmax(similarities / self.temperature, dim=-1)  # [B, N]
        
        # 加权融合局部特征
        # weights: [B, N] -> [B, N, 1]
        # local_features: [B, N, D]
        # sum([B, N, 1] * [B, N, D], dim=1) -> [B, D]
        weighted_local_feature = torch.sum(
            weights.unsqueeze(-1) * local_features,
            dim=1
        )
        
        # === 步骤2: 门控融合 (通道级) ===
        # 拼接加权后的局部特征和全局特征
        concat_features = torch.cat([weighted_local_feature, global_feature], dim=1)  # [B, 2*D]
        
        # 计算门控值 (决定每个特征维度多大程度上使用局部特征)
        gate = self.gate_net(concat_features)  # [B, D]
        
        # 最终融合: gate * local + (1-gate) * global
        # 这里的乘法是逐元素的 (element-wise)，实现了通道级的选择
        fused_features = gate * weighted_local_feature + (1 - gate) * global_feature
        
        # L2归一化确保输出特征单位长度
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features


class SimpleFusion(nn.Module):
    """
    简单平均融合
    """
    
    def forward(self, multi_view_features):
        """
        Args:
            multi_view_features (torch.Tensor): [B, N, D]
        """
        fused_features = multi_view_features.mean(dim=1)  # [B, D]
        
        # L2归一化确保输出特征单位长度
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features


class ConcatFusion(nn.Module):
    """
    拼接多视角特征并使用线性层降维的融合方法
    
    策略：
    1. 将多个视角的特征直接拼接 [B, N, D] -> [B, N*D]
    2. 使用线性层将拼接后的高维特征降维到原始维度 [B, N*D] -> [B, D]
    3. 使用L2归一化确保输出特征单位长度，与余弦相似度计算理论匹配
    
    优势：
    - 保留所有视角的完整信息，无信息损失
    - 允许模型学习视角间的复杂交互模式
    - 通过降维层学习最优的特征组合方式
    - L2归一化与CLIP的余弦相似度计算在理论上更一致
    """
    
    def __init__(self, feature_dim, num_views, dropout=0.1):
        """
        Args:
            feature_dim (int): 单个视角特征维度
            num_views (int): 视角数量
            dropout (float): Dropout概率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_views = num_views
        
        # 拼接后的特征维度
        concat_dim = feature_dim * num_views
        
        # 中间层维度：设置为输入维度的一半，平衡表达能力和参数效率
        intermediate_dim = concat_dim // 2  # 自适应：N×D -> (N×D)/2 -> D
        
        # 降维网络：拼接维度 -> 中间维度 -> 输出维度
        self.fusion_net = nn.Sequential(
            # 第一层：降维
            nn.Linear(concat_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第二层：输出层
            nn.Linear(intermediate_dim, feature_dim),
        )
        
        # 改进权重初始化以提高数值稳定性
        self._init_fusion_weights()
        
        print(f"    🔗 拼接融合模块初始化:")
        print(f"      - 输入维度: [B, {num_views}, {feature_dim}]")
        print(f"      - 拼接维度: [B, {concat_dim}]")
        print(f"      - 中间维度: {intermediate_dim} (输入维度的一半)")
        print(f"      - 输出维度: [B, {feature_dim}]")
        print(f"      - 最终归一化: L2归一化 (与余弦相似度匹配)")
        print(f"      - 残差连接: 禁用")
        print(f"      - Dropout: {dropout}")
    
    def _init_fusion_weights(self):
        """改进的权重初始化，提高数值稳定性"""
        for module in self.fusion_net:
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，适合ReLU激活函数
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    # 偏置初始化为小的正值，避免死神经元
                    nn.init.constant_(module.bias, 0.01)
        print(f"    🔧 融合网络权重初始化完成 (Xavier + 小偏置)")
    
    def forward(self, multi_view_features):
        """
        拼接融合前向传播
        
        Args:
            multi_view_features (torch.Tensor): 多视角特征 [B, N, D]
            
        Returns:
            torch.Tensor: 融合后的特征 [B, D]
        """
        batch_size, num_views, feature_dim = multi_view_features.shape
        
        # 验证输入维度
        assert num_views == self.num_views, f"期望{self.num_views}个视角，得到{num_views}个"
        assert feature_dim == self.feature_dim, f"期望特征维度{self.feature_dim}，得到{feature_dim}"
        
        # === 步骤1: 拼接所有视角特征 包含cls

        concat_features = multi_view_features.reshape(batch_size, -1)
        
        # === 步骤2: 通过降维网络融合 ===
        # [B, N*D] -> [B, D]
        fused_features = self.fusion_net(concat_features)
        
        # === 步骤3: L2归一化 ===
        # 应用L2归一化，与后续余弦相似度计算理论匹配
        # F.normalize(input, p=2, dim=1): 沿特征维度进行L2归一化
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features


class TransformerFusion(nn.Module):
    """
    基于Transformer的多视角特征融合
    
    核心思想：
    1. 将N个视角特征看作一个序列 [B, N, D]
    2. 在序列前添加一个可学习的CLS Token
    3. 使用Transformer编码器层进行特征交互和"投票"
    4. 最终输出CLS Token作为融合后的特征表示
    
    架构流程：
    [CLS, view1, view2, ..., viewN] → Transformer Encoder → CLS Token (融合特征)
    """
    
    def __init__(self, feature_dim, num_views, num_heads=8, num_layers=2, 
                 dropout=0.1, feedforward_dim=None):
        """
        Args:
            feature_dim (int): 输入的每个视角的特征维度
            num_views (int): 视角数量
            num_heads (int): 多头注意力头数
            num_layers (int): Transformer层数
            dropout (float): Dropout概率
            feedforward_dim (int): 默认为4*feature_dim  Transformer编码器层内部的前馈网络（一个小型MLP）的隐藏层维度。
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_views = num_views
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 设置前馈网络维度
        if feedforward_dim is None:
            feedforward_dim = 4 * feature_dim
        
        # === 可学习的CLS Token ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # === 位置编码（可选） ===
        # 为序列中的每个位置（CLS + N个视角）创建位置编码  可学习的位置编码
        max_seq_len = num_views + 1  # +1 for CLS token
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, feature_dim))
        
        # === Transformer编码器层  Encoder layer 层数由参数控制===
        encoder_layer = nn.TransformerEncoderLayer(  #多头注意力机制
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 输入格式为 [batch, seq, feature]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # === 输出投影层（可选，简化为单层） ===
        self.output_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # === 初始化 ===
        self._init_parameters()
        
        print(f"    🤖 Transformer融合模块初始化:")
        print(f"      - 特征维度: {feature_dim}")
        print(f"      - 视角数量: {num_views}")
        print(f"      - 注意力头数: {num_heads}")
        print(f"      - Transformer层数: {num_layers}")
        print(f"      - 序列长度: {max_seq_len} (CLS + {num_views}个视角)")
        print(f"      - 前馈维度: {feedforward_dim}")
        print(f"      - 位置编码: 启用")
    
    def _init_parameters(self):
        """初始化参数"""
        # CLS Token初始化为小的随机值
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 位置编码初始化
        nn.init.normal_(self.position_embeddings, std=0.02)
        
        # 输出投影层初始化
        for module in self.output_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, multi_view_features):
        """
        Transformer融合前向传播
        
        Args:
            multi_view_features (torch.Tensor): 多视角特征 [B, N, D]
            
        Returns:
            torch.Tensor: 融合后的特征 [B, D]
        """
        batch_size, num_views, feature_dim = multi_view_features.shape
        
        # 验证输入维度
        assert num_views == self.num_views, f"期望{self.num_views}个视角，得到{num_views}个"
        assert feature_dim == self.feature_dim, f"期望特征维度{self.feature_dim}，得到{feature_dim}"
        
        # === 步骤1: 准备CLS Token ===
        # 扩展CLS Token到当前batch大小
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # === 步骤2: 构建输入序列 ===
        # 将CLS Token添加到视角特征序列的开头
        sequence = torch.cat([cls_tokens, multi_view_features], dim=1)  # [B, 1+N, D]
        
        # === 步骤3: 添加位置编码 ===
        sequence = sequence + self.position_embeddings  # [B, 1+N, D]
        
        # === 步骤4: Transformer编码 ===
        # 通过Transformer编码器进行特征交互
        encoded_sequence = self.transformer_encoder(sequence)  # [B, 1+N, D]
        
        # === 步骤5: 提取CLS Token ===
        # CLS Token包含了所有视角的上下文信息
        cls_output = encoded_sequence[:, 0, :]  # [B, D] - 第0个位置是CLS Token
        
        # === 步骤6: 输出投影 ===
        # 通过投影层进一步处理CLS Token
        fused_features = self.output_projection(cls_output)  # [B, D]
        
        # === 步骤7: L2归一化 ===
        # 与其他融合方法保持一致，应用L2归一化
        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features
