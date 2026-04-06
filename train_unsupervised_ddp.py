"""
无监督多卡训练脚本（DDP）

核心特点：
1. 无监督训练：不使用标签信息
2. 损失组合：KL一致性 + 公平性正则化 + 特征去相关
3. 多卡并行：使用 DistributedDataParallel
4. 断点续训：支持从检查点恢复训练
5. 只保存最佳验证损失模型

运行方式：
  # 多卡训练（推荐）
  torchrun --nproc_per_node=2 train_unsupervised_ddp.py \
    --dataset oxford_pets \
    --config oxford_pets \
    --num_workers 8 \
    --seed 42

  # 单卡训练
  python train_unsupervised_ddp.py \
    --dataset cub \
    --config cub \
    --num_workers 8 \
    --seed 42

检查点保存：
- 保存位置: checkpoints/unsupervised/{dataset}/
- 最佳模型: {model_name}_best.pth (最佳验证损失)
- 最新检查点: {model_name}_latest.pth (用于断点续训)

说明：
- 配置文件位于 UN-STN-Config/{config}.yaml
- 使用 per-GPU 的 batch_size
- 使用 DistributedSampler 切分数据  多卡训练使用
- 仅在 rank0 打印与保存检查点
- 准确率仅用于监控，不保存单独的最佳准确率模型
"""

import os
import yaml
import argparse
import contextlib

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        class _Dummy:
            def __init__(self, it):
                self.it = it
            def __iter__(self):
                return iter(self.it)
            def set_postfix(self, *args, **kwargs):
                pass
        return _Dummy(iterable) if iterable is not None else _Dummy([])

# 项目模块
from clip import clip
from stn.multi_view_stn import MultiViewSTNModel
from data_preprocess import MultiViewDataset, prepare_clip_input
from train_multi_view_stn import compute_and_save_text_features
from main_stn import set_seed, get_stn_config_path


# ============================================================================
# 分布式工具函数
# ============================================================================

def is_dist_avail_and_initialized() -> bool:
    """检查分布式是否可用且已初始化"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """获取当前进程的rank"""
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    """判断是否为主进程"""
    return get_rank() == 0


@contextlib.contextmanager
def suppress_stdout_if_not_main():
    """在非主进程静默第三方函数的打印"""
    if is_main_process():
        yield
        return
    backup = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(backup, 1)
        os.close(devnull)
        os.close(backup)


def init_distributed():
    """根据 torchrun 环境变量初始化分布式"""
    if is_dist_avail_and_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # 检查GPU数量
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count == 0:
                raise RuntimeError("检测到CUDA可用，但设备数为0")
            if local_rank < 0 or local_rank >= device_count:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} 超出可见CUDA设备数量 {device_count}"
                )
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        return local_rank
    else:
        # 非分布式
        return int(os.environ.get("LOCAL_RANK", 0))



# ============================================================================
# 数据加载
# ============================================================================

def build_dataloader(dataset_name: str,
                     data_path: str,
                     split: str,
                     batch_size: int,
                     num_workers: int,
                     target_size: int,
                     scale_short_edge: int,
                     flip_prob: float,
                     center_crop: bool,
                     drop_last: bool,
                     persistent_workers: bool | None = None):
    """构建数据加载器"""
    helper = MultiViewDataset(
        data_root=data_path,
        dataset_name=dataset_name,
        split=split,
        target_size=target_size,
        scale_short_edge=scale_short_edge,
        flip_prob=flip_prob,
        center_crop=center_crop,
    )
    dataset = helper._create_base_dataset_with_transform()

    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'), drop_last=drop_last)
    else:
        sampler = None

    # persistent_workers: 训练集可开启以提速；验证集建议关闭
    if persistent_workers is None:
        pw = True if num_workers > 0 else False
    else:
        pw = persistent_workers and (num_workers > 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=pw,
        drop_last=drop_last,
    )
    return loader, sampler



# ============================================================================
# 无监督损失函数初始化
# ============================================================================

def setup_unsupervised_training(model, config, device):
    """
    设置无监督训练的优化器、损失函数和调度器
    
    Args:
        model: STN模型
        config: 配置字典
        device: 设备
        
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    from stn.loss_multi import MultiViewSTNLoss
    import torch.optim as optim
    
    # 从配置中读取损失权重
    stn_config = config.get('stn_config', {})
    logits_temp = stn_config.get('logits_temp', 0.07)
    
    # 无监督损失配置
    classification_weight = stn_config.get('classification_weight', 0.0)  # 无监督：禁用
    decorrelation_weight = stn_config.get('decorrelation_weight', 0.1)
    adaptive_weight = stn_config.get('adaptive_weight', 0.0)
    kl_consistency_weight = stn_config.get('kl_consistency_weight', 1.0)
    fairness_weight = stn_config.get('fairness_weight', 0.1)
    
    # 初始化损失函数（传入stn_config用于读取温度参数）
    criterion = MultiViewSTNLoss(
        logits_temp=logits_temp,
        classification_weight=classification_weight,
        decorrelation_weight=decorrelation_weight,
        adaptive_weight=adaptive_weight,
        kl_consistency_weight=kl_consistency_weight,
        fairness_weight=fairness_weight,
        stn_config=stn_config  # 传入配置字典
    )
    
    # 优化器配置
    training_config = config.get('training', {})
    lr = float(training_config.get('learning_rate', 1e-4))
    weight_decay = float(training_config.get('weight_decay', 1e-4))
    
    # 只优化STN组件
    trainable_params = []
    if hasattr(model, 'localization_network'):
        trainable_params.extend(list(model.localization_network.parameters()))
    if hasattr(model, 'fusion_module') and model.fusion_module is not None:
        trainable_params.extend(list(model.fusion_module.parameters()))
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    total_epochs = training_config.get('epochs', 50)
    warmup_epochs = training_config.get('warmup_epochs', 5)
    
    # CosineAnnealingLR
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=lr * 0.01
    )
    
    # Warmup调度器
    from train_multi_view_stn import WarmupScheduler
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        base_scheduler=cosine_scheduler,
        warmup_factor=0.1
    )
    
    return criterion, optimizer, scheduler


def get_two_stage_config(config: dict) -> dict:
    """读取两阶段训练配置，并提供稳健默认值。"""
    stn_config = config.get('stn_config', {})
    training_config = config.get('training', {})
    two_stage = stn_config.get('two_stage', {})
    if not isinstance(two_stage, dict):
        two_stage = {}

    base_temp = float(stn_config.get('logits_temp', 0.07))
    warmup_epochs = int(two_stage.get('warmup_epochs', training_config.get('warmup_epochs', 5)))

    # 周期刷新间隔（单位：epoch）
    target_update_interval_epochs = max(
        1,
        int(two_stage.get('target_update_interval_epochs', 1))
    )

    return {
        # 阶段切换
        'warmup_epochs': max(0, warmup_epochs),
        'target_update_interval_epochs': target_update_interval_epochs,
        # 阶段一：Global -> (Local, Fused)
        'teacher_temp': float(two_stage.get('teacher_temp', base_temp)),
        'warmup_student_temp': float(two_stage.get('warmup_student_temp', base_temp)),
        'warmup_local_weight': float(two_stage.get('warmup_local_weight', 1.0)),
        'warmup_fused_weight': float(two_stage.get('warmup_fused_weight', 1.0)),
        # 阶段二：periodic target update
        'dec_target_temp': float(two_stage.get('dec_target_temp', 0.05)),
        'dec_student_temp': float(two_stage.get('dec_student_temp', 0.1)),
    }


def build_unsupervised_model_name(dataset_name: str, config: dict) -> str:
    """按当前训练逻辑构建无监督模型文件名前缀。"""
    model_size = config['model_size']
    stn_cfg = config.get('stn_config', {})
    two_stage_cfg = get_two_stage_config(config)

    model_name_suffix = (
        f"views{stn_cfg.get('num_views', 4)}"
        f"_{stn_cfg.get('fusion_mode', 'simple')}"
        f"_dim{stn_cfg.get('hidden_dim', 512)}"
    )

    stage2_kl_weight = float(stn_cfg.get('kl_consistency_weight', 1.0))
    loss_suffix = (
        f"twostage_w{two_stage_cfg['warmup_epochs']}"
        f"_mperiodic_u{two_stage_cfg['target_update_interval_epochs']}"
        f"_tg{two_stage_cfg['teacher_temp']:.3f}"
        f"_td{two_stage_cfg['dec_target_temp']:.3f}"
        f"_kl{stage2_kl_weight:.2f}"
    )

    if stn_cfg.get('fairness_weight', 0.0) > 0:
        loss_suffix += f"_fair{stn_cfg.get('fairness_weight', 0.0)}"
    if stn_cfg.get('decorrelation_weight', 0.0) > 0:
        loss_suffix += f"_dec{stn_cfg.get('decorrelation_weight', 0.0)}"

    base_model_name = (
        f"multi_view_stn_{dataset_name}_{model_size.replace('/', '_')}"
        f"_{model_name_suffix}_{loss_suffix}"
    )
    return base_model_name


def build_unsupervised_checkpoint_paths(dataset_name: str, config: dict) -> dict:
    """构建无监督训练的checkpoint路径（best/latest）。"""
    base_model_name = build_unsupervised_model_name(dataset_name, config)
    ckpt_dir = os.path.join('checkpoints', 'unsupervised', dataset_name)
    return {
        'dir': ckpt_dir,
        'base_model_name': base_model_name,
        'best': os.path.join(ckpt_dir, f"{base_model_name}_best.pth"),
        'latest': os.path.join(ckpt_dir, f"{base_model_name}_latest.pth"),
    }


def _kl_from_logits(student_logits: torch.Tensor,
                    target_probs: torch.Tensor,
                    student_temp: float) -> torch.Tensor:
    """计算 KL(target || student) 的 batchmean。"""
    temp = max(student_temp, 1e-6)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    return F.kl_div(student_log_probs, target_probs, reduction='batchmean')


def _multi_view_kl_from_logits(view_logits: torch.Tensor,
                               target_probs: torch.Tensor,
                               student_temp: float) -> torch.Tensor:
    """计算多视角 KL(target || view_i) 并对所有视角做 batchmean。"""
    batch_size, num_views, num_classes = view_logits.shape
    temp = max(student_temp, 1e-6)
    view_log_probs = F.log_softmax(view_logits / temp, dim=-1)
    target_expand = target_probs.unsqueeze(1).expand(-1, num_views, -1)
    return F.kl_div(
        view_log_probs.reshape(-1, num_classes),
        target_expand.reshape(-1, num_classes),
        reduction='batchmean'
    )


def compute_global_logits(model_without_ddp,
                          images_448: torch.Tensor,
                          text_features: torch.Tensor) -> torch.Tensor:
    """
    计算冻结CLIP的全局logits（teacher）。
    输入 images_448 已经是标准化图像，只需下采样到224。
    """
    with torch.no_grad():
        clip_input = prepare_clip_input(images_448, clip_size=224)
        global_features = model_without_ddp.clip_model.encode_image(clip_input)
        # 自定义CLIP ViT实现返回(cls_features, patch_features)，这里只需要cls_features
        if isinstance(global_features, (tuple, list)):
            global_features = global_features[0]
        global_features = global_features.float()
        global_features = F.normalize(global_features, dim=-1)
        global_logits_raw = global_features @ text_features
    return global_logits_raw


def build_or_refresh_periodic_teacher(teacher_model,
                                      student_model_without_ddp,
                                      device: torch.device):
    """
    构建或刷新周期性目标分布teacher模型。
    teacher在stage2中以固定周期更新参数，用于提供相对静态的目标分布。
    """
    if teacher_model is None:
        with suppress_stdout_if_not_main():
            teacher_model = MultiViewSTNModel(
                clip_model=student_model_without_ddp.clip_model,
                config=student_model_without_ddp.config,
                num_views=student_model_without_ddp.num_views,
            ).to(device)
            teacher_model = teacher_model.float()

    teacher_model.load_state_dict(student_model_without_ddp.state_dict(), strict=True)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    return teacher_model


def compute_periodic_target_probs(teacher_model,
                                  images_448: torch.Tensor,
                                  text_features: torch.Tensor,
                                  target_temp: float) -> torch.Tensor:
    """使用周期性teacher计算当前batch目标分布。"""
    with torch.no_grad():
        teacher_fused_features, _ = teacher_model(images_448, mode='train')
        teacher_fused_features = F.normalize(teacher_fused_features.float(), dim=-1)
        teacher_logits = teacher_fused_features @ text_features
        target_probs = F.softmax(teacher_logits / max(target_temp, 1e-6), dim=-1)
    return target_probs


def compute_two_stage_unsupervised_loss(criterion,
                                        fused_features: torch.Tensor,
                                        view_features: torch.Tensor,
                                        text_features: torch.Tensor,
                                        labels: torch.Tensor,
                                        global_logits_raw: torch.Tensor,
                                        epoch: int,
                                        global_step: int,
                                        config: dict,
                                        periodic_target_probs: torch.Tensor | None = None):
    """
    两阶段无监督目标：
    - 阶段一（warmup）：Global -> Local + Global -> Fused
    - 阶段二（periodic）：周期刷新teacher分布，最小化 Local <- Fused_teacher
    """
    if view_features is None:
        raise ValueError("两阶段训练需要 view_features，当前为 None")

    two_stage_cfg = get_two_stage_config(config)

    # 统一归一化，确保logits尺度稳定
    fused_features = F.normalize(fused_features.float(), dim=-1)
    view_features = F.normalize(view_features.float(), dim=-1)

    fused_logits_raw = fused_features @ text_features              # [B, C]
    view_logits_raw = torch.matmul(view_features, text_features)   # [B, N, C]

    # 初始化loss详情
    loss_details = {
        'phase': 'warmup' if epoch < two_stage_cfg['warmup_epochs'] else 'dec',
        'warmup_local': 0.0,
        'warmup_local_weighted': 0.0,
        'warmup_fused': 0.0,
        'warmup_fused_weighted': 0.0,
        'dec_local': 0.0,
        'dec_local_weighted': 0.0,
        'dec_fused': 0.0,
        'dec_fused_weighted': 0.0,
        'dec_local_active': 0.0,
        'dec_fused_active': 0.0,
        'classification': 0.0,
        'classification_weighted': 0.0,
        'adaptive': 0.0,
        'adaptive_weighted': 0.0,
        'decorrelation': 0.0,
        'decorrelation_weighted': 0.0,
        'fairness': 0.0,
        'fairness_weighted': 0.0,
        'stage_total': 0.0,
        'regularization_total': 0.0,
        'total': 0.0,
    }

    # === 阶段损失 ===
    if epoch < two_stage_cfg['warmup_epochs']:
        # 阶段一：原图全局分布指导局部与融合
        target_global = F.softmax(global_logits_raw / max(two_stage_cfg['teacher_temp'], 1e-6), dim=-1).detach()

        warmup_local = _multi_view_kl_from_logits(
            view_logits=view_logits_raw,
            target_probs=target_global,
            student_temp=two_stage_cfg['warmup_student_temp']
        )
        warmup_fused = _kl_from_logits(
            student_logits=fused_logits_raw,
            target_probs=target_global,
            student_temp=two_stage_cfg['warmup_student_temp']
        )

        stage_loss = (
            two_stage_cfg['warmup_local_weight'] * warmup_local +
            two_stage_cfg['warmup_fused_weight'] * warmup_fused
        )

        loss_details['phase'] = 'warmup'
        loss_details['warmup_local'] = float(warmup_local.item())
        loss_details['warmup_local_weighted'] = float((two_stage_cfg['warmup_local_weight'] * warmup_local).item())
        loss_details['warmup_fused'] = float(warmup_fused.item())
        loss_details['warmup_fused_weighted'] = float((two_stage_cfg['warmup_fused_weight'] * warmup_fused).item())
    else:
        # 阶段二（periodic）：使用周期性刷新的teacher分布监督局部视角
        if periodic_target_probs is None:
            # 容错回退：若未提供teacher目标，则退化为当前融合分布
            target_from_teacher = F.softmax(
                fused_logits_raw.detach() / max(two_stage_cfg['dec_target_temp'], 1e-6),
                dim=-1
            )
        else:
            target_from_teacher = periodic_target_probs.detach()

        dec_local = _multi_view_kl_from_logits(
            view_logits=view_logits_raw,
            target_probs=target_from_teacher,
            student_temp=two_stage_cfg['dec_student_temp']
        )
        stage2_kl_weight = float(getattr(criterion, 'kl_consistency_weight', 1.0))
        weighted_dec_local = stage2_kl_weight * dec_local

        stage_loss = weighted_dec_local
        loss_details['phase'] = 'dec_periodic'
        loss_details['dec_local_active'] = float(weighted_dec_local.item())
        loss_details['dec_local'] = float(dec_local.item())
        loss_details['dec_local_weighted'] = float(weighted_dec_local.item())
        loss_details['dec_fused'] = 0.0
        loss_details['dec_fused_weighted'] = 0.0

    # === 兼容已有正则项（可选）===
    reg_total = torch.tensor(0.0, device=fused_logits_raw.device, dtype=fused_logits_raw.dtype)

    if getattr(criterion, 'classification_loss', None) is not None and getattr(criterion, 'classification_weight', 0.0) > 0:
        cls = criterion.classification_loss(fused_logits_raw, labels)
        weighted_cls = cls * criterion.classification_weight
        reg_total += weighted_cls
        loss_details['classification'] = float(cls.item())
        loss_details['classification_weighted'] = float(weighted_cls.item())

    if getattr(criterion, 'adaptive_loss', None) is not None and getattr(criterion, 'adaptive_weight', 0.0) > 0:
        adp = criterion.adaptive_loss(view_features, text_features, labels)
        weighted_adp = adp * criterion.adaptive_weight
        reg_total += weighted_adp
        loss_details['adaptive'] = float(adp.item())
        loss_details['adaptive_weighted'] = float(weighted_adp.item())

    if getattr(criterion, 'decorrelation_loss', None) is not None and getattr(criterion, 'decorrelation_weight', 0.0) > 0:
        dec = criterion.decorrelation_loss(view_features)
        weighted_dec = dec * criterion.decorrelation_weight
        reg_total += weighted_dec
        loss_details['decorrelation'] = float(dec.item())
        loss_details['decorrelation_weighted'] = float(weighted_dec.item())

    if getattr(criterion, 'fairness_loss', None) is not None and getattr(criterion, 'fairness_weight', 0.0) > 0:
        fair = criterion.fairness_loss(view_features, text_features)
        weighted_fair = fair * criterion.fairness_weight
        reg_total += weighted_fair
        loss_details['fairness'] = float(fair.item())
        loss_details['fairness_weighted'] = float(weighted_fair.item())

    total_loss = stage_loss + reg_total
    loss_details['stage_total'] = float(stage_loss.item())
    loss_details['regularization_total'] = float(reg_total.item())
    loss_details['total'] = float(total_loss.item())

    return total_loss, loss_details, fused_logits_raw



# ============================================================================
# 无监督训练一个epoch
# ============================================================================

def unsupervised_train_one_epoch(model: DDP,
                                 optimizer: torch.optim.Optimizer,
                                 criterion,
                                 train_loader: DataLoader,
                                 device: torch.device,
                                 epoch: int,
                                 config: dict,
                                 text_features: torch.Tensor,
                                 teacher_model=None):
    """
    无监督训练一个epoch
    
    注意：不使用标签信息，labels仅用于计算准确率（监控用）
    """
    model.train()
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    temperature = float(config['stn_config'].get('logits_temp', 0.07))
    two_stage_cfg = get_two_stage_config(config)

    # 梯度裁剪配置
    max_grad_norm = float(config['training'].get('max_grad_norm', 1.0))
    grad_norm_type = float(config['training'].get('grad_norm_type', 2.0))

    model_without_ddp = model.module if hasattr(model, 'module') else model

    # 仅在主进程显示进度条
    iterable = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False) if is_main_process() else train_loader

    for batch_idx, (images, labels) in enumerate(iterable):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        global_step = epoch * len(train_loader) + batch_idx

        # 前向传播：复用模型内部已提取的原图CLS特征，避免重复提取全局特征
        fused_features, view_features, original_features = model(
            images,
            mode='train',
            return_original_features=True,
        )
        global_logits_raw = torch.matmul(original_features, text_features)

        periodic_target_probs = None
        if (
            epoch >= two_stage_cfg['warmup_epochs']
            and teacher_model is not None
        ):
            periodic_target_probs = compute_periodic_target_probs(
                teacher_model=teacher_model,
                images_448=images,
                text_features=text_features,
                target_temp=two_stage_cfg['dec_target_temp'],
            )

        # 两阶段损失（阶段一: Global指导；阶段二: DEC式交替）
        loss, loss_details, fused_logits_raw = compute_two_stage_unsupervised_loss(
            criterion=criterion,
            fused_features=fused_features,
            view_features=view_features,
            text_features=text_features,
            labels=labels,
            global_logits_raw=global_logits_raw,
            epoch=epoch,
            global_step=global_step,
            config=config,
            periodic_target_probs=periodic_target_probs,
        )

        # 仅用于监控准确率
        logits_for_acc = fused_logits_raw / temperature

        loss.backward()

        # 梯度连续化优化
        for p in model_without_ddp.parameters():
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()

        # 梯度裁剪
        clip_params = []
        if hasattr(model_without_ddp, 'localization_network'):
            clip_params.extend([p for p in model_without_ddp.localization_network.parameters() if p.requires_grad])
        if hasattr(model_without_ddp, 'fusion_module') and model_without_ddp.fusion_module is not None:
            clip_params.extend([p for p in model_without_ddp.fusion_module.parameters() if p.requires_grad])
        if clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_grad_norm, norm_type=grad_norm_type)

        optimizer.step()

        # 统计（准确率仅用于监控）
        with torch.no_grad():
            bs = labels.size(0)
            total_loss_sum += loss.item() * bs
            preds = logits_for_acc.argmax(dim=-1)  # 使用缩放后的相似度计算准确率
            total_correct += (preds == labels).sum().item()
            total_samples += bs

        # 更新进度条
        if is_main_process() and hasattr(iterable, 'set_postfix'):
            current_lr = optimizer.param_groups[0]['lr']
            progress_dict = {
                'Total': f'{loss_details["total"]:.3f}',
                'Phase': loss_details.get('phase', 'N/A'),
                'Acc': f'{(preds == labels).float().mean().item():.3f}',
                'LR': f'{current_lr:.1e}'
            }

            # 显示两阶段损失分量
            if loss_details.get('phase') == 'warmup':
                progress_dict['WLoc'] = f"{loss_details.get('warmup_local', 0.0):.3f}/{loss_details.get('warmup_local_weighted', 0.0):.3f}"
                progress_dict['WFus'] = f"{loss_details.get('warmup_fused', 0.0):.3f}/{loss_details.get('warmup_fused_weighted', 0.0):.3f}"
            elif loss_details.get('phase') == 'dec_periodic':
                progress_dict['DecP'] = f"{loss_details.get('dec_local', 0.0):.3f}/{loss_details.get('dec_local_active', 0.0):.3f}"

            # 显示正则项损失分量
            if getattr(criterion, 'classification_weight', 0.0) > 0:
                progress_dict['Cls'] = f"{loss_details.get('classification', 0.0):.3f}/{loss_details.get('classification_weighted', 0.0):.3f}"
            if getattr(criterion, 'fairness_weight', 0.0) > 0:
                progress_dict['Fair'] = f"{loss_details.get('fairness', 0.0):.3f}/{loss_details.get('fairness_weighted', 0.0):.3f}"
            if getattr(criterion, 'decorrelation_weight', 0.0) > 0:
                progress_dict['Dec'] = f"{loss_details.get('decorrelation', 0.0):.3f}/{loss_details.get('decorrelation_weighted', 0.0):.3f}"
            if getattr(criterion, 'adaptive_weight', 0.0) > 0:
                progress_dict['Adp'] = f"{loss_details.get('adaptive', 0.0):.3f}/{loss_details.get('adaptive_weighted', 0.0):.3f}"
            
            iterable.set_postfix(progress_dict)

    # 关闭进度条
    if is_main_process() and hasattr(iterable, 'close'):
        try:
            iterable.close()
        except Exception:
            pass

    # 全局归并
    if is_dist_avail_and_initialized():
        t = torch.tensor([total_loss_sum, total_correct, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_sum, total_correct, total_samples = t.tolist()

    avg_loss = total_loss_sum / max(total_samples, 1.0)
    avg_acc = total_correct / max(total_samples, 1.0)
    return float(avg_loss), float(avg_acc)



# ============================================================================
# 无监督验证
# ============================================================================

def unsupervised_validate(model: DDP,
                          criterion,
                          val_loader: DataLoader,
                          device: torch.device,
                          epoch: int,
                          config: dict,
                          text_features: torch.Tensor,
                          teacher_model=None):
    """无监督验证"""
    model.eval()

    temperature = float(config['stn_config'].get('logits_temp', 0.07))
    two_stage_cfg = get_two_stage_config(config)
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model_without_ddp = model.module if hasattr(model, 'module') else model
        iterable = tqdm(val_loader, desc="Val", leave=False) if is_main_process() else val_loader
        
        for batch_idx, (images, labels) in enumerate(iterable):
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            global_step = epoch * len(val_loader) + batch_idx

            # 验证时复用模型内部原图CLS特征，避免重复提取
            fused_features, view_features, original_features = model(
                images,
                mode='train',
                return_original_features=True,
            )
            global_logits_raw = torch.matmul(original_features, text_features)

            periodic_target_probs = None
            if (
                epoch >= two_stage_cfg['warmup_epochs']
                and teacher_model is not None
            ):
                periodic_target_probs = compute_periodic_target_probs(
                    teacher_model=teacher_model,
                    images_448=images,
                    text_features=text_features,
                    target_temp=two_stage_cfg['dec_target_temp'],
                )

            loss, loss_details, fused_logits_raw = compute_two_stage_unsupervised_loss(
                criterion=criterion,
                fused_features=fused_features,
                view_features=view_features,
                text_features=text_features,
                labels=labels,
                global_logits_raw=global_logits_raw,
                epoch=epoch,
                global_step=global_step,
                config=config,
                periodic_target_probs=periodic_target_probs,
            )

            similarity = fused_logits_raw / temperature

            bs = labels.size(0)
            total_loss_sum += loss.item() * bs
            preds = similarity.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += bs

            if is_main_process() and hasattr(iterable, 'set_postfix'):
                progress_dict = {
                    'loss': f"{loss.item():.3f}",
                    'phase': loss_details.get('phase', 'N/A'),
                    'acc': f"{(preds == labels).float().mean().item():.3f}"
                }
                if loss_details.get('phase') == 'warmup':
                    progress_dict['wloc'] = f"{loss_details.get('warmup_local', 0.0):.3f}/{loss_details.get('warmup_local_weighted', 0.0):.3f}"
                    progress_dict['wfus'] = f"{loss_details.get('warmup_fused', 0.0):.3f}/{loss_details.get('warmup_fused_weighted', 0.0):.3f}"
                else:
                    progress_dict['decp'] = f"{loss_details.get('dec_local', 0.0):.3f}/{loss_details.get('dec_local_weighted', 0.0):.3f}"
                if getattr(criterion, 'classification_weight', 0.0) > 0:
                    progress_dict['cls'] = f"{loss_details.get('classification', 0.0):.3f}/{loss_details.get('classification_weighted', 0.0):.3f}"
                if getattr(criterion, 'fairness_weight', 0.0) > 0:
                    progress_dict['fair'] = f"{loss_details.get('fairness', 0.0):.3f}/{loss_details.get('fairness_weighted', 0.0):.3f}"
                if getattr(criterion, 'decorrelation_weight', 0.0) > 0:
                    progress_dict['dec'] = f"{loss_details.get('decorrelation', 0.0):.3f}/{loss_details.get('decorrelation_weighted', 0.0):.3f}"
                if getattr(criterion, 'adaptive_weight', 0.0) > 0:
                    progress_dict['adp'] = f"{loss_details.get('adaptive', 0.0):.3f}/{loss_details.get('adaptive_weighted', 0.0):.3f}"

                iterable.set_postfix(progress_dict)

        # 关闭进度条
        if is_main_process() and hasattr(iterable, 'close'):
            try:
                iterable.close()
            except Exception:
                pass

    # 全局归并
    if is_dist_avail_and_initialized():
        t = torch.tensor([total_loss_sum, total_correct, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_sum, total_correct, total_samples = t.tolist()

    avg_loss = total_loss_sum / max(total_samples, 1.0)
    avg_acc = total_correct / max(total_samples, 1.0)
    return float(avg_loss), float(avg_acc)



# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="无监督DDP多视角STN-CLIP训练")
    parser.add_argument('--dataset', type=str, default='imagenet', help='数据集名称')
    parser.add_argument('--config', type=str, default=None, help='配置文件名（UN-STN-Config目录下）')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 初始化分布式
    local_rank = init_distributed()
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    if is_main_process():
        print(f"[无监督DDP] World Size: {dist.get_world_size() if is_dist_avail_and_initialized() else 1}")
        print(f"[无监督DDP] Device: {device}")
        print(f"[无监督DDP] Dataset: {args.dataset}")
        print(f"[无监督DDP] 训练模式: 无监督（不使用标签）")

    # 设置随机种子
    set_seed(args.seed + get_rank())

    # 读取配置（从UN-STN-Config目录）
    if args.config:
        # 用户指定了配置文件
        if args.config.startswith('UN-STN-Config/'):
            config_path = args.config
        elif args.config.endswith('.yaml'):
            config_path = f"UN-STN-Config/{args.config}"
        else:
            config_path = f"UN-STN-Config/{args.config}.yaml"
    else:
        # 默认使用数据集名称作为配置文件名
        config_path = f"UN-STN-Config/{args.dataset}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['dataset'] = args.dataset
    
    if is_main_process():
        print(f"📋 加载配置: {config_path}")

    # 加载CLIP模型
    model_size = config['model_size']
    if is_main_process():
        print(f"Loading CLIP model: {model_size}")
    
    with suppress_stdout_if_not_main():
        clip_model, _ = clip.load(model_size, device=device)
    
    clip_model = clip_model.float()

    # 构建STN模型
    num_views = config['stn_config'].get('num_views', 4)
    stn_model = MultiViewSTNModel(clip_model, config['stn_config'], num_views=num_views).to(device)
    stn_model = stn_model.float()

    # 设置无监督训练
    with suppress_stdout_if_not_main():
        criterion, optimizer, scheduler = setup_unsupervised_training(stn_model, config, device)

    # 构建DDP包装
    if is_dist_avail_and_initialized():
        stn_model = DDP(
            stn_model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=True,  # 启用未使用参数检测（融合模块中某些参数可能不参与所有损失）
            gradient_as_bucket_view=True,
        )

    # 构建数据加载器
    bs = int(config['training']['batch_size'])
    train_loader, train_sampler = build_dataloader(
        dataset_name=args.dataset,
        data_path=config['data_path'],
        split='train',
        batch_size=bs,
        num_workers=args.num_workers,
        target_size=448,
        scale_short_edge=512,
        flip_prob=0.5,
        center_crop=False,
        drop_last=True,
        persistent_workers=None,
    )
    val_loader, val_sampler = build_dataloader(
        dataset_name=args.dataset,
        data_path=config['data_path'],
        split='val',
        batch_size=bs,
        num_workers=args.num_workers,
        target_size=448,
        scale_short_edge=512,
        flip_prob=0.0,
        center_crop=True,
        drop_last=False,
        persistent_workers=False,
    )
    
    if is_main_process():
        train_dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'Unknown'
        val_dataset_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 'Unknown'
        print(f"📊 数据集信息:")
        print(f"   - 训练集: {train_dataset_size} 样本, {len(train_loader)} batches")
        print(f"   - 验证集: {val_dataset_size} 样本, {len(val_loader)} batches")

    # 加载文本特征
    model_name = model_size.replace('/', '_')
    text_features_path = f"text_features/{args.dataset}_{model_name}.pt"
    os.makedirs("text_features", exist_ok=True)

    if is_main_process():
        try:
            tf = torch.load(text_features_path, map_location='cpu')
            print(f"✅ 已找到预计算文本特征: {text_features_path}")
        except FileNotFoundError:
            print(f"⚠️ 未找到文本特征，开始计算: {text_features_path}")
            tf = compute_and_save_text_features(
                clip_model=(stn_model.module.clip_model if hasattr(stn_model, 'module') else stn_model.clip_model),
                dataset_name=args.dataset,
                model_size=model_size,
                device=device,
                text_scale=7.39,
                use_weighted_aggregation=True,
            )
            if tf is None:
                raise RuntimeError("文本特征计算失败")
    
    # 同步
    if is_dist_avail_and_initialized():
        dist.barrier()

    # 加载文本特征到设备
    all_classes_text_features = torch.load(text_features_path, map_location=device).float().to(device)
    all_classes_text_features.requires_grad_(False)

    # 训练配置
    total_epochs = int(config['training']['epochs'])
    patience = int(config['training'].get('patience', 5))
    two_stage_cfg = get_two_stage_config(config)

    # 构建checkpoint路径（训练/测试统一命名来源）
    stn_cfg = config.get('stn_config', {})
    stage2_kl_weight = float(stn_cfg.get('kl_consistency_weight', 1.0))
    ckpt_paths = build_unsupervised_checkpoint_paths(args.dataset, config)
    ckpt_dir = ckpt_paths['dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    # 只保存最佳验证集损失的模型和最新检查点
    ckpt_best_loss = ckpt_paths['best']
    ckpt_latest = ckpt_paths['latest']

    # 早停和最佳模型跟踪（只跟踪最佳损失）
    best_val_loss = float('inf')
    best_loss_epoch = 0
    best_loss_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    # 断点续训
    if os.path.exists(ckpt_latest):
        if is_main_process():
            print(f"\n🔄 发现检查点，尝试恢复训练: {ckpt_latest}")
        
        try:
            checkpoint = torch.load(ckpt_latest, map_location=device)
            
            if hasattr(stn_model, 'module'):
                stn_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                stn_model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_loss_acc = checkpoint.get('best_loss_acc', 0.0)
            best_loss_epoch = checkpoint.get('best_loss_epoch', 0)
            patience_counter = checkpoint.get('patience_counter', 0)
            
            if is_main_process():
                print(f"✅ 成功恢复训练状态:")
                print(f"   - 起始Epoch: {start_epoch}/{total_epochs}")
                print(f"   - 最佳Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}, Acc={best_loss_acc:.3f})")
                print(f"   - 早停计数: {patience_counter}/{patience}")
        
        except Exception as e:
            if is_main_process():
                print(f"⚠️ 恢复检查点失败: {e}")
                print(f"   从头开始训练...")
            start_epoch = 0
    else:
        if is_main_process():
            print(f"\n🆕 未找到检查点，从头开始训练")

    if is_main_process():
        print(f"\n开始无监督分布式训练：")
        print(f"  - 总轮数: {total_epochs}")
        print(f"  - 起始Epoch: {start_epoch}")
        print(f"  - 耐心值: {patience}")
        print(f"  - Batch Size: {bs} (per GPU)")
        print(f"  - 学习率: {config['training'].get('learning_rate', 1e-4)}")
        print(f"  - 两阶段训练:")
        print(f"    * Warmup轮数: {two_stage_cfg['warmup_epochs']}")
        print(f"    * 阶段一(Global->Local/Fused): 温度(T_teacher={two_stage_cfg['teacher_temp']}, T_student={two_stage_cfg['warmup_student_temp']}), 权重(Local={two_stage_cfg['warmup_local_weight']}, Fused={two_stage_cfg['warmup_fused_weight']})")
        print(f"    * 阶段二(周期目标): 温度(T_target={two_stage_cfg['dec_target_temp']}, T_student={two_stage_cfg['dec_student_temp']}), KL权重(kl_consistency_weight)={stage2_kl_weight}, 目标刷新间隔={two_stage_cfg['target_update_interval_epochs']} epochs")
        print(f"  - 额外正则:")
        print(f"    * 公平性正则化: {stn_cfg.get('fairness_weight', 0.0)}")
        print(f"    * 特征去相关: {stn_cfg.get('decorrelation_weight', 0.0)}")
        print(f"  - 模型保存: {ckpt_dir}")
        print()

    teacher_model = None

    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        model_without_ddp = stn_model.module if hasattr(stn_model, 'module') else stn_model

        # 阶段二周期性刷新teacher目标模型（按epoch刷新）
        if epoch >= two_stage_cfg['warmup_epochs']:
            stage2_epoch_idx = epoch - two_stage_cfg['warmup_epochs']
            should_refresh_teacher = (
                teacher_model is None or
                (stage2_epoch_idx % two_stage_cfg['target_update_interval_epochs'] == 0)
            )
            if should_refresh_teacher:
                teacher_model = build_or_refresh_periodic_teacher(
                    teacher_model=teacher_model,
                    student_model_without_ddp=model_without_ddp,
                    device=device,
                )
                if is_main_process():
                    print(
                        f"🔄 阶段二周期目标已刷新: epoch={epoch+1}, "
                        f"refresh_interval={two_stage_cfg['target_update_interval_epochs']} epochs"
                    )

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = unsupervised_train_one_epoch(
            model=stn_model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            config=config,
            text_features=all_classes_text_features,
            teacher_model=teacher_model,
        )

        val_loss, val_acc = unsupervised_validate(
            model=stn_model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            config=config,
            text_features=all_classes_text_features,
            teacher_model=teacher_model,
        )

        if scheduler is not None:
            scheduler.step()

        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{total_epochs} | Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                  f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | lr={current_lr:.2e}")

            # 只保存最佳验证损失的模型
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_loss_acc = val_acc
                best_loss_epoch = epoch + 1
                patience_counter = 0

                to_save_state = stn_model.module.state_dict() if hasattr(stn_model, 'module') else stn_model.state_dict()
                torch.save(to_save_state, ckpt_best_loss)
                print(f"  🎯 验证损失改善 -{improvement:.6f}! 新最佳Loss: {best_val_loss:.6f}, Acc: {best_loss_acc:.3f} (第{best_loss_epoch}轮)")
                print(f"  💾 已保存最佳模型: {ckpt_best_loss}")
            else:
                patience_counter += 1
                print(f"  ⏳ 验证损失未改善 {patience_counter}/{patience} 轮")
            
            # 保存最新检查点（用于断点续训）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': stn_model.module.state_dict() if hasattr(stn_model, 'module') else stn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
                'best_loss_acc': best_loss_acc,
                'best_loss_epoch': best_loss_epoch,
                'patience_counter': patience_counter,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, ckpt_latest)
            print(f"  💾 已保存最新检查点: {ckpt_latest}")

            if patience_counter >= patience:
                print("🛑 早停触发，结束训练")
                break

        # 同步早停决定（确保所有进程同步）
        should_stop = False
        if is_dist_avail_and_initialized():
            # 主进程决定是否早停
            stop_flag = torch.tensor(1 if (is_main_process() and patience_counter >= patience) else 0, 
                                    dtype=torch.int32, device=device)
            dist.broadcast(stop_flag, src=0)
            should_stop = (stop_flag.item() == 1)
        else:
            # 单卡训练
            should_stop = (patience_counter >= patience)
        
        if should_stop:
            if is_main_process():
                print("🛑 所有进程同步早停")
            break

        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 训练完成
    if is_main_process():
        print("\n🎉 无监督分布式训练完成")
        print(f"🏆 最佳模型: Epoch {best_loss_epoch}, Val Loss={best_val_loss:.6f}, Val Acc={best_loss_acc:.3f}")
        print(f"💾 模型保存位置: {ckpt_best_loss}")

    # 清理
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    try:
        del train_loader
        del val_loader
    except Exception:
        pass

    if is_dist_avail_and_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == '__main__':
    main()
