"""
底层训练工具库（中文注释版）

该模块包含训练/评估过程中常用的工具函数与类（已统一为 step-based 调度器）：
- 基于 step 的 Cosine 调度器（带线性 warmup）
- 优化器构造（按参数分组设置不同权重衰减与头部学习率倍率）
- 训练单轮函数（包含梯度裁剪与梯度范数记录）
- 评估函数与若干辅助工具（结果保存）

注：本文件的注释和 docstring 使用中文，便于团队阅读与维护，
    代码逻辑未做功能性修改，仅添加说明。
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import json
from typing import Optional


#训练一个epoch
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, max_grad_norm: float = 1.0, ema=None):
    """
    训练一个 epoch 的主循环

        主要功能和保护措施：
            - forward/backward/optimizer.step()
            - 支持基于 step 的调度器（若传入 scheduler，会在每个 batch 后调用 scheduler.step()）
            - 梯度裁剪（clip_grad_norm_）以防梯度爆炸

    参数：
      model: torch.nn.Module
      dataloader: DataLoader
      criterion: 损失函数
      optimizer: 优化器
      device: 运行设备（'cpu' 或 'cuda'）
      epoch: 当前 epoch 索引（仅用于进度条显示）
      scheduler: 可选，若为 step-based 调度器则在每个 batch 后调用
      max_grad_norm: 梯度裁剪阈值

    返回： (avg_loss, avg_acc)
    """
    model.train()

    # 使用加权累计统计本 epoch 的平均损失与准确率，并记录梯度范数
    loss_sum = 0.0
    acc_sum = 0.0
    sample_count = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        try:
            images, labels = batch
        except Exception as e:
            # DataLoader worker may have raised an exception; surface it
            raise RuntimeError(f"Failed to read batch {batch_idx} from dataloader: {e}") from e
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # Fast-fail on NaN/Inf in logits or loss with diagnostics
        if not torch.isfinite(loss):
            # collect lightweight diagnostics
            img_min = float(torch.min(images).cpu()) if images.numel() > 0 else float('nan')
            img_max = float(torch.max(images).cpu()) if images.numel() > 0 else float('nan')
            img_mean = float(torch.mean(images).cpu()) if images.numel() > 0 else float('nan')
            unique_labels = torch.unique(labels).cpu().tolist() if labels.numel() > 0 else []
            raise RuntimeError(
                f"Detected non-finite loss at epoch={epoch}, batch={batch_idx}. "
                f"loss={loss}. Image stats: min={img_min:.6e}, max={img_max:.6e}, mean={img_mean:.6e}. "
                f"labels_sample={unique_labels[:10]}"
            )

        # Backward pass
        loss.backward()

        # 计算并记录梯度范数；若配置了 max_grad_norm>0，则同时进行裁剪
        # 优先使用 clip_grad_norm_（既计算范数又裁剪）；在只计算时对稀疏/密集梯度做显式处理以避免异常
        if max_grad_norm is None or max_grad_norm <= 0:
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                # 对稀疏梯度使用 coalesce().values() 再求范数
                if getattr(p.grad, 'is_sparse', False):
                    grad_vals = p.grad.coalesce().values()
                    param_norm = grad_vals.detach().float().norm(2)
                else:
                    param_norm = p.grad.detach().float().norm(2)
                total_norm_sq += (param_norm.item() ** 2)
            total_norm = total_norm_sq ** 0.5
        else:
            # clip_grad_norm_ 返回裁剪前的全局范数
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        grad_norm_sum += float(total_norm)
        grad_norm_count += 1

        # 梯度更新
        optimizer.step()

        # 如果为 step-based 调度器（基于 iteration），则在每个 batch 后调用 scheduler.step()
        if scheduler:
            scheduler.step()

        # EMA 更新（若启用）
        if ema is not None:
            try:
                ema.update(model)
            except Exception:
                # EMA 失败不影响主流程
                pass

        # Compute accuracy（按样本数加权累计）
        pred = logits.argmax(dim=1)
        batch_acc = (pred == labels).float().mean().item()

        loss_sum += float(loss.item()) * batch_size
        acc_sum += float(batch_acc) * batch_size
        sample_count += batch_size

    # Update progress bar（显示到当前为止的累计平均和当前梯度范数）
    avg_loss = loss_sum / max(1, sample_count)
    avg_acc = acc_sum / max(1, sample_count)
    avg_grad_norm = grad_norm_sum / max(1, grad_norm_count) if grad_norm_count > 0 else 0.0
    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}', 'grad_norm': f'{avg_grad_norm:.3f}'})

    avg_loss = loss_sum / max(1, sample_count)
    avg_acc = acc_sum / max(1, sample_count)
    return avg_loss, avg_acc


#评估模型
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, split='Val'):
    """
    在给定数据集上评估模型性能（不更新参数）

    返回： (avg_loss, avg_acc, all_preds, all_labels)
    """
    model.eval()
    
    # 使用加权累计统计
    loss_sum = 0.0
    acc_sum = 0.0
    sample_count = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'[{split}]')
    
    for batch_idx, batch in enumerate(pbar):
        try:
            images, labels = batch
        except Exception as e:
            raise RuntimeError(f"Failed to read batch {batch_idx} from dataloader during eval: {e}") from e
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Fail-fast on NaN/Inf in eval
        if not torch.isfinite(loss):
            img_min = float(torch.min(images).cpu()) if images.numel() > 0 else float('nan')
            img_max = float(torch.max(images).cpu()) if images.numel() > 0 else float('nan')
            img_mean = float(torch.mean(images).cpu()) if images.numel() > 0 else float('nan')
            unique_labels = torch.unique(labels).cpu().tolist() if labels.numel() > 0 else []
            raise RuntimeError(
                f"Detected non-finite validation loss at split={split}, batch={batch_idx}. "
                f"loss={loss}. Image stats: min={img_min:.6e}, max={img_max:.6e}, mean={img_mean:.6e}. "
                f"labels_sample={unique_labels[:10]}"
            )
        
        # Compute accuracy 并累计
        pred = logits.argmax(dim=1)
        batch_acc = (pred == labels).float().mean().item()

        loss_sum += float(loss.item()) * batch_size
        acc_sum += float(batch_acc) * batch_size
        sample_count += batch_size
        
        # Store predictions
        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss = loss_sum / max(1, sample_count)
        avg_acc = acc_sum / max(1, sample_count)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})
    
    # 连接所有预测与标签
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.array([])
        all_labels = np.array([])
    
    avg_loss = loss_sum / max(1, sample_count)
    avg_acc = acc_sum / max(1, sample_count)
    return avg_loss, avg_acc, all_preds, all_labels



class CosineScheduleWithWarmup(LambdaLR):
    """
    基于 step 的余弦学习率调度器 （继承自 PyTorch 的 LambdaLR）

    功能：在前 num_warmup_steps 步使用线性预热（从 min_factor*base_lr 增加到 base_lr），
    之后使用余弦衰减；支持设置 min_factor 作为 lr 下限（相对于 base_lr 的比例）。  把学习率下限以“比例”形式（min_factor）固定为 base_lr 的一部分，避免衰减到 0

    参数说明：
      - optimizer: torch.optim.Optimizer      要控制学习率的 torch 优化器（如 AdamW）
      - num_warmup_steps: 预热步数（step 级别）    预热的步数
      - num_training_steps: 总训练步数（step 级别）   总训练步数
      - num_cycles: 余弦周期数（默认 0.5，对应半周期）   余弦周期数（控制余弦的一个或多周期行为）。常用 0.5 表示半周期（从 1 到 0 的单段衰减）
      - min_factor: 学习率下限，作为 base_lr 的比例（0.0 表示衰减到 0）
    """

    def __init__(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, min_factor: float = 0.0):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

        self.min_factor = float(min_factor)
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):  #返回一个乘子（scale factor），LambdaLR 会把它乘以每个参数组的 base_lr，得到当前 step 的有效学习率
        # 预热阶段：线性从 min_factor -> 1.0
        if current_step < self.num_warmup_steps:
            warm_ratio = float(current_step) / float(max(1, self.num_warmup_steps))
            return self.min_factor + (1.0 - self.min_factor) * warm_ratio
        # 主要余弦衰减阶段：根据训练进度计算余弦值，并缩放到 [min_factor, 1.0]
        denom = float(max(1, self.num_training_steps - self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / denom
        # 限制 progress 到 [0,1]，并在训练步数耗尽后固定返回 min_factor
        if progress >= 1.0:
            return float(self.min_factor)
        if progress <= 0.0:
            base = 1.0
        else:
            base = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * float(progress))))
        return self.min_factor + (1.0 - self.min_factor) * base




#创建优化器
def get_optimizer(model, lr, weight_decay, optimizer_type='adamw', llrd_decay: float = 0.65, head_lr_mult: float = 5.0, **kwargs):
    """
    为模型创建优化器，并应用层级学习率衰减（Layer-wise LR Decay, LLRD）与参数分组

    逻辑说明：
      - 对 CLIP 视觉 backbone 的 Transformer block 按深度应用学习率衰减：越靠近输出的层学习率越大（接近 base lr），越靠近输入的层越小（乘以 llrd_decay^k）
      - 分类头 (classifier) 使用更高的学习率倍率（head_lr_mult）
      - 对于偏置/归一化层/某些嵌入权重禁用权重衰减

    参数：与原实现一致。kwargs 将传递给对应优化器（例如 AdamW 的 betas, eps）
    返回：torch.optim.Optimizer
    """
    # 直接使用传入的 model（单卡场景）
    # NOTE: 这里不创建额外别名，后续直接引用 model
    lr = float(lr)

    # 识别视觉 Transformer 的 block 层编号（用于 LLRD）
    import re
    block_idx_pattern = re.compile(r"clip_model\.visual\.transformer\.resblocks\.(\d+)\.")
    max_block_idx = -1
    for name, p in model.named_parameters():
        m = block_idx_pattern.search(name)
        if m:
            idx = int(m.group(1))
            if idx > max_block_idx:
                max_block_idx = idx

    # 要求模型必须包含视觉 Transformer 的 resblocks，否则无法应用 LLRD
    if max_block_idx < 0:
        raise RuntimeError("get_optimizer: 未检测到视觉 Transformer 的 resblocks，无法应用 LLRD。请确认模型命名或使用支持的模型结构。")

    # 使用 LLRD：为每个 block 创建两组（decay / no_decay），并对特殊层分配合适深度
    num_blocks = max_block_idx + 1

    # 函数：根据参数名判断是否不使用 weight decay
    def use_no_decay(param_name: str) -> bool:
        lname = param_name.lower()
        if lname.endswith('bias'):
            return True
        if ('ln' in lname) or ('norm' in lname) or ('bn' in lname):
            return True
        if ('positional_embedding' in lname) or ('class_embedding' in lname):
            return True
        return False

    # 函数：为非 resblocks 的特殊层指定深度（用于 LLRD）
    # 约定：靠近输出的层（ln_post, proj）使用最高深度 = num_blocks（学习率最大）
    #      靠近输入的层（conv1, ln_pre, embeddings）使用最小深度 = 0（学习率最小）
    def special_depth(param_name: str) -> int | None:
        if 'clip_model.visual.transformer.resblocks' in param_name:
            return None  # 由正则匹配的 block 深度处理
        if 'clip_model.visual.ln_post' in param_name or 'clip_model.visual.proj' in param_name:
            return num_blocks  # 顶层
        if ('clip_model.visual.conv1' in param_name or
            'clip_model.visual.ln_pre' in param_name or
            'clip_model.visual.class_embedding' in param_name or
            'clip_model.visual.positional_embedding' in param_name):
            return 0  # 底层
        # 其他视觉层未明确归类的，给一个中间深度：1
        if 'clip_model.visual.' in param_name:
            return 1
        return None

    # 为每个深度收集参数（区分 decay/no_decay）
    layers_decay = {d: [] for d in range(0, num_blocks + 1)}
    layers_no_decay = {d: [] for d in range(0, num_blocks + 1)}
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('classifier'):
            (head_no_decay if use_no_decay(name) else head_decay).append(param)
            continue

        # 视觉 backbone 参数
        m = block_idx_pattern.search(name)
        if m:
            depth = int(m.group(1))  # 0..max
        else:
            sd = special_depth(name)
            if sd is None:
                # 非视觉/非分类头（比如文本编码器，保留默认学习率与权重衰减策略）
                sd = 1
            depth = int(sd)

        if use_no_decay(name):
            layers_no_decay[depth].append(param)
        else:
            layers_decay[depth].append(param)

    # 组装 param_groups：深度从低到高（靠近输入 -> 靠近输出），学习率随深度递增
    param_groups = []
    for depth in range(0, num_blocks + 1):
        # 计算该深度的学习率倍率：靠近输出（depth=num_blocks）应为 1.0，
        # 靠近输入（depth=0）应为 llrd_decay ** num_blocks
        layer_mult = llrd_decay ** (num_blocks - depth)
        if layers_decay[depth]:
            param_groups.append({
                "params": layers_decay[depth],
                "weight_decay": weight_decay,
                "lr": lr * layer_mult,
            })
        if layers_no_decay[depth]:
            param_groups.append({
                "params": layers_no_decay[depth],
                "weight_decay": 0.0,
                "lr": lr * layer_mult,
            })

    # 最后添加分类头（使用更大学习率倍率）
    if head_decay:
        param_groups.append({
            "params": head_decay,
            "weight_decay": weight_decay,
            "lr": lr * head_lr_mult,
        })
    if head_no_decay:
        param_groups.append({
            "params": head_no_decay,
            "weight_decay": 0.0,
            "lr": lr * head_lr_mult,
        })
    
    opt_name = optimizer_type.lower()

    # 通用解析函数
    def parse_betas(b):
        if b is None:
            return (0.9, 0.999)
        if isinstance(b, str):
            s = b.strip().lstrip('[').rstrip(']')
            parts = [float(x) for x in s.split(',') if x.strip()]
            if len(parts) == 2:
                return (parts[0], parts[1])
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return (float(b[0]), float(b[1]))
        return (0.9, 0.999)

    def parse_float(x, default):
        if x is None:
            return float(default)
        try:
            return float(x)
        except Exception:
            return float(default)

    weight_decay = parse_float(weight_decay, 0.0)

    if opt_name == 'adamw':
        betas = parse_betas(kwargs.get('betas', None))
        eps = parse_float(kwargs.get('eps', None), 1e-8)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif opt_name == 'sgd':
        momentum = parse_float(kwargs.get('momentum', None), 0.9)
        dampening = parse_float(kwargs.get('dampening', None), 0.0)
        nesterov = bool(kwargs.get('nesterov', False))
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_type}，可用值: 'adamw', 'sgd'")
    
    return optimizer


# 简单的 epoch-based 调度器封装，兼容 LP-FT 脚本
def get_scheduler(optimizer, scheduler_type: str = 'cosine', epochs: int = 100, eta_min: float = 0.0):
    """
    返回一个基于 epoch 的学习率调度器；若不需要则返回 None。

    支持：
      - 'cosine': torch.optim.lr_scheduler.CosineAnnealingLR
      - 'none' / None: 不使用调度器
    """
    if optimizer is None:
        return None
    st = str(scheduler_type or '').lower()
    if st in ('none', 'no', 'off', ''):
        return None
    if st in ('cosine', 'cos', 'cosineannealing', 'cosineannealinglr'):
        try:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=float(eta_min))
        except Exception:
            # 回退：若参数不合法，返回 None 以不中断训练
            return None
    # 未知类型：返回 None 或抛错，这里选择更宽松的 None
    return None


def save_results(results_obj, save_path: str) -> Optional[str]:
    """
    将结果字典保存为 JSON 文件；若保存成功返回路径，否则返回 None。
    """
    try:
        directory = os.path.dirname(save_path) or '.'
        os.makedirs(directory, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_obj, f, ensure_ascii=False, indent=4)
        return save_path
    except Exception:
        return None




 




