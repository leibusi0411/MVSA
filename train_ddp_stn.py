"""
分布式多卡联合训练脚本（DDP）

- 保留现有 main_stn.py 与 train_multi_view_stn.py 不变
- 新增本脚本，统一在多卡上训练多视角 STN-CLIP
- 仅在 rank0 打印与保存检查点

运行方式（示例）：

  torchrun --nproc_per_node=2 train_ddp_stn.py \
    --dataset cub \
    --stn_config cub \
    --num_workers 8 \
    --seed 42

说明：
- 默认将配置从 STN-Config/{dataset}.yaml 读取（与 main_stn 保持一致）
- 使用 per-GPU 的 batch_size（即配置文件里的 batch_size 即为每张卡的批大小）
- 使用 DistributedSampler 切分数据；训练/验证都进行全局度量（all-reduce）
"""

import os
import io
import yaml
import time
import argparse
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    # 简易回退：没有tqdm时提供一个空实现，避免导入错误
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
from data_preprocess import MultiViewDataset
from train_multi_view_stn import (
    setup_multi_view_training,
    compute_and_save_text_features,
    MultiViewSTNLoss,
)
from main_stn import set_seed, get_stn_config_path


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


@contextlib.contextmanager
def suppress_stdout_if_not_main():
    """在非主进程静默第三方函数的打印。"""
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
    """根据 torchrun 环境变量初始化分布式。"""
    if is_dist_avail_and_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 友好检查：本机可见GPU数量必须大于local_rank
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count == 0:
                raise RuntimeError("检测到CUDA可用，但设备数为0。请检查驱动/环境，或改用 --nproc_per_node=1 单卡运行。")
            if local_rank < 0 or local_rank >= device_count:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} 超出可见CUDA设备数量 {device_count}。"
                    f" 请将 --nproc_per_node 设置为 {device_count}（或更小），"
                    f"或通过 CUDA_VISIBLE_DEVICES 指定足够的设备。"
                )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        return local_rank
    else:
        # 非分布式：返回本地设备索引0，不初始化进程组
        return int(os.environ.get("LOCAL_RANK", 0))


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
                     persistent_workers: bool | None = None) -> tuple[DataLoader, DistributedSampler]:
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

    # persistent_workers: 训练集可开启以提速；验证集建议关闭，避免退出卡住
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


def ddp_train_one_epoch(model: DDP,
                        optimizer: torch.optim.Optimizer,
                        criterion: MultiViewSTNLoss,
                        train_loader: DataLoader,
                        device: torch.device,
                        epoch: int,
                        config: dict,
                        text_features: torch.Tensor) -> tuple[float, float]:
    model.train()
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    temperature = float(config['stn_config'].get('logits_temp', 0.07))
    need_intermediate = (
        getattr(criterion, 'decorrelation_weight', 0.0) > 0 or
        getattr(criterion, 'adaptive_weight', 0.0) > 0
    )

    # 梯度裁剪配置
    max_grad_norm = float(config['training'].get('max_grad_norm', 1.0))
    grad_norm_type = float(config['training'].get('grad_norm_type', 2.0))

    model_without_ddp = model.module if hasattr(model, 'module') else model

    # 仅在主进程显示进度条
    iterable = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False) if is_main_process() else train_loader
    for images, labels in iterable:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        # 训练阶段统一使用 train 模式
        fused_features, view_features = model(images, mode='train')
        
        # 如果不需要特殊损失，将 view_features 设为 None
        if not need_intermediate:
            view_features = None

        similarity = (fused_features @ text_features) / temperature  # [B,C]

        loss, _ = criterion(
            labels=labels,
            similarity_or_logits=similarity,
            view_features=view_features,
            text_features=text_features,
        )

        loss.backward()

        # 可选优化：将梯度变为连续内存，减少DDP梯度布局不匹配带来的额外拷贝
        for p in model_without_ddp.parameters():
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()

        # 按原脚本：仅裁剪 STN 可训练参数
        clip_params = []
        if hasattr(model_without_ddp, 'localization_network'):
            clip_params.extend([p for p in model_without_ddp.localization_network.parameters() if p.requires_grad])
        if hasattr(model_without_ddp, 'fusion_module') and model_without_ddp.fusion_module is not None:
            clip_params.extend([p for p in model_without_ddp.fusion_module.parameters() if p.requires_grad])
        if clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_grad_norm, norm_type=grad_norm_type)

        optimizer.step()

        with torch.no_grad():
            bs = labels.size(0)
            total_loss_sum += loss.item() * bs
            preds = similarity.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += bs

        if is_main_process() and hasattr(iterable, 'set_postfix'):
            current_lr = optimizer.param_groups[0]['lr']
            iterable.set_postfix({
                'loss': f"{loss.item():.3f}",
                'lr': f"{current_lr:.1e}"
            })

    # 全局归并
    if is_dist_avail_and_initialized():
        t = torch.tensor([total_loss_sum, total_correct, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_sum, total_correct, total_samples = t.tolist()

    # 关闭进度条，避免句柄占用
    if is_main_process() and hasattr(iterable, 'close'):
        try:
            iterable.close()
        except Exception:
            pass

    avg_loss = total_loss_sum / max(total_samples, 1.0)
    avg_acc = total_correct / max(total_samples, 1.0)
    return float(avg_loss), float(avg_acc)


def ddp_validate(model: DDP,
                 criterion: MultiViewSTNLoss,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: dict,
                 text_features: torch.Tensor) -> tuple[float, float]:
    model.eval()

    temperature = float(config['stn_config'].get('logits_temp', 0.07))
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        # 仅在主进程显示进度条
        iterable = tqdm(val_loader, desc="Val", leave=False) if is_main_process() else val_loader
        for images, labels in iterable:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            fused_features, view_features = model(images, mode='train')

            similarity = (fused_features @ text_features) / temperature
            loss, _ = criterion(
                labels=labels,
                similarity_or_logits=similarity,
                view_features=view_features,
                text_features=text_features,
            )

            bs = labels.size(0)
            total_loss_sum += loss.item() * bs
            preds = similarity.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += bs

            if is_main_process() and hasattr(iterable, 'set_postfix'):
                iterable.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'acc': f"{(preds == labels).float().mean().item():.3f}"
                })

        # 关闭进度条
        if is_main_process() and hasattr(iterable, 'close'):
            try:
                iterable.close()
            except Exception:
                pass

    if is_dist_avail_and_initialized():
        t = torch.tensor([total_loss_sum, total_correct, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_sum, total_correct, total_samples = t.tolist()

    avg_loss = total_loss_sum / max(total_samples, 1.0)
    avg_acc = total_correct / max(total_samples, 1.0)
    return float(avg_loss), float(avg_acc)


def main():
    parser = argparse.ArgumentParser(description="DDP 多视角 STN-CLIP 训练")
    parser.add_argument('--dataset', type=str, default='imagenet', help='数据集名称，如 cub, food101, ...')
    parser.add_argument('--stn_config', type=str, default=None, help='STN-Config 下的配置名（不含路径/后缀）或完整路径')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    local_rank = init_distributed()
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    if is_main_process():
        print(f"[DDP] World Size: {dist.get_world_size() if is_dist_avail_and_initialized() else 1}")
        print(f"[DDP] Device: {device}")
        print(f"[DDP] Dataset: {args.dataset}")

    set_seed(args.seed + get_rank())

    # 读取配置
    config_path = get_stn_config_path(args.dataset, args.stn_config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['dataset'] = args.dataset

    # 加载 CLIP 并构建 STN 模型
    model_size = config['model_size']
    if is_main_process():
        print(f"Loading CLIP model: {model_size}")
    clip_model, _ = clip.load(model_size, device=device)
    clip_model = clip_model.float()

    num_views = config['stn_config'].get('num_views', 4)
    stn_model = MultiViewSTNModel(clip_model, config['stn_config'], num_views=num_views).to(device)
    stn_model = stn_model.float()

    # 优化器/损失/调度器（在各rank均初始化；非主进程静默打印）
    with suppress_stdout_if_not_main():
        criterion, optimizer, scheduler = setup_multi_view_training(stn_model, config, device)

    # 构建 DDP 包装（仅在分布式环境下）
    if is_dist_avail_and_initialized():
        stn_model = DDP(
            stn_model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # 使用bucket视图作为梯度，优化内存布局
        )

    # DataLoader + DistributedSampler
    bs = int(config['training']['batch_size'])  # 按每卡批大小理解
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
        persistent_workers=None,  # 训练集按 num_workers 自动开启
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
        persistent_workers=False,  # 验证集禁用持久workers，避免退出卡住
    )

    # 文本特征：仅主进程计算与保存，其他进程等待并加载
    model_name = model_size.replace('/', '_')
    text_features_path = f"text_features/{args.dataset}_{model_name}.pt"
    os.makedirs("text_features", exist_ok=True)

    if is_main_process():
        try:
            tf = torch.load(text_features_path, map_location='cpu')
            if is_main_process():
                print(f"✅ 已找到预计算文本特征: {text_features_path}")
        except FileNotFoundError:
            if is_main_process():
                print(f"⚠️ 未找到文本特征，开始计算并保存: {text_features_path}")
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

    # 各rank加载到设备并设为FP32
    all_classes_text_features = torch.load(text_features_path, map_location=device).float().to(device)
    all_classes_text_features.requires_grad_(False)

    total_epochs = int(config['training']['epochs'])

    # 保存名与原脚本一致
    stn_cfg = config.get('stn_config', {})
    model_name_suffix = f"views{stn_cfg.get('num_views', 4)}_{stn_cfg.get('fusion_mode', 'simple')}_dim{stn_cfg.get('hidden_dim', 512)}"
    loss_suffix = f"temp{stn_cfg.get('logits_temp', 0.07)}"
    if stn_cfg.get('classification_weight', 1.0) > 0:
        loss_suffix += f"_cls{stn_cfg.get('classification_weight', 1.0)}"
    if stn_cfg.get('contrastive_weight', 0.0) > 0:
        loss_suffix += f"_con{stn_cfg.get('contrastive_weight', 0.0)}"
    if stn_cfg.get('decorrelation_weight', 0.0) > 0:
        loss_suffix += f"_dec{stn_cfg.get('decorrelation_weight', 0.0)}"
    if stn_cfg.get('adaptive_weight', 0.0) > 0:
        loss_suffix += f"_adp{stn_cfg.get('adaptive_weight', 0.0)}"

    base_model_name = f"multi_view_stn_{args.dataset}_{model_size.replace('/', '_')}_{model_name_suffix}_{loss_suffix}"
    # 将检查点按数据集分类保存到二级目录，例如 checkpoints/cub/
    ckpt_dir = os.path.join('checkpoints', args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)
    # 跟踪两类最佳：最小验证损失与最高验证准确率
    best_val_loss = float('inf')
    best_loss_epoch = 0
    best_loss_acc = 0.0

    best_val_acc = 0.0
    best_acc_epoch = 0

    ckpt_best_loss = os.path.join(ckpt_dir, f"{base_model_name}_best_loss.pth")
    ckpt_best_acc = os.path.join(ckpt_dir, f"{base_model_name}_best_acc.pth")
    ckpt_latest = os.path.join(ckpt_dir, f"{base_model_name}_latest.pth")  # 最新检查点
    patience = int(config['training'].get('patience', 5))
    patience_counter = 0
    best_epoch = 0
    start_epoch = 0  # 起始epoch

    # === 断点续训：检查是否存在最新检查点 ===
    if os.path.exists(ckpt_latest):
        if is_main_process():
            print(f"\n🔄 发现检查点，尝试恢复训练: {ckpt_latest}")
        
        try:
            checkpoint = torch.load(ckpt_latest, map_location=device)
            
            # 恢复模型权重
            if hasattr(stn_model, 'module'):
                stn_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
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
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_acc_epoch = checkpoint.get('best_acc_epoch', 0)
            patience_counter = checkpoint.get('patience_counter', 0)
            best_epoch = checkpoint.get('best_epoch', 0)
            
            if is_main_process():
                print(f"✅ 成功恢复训练状态:")
                print(f"   - 起始Epoch: {start_epoch}/{total_epochs}")
                print(f"   - 最佳Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch}, Acc={best_loss_acc:.3f})")
                print(f"   - 最佳Acc: {best_val_acc:.3f} (Epoch {best_acc_epoch})")
                print(f"   - 早停计数: {patience_counter}/{patience}")
        
        except Exception as e:
            if is_main_process():
                print(f"⚠️  恢复检查点失败: {e}")
                print(f"   从头开始训练...")
            start_epoch = 0
    else:
        if is_main_process():
            print(f"\n🆕 未找到检查点，从头开始训练")

    if is_main_process():
        print(f"开始分布式训练：总轮数={total_epochs}，耐心值={patience}，起始Epoch={start_epoch}")

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = ddp_train_one_epoch(
            model=stn_model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            config=config,
            text_features=all_classes_text_features,
        )

        val_loss, val_acc = ddp_validate(
            model=stn_model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            config=config,
            text_features=all_classes_text_features,
        )

        if scheduler is not None:
            # 与单卡一致：每epoch调用一次
            if hasattr(scheduler, 'step'):
                scheduler.step()

        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{total_epochs} | Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                  f"Val: loss={val_loss:.4f}, acc={val_acc:.3f} | lr={current_lr:.2e}")

            # 先判定是否为最佳loss
            loss_improved = val_loss < best_val_loss
            if loss_improved:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_loss_acc = val_acc
                best_loss_epoch = epoch + 1
                best_epoch = best_loss_epoch
                patience_counter = 0

                to_save_state = stn_model.module.state_dict() if hasattr(stn_model, 'module') else stn_model.state_dict()
                torch.save(to_save_state, ckpt_best_loss)
                print(f"  🎯 验证损失改善 -{improvement:.6f}! 新最佳Loss: {best_val_loss:.6f} (第{best_loss_epoch}轮), 当次Acc={best_loss_acc:.3f}")
                print(f"  💾 已保存最佳损失模型: {ckpt_best_loss}")
            else:
                patience_counter += 1
                print(f"  ⏳ 验证损失未改善 {patience_counter}/{patience} 轮 (当前: {val_loss:.6f}, 最佳: {best_val_loss:.6f})")

            # 再判定是否为最佳acc（独立于loss）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc_epoch = epoch + 1
                to_save_state = stn_model.module.state_dict() if hasattr(stn_model, 'module') else stn_model.state_dict()
                torch.save(to_save_state, ckpt_best_acc)
                print(f"  🏅 验证准确率提升! 新最佳Acc: {best_val_acc:.3f} (第{best_acc_epoch}轮)")
                print(f"  💾 已保存最佳准确率模型: {ckpt_best_acc}")
            
            # === 每个epoch保存最新检查点（用于断点续训）===
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': stn_model.module.state_dict() if hasattr(stn_model, 'module') else stn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
                'best_loss_acc': best_loss_acc,
                'best_loss_epoch': best_loss_epoch,
                'best_val_acc': best_val_acc,
                'best_acc_epoch': best_acc_epoch,
                'patience_counter': patience_counter,
                'best_epoch': best_epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            torch.save(checkpoint, ckpt_latest)
            print(f"  💾 已保存最新检查点: {ckpt_latest} (Epoch {epoch+1})")

            if patience_counter >= patience:
                print("🛑 早停触发，结束训练以避免过拟合。")
                break

        # 同步各 rank 的早停决定
        if is_dist_avail_and_initialized():
            flag = torch.tensor(1 if (is_main_process() and patience_counter >= patience) else 0, device=device)
            dist.broadcast(flag, src=0)
            if flag.item() == 1:
                break

        # 每 epoch 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 训练完成汇总
    if is_main_process():
        print("\n🎉 分布式多视角 STN-CLIP 训练完成")
        print(f"🏆 最佳Loss: Epoch {best_loss_epoch}, Val Loss={best_val_loss:.6f}, Acc@bestLoss={best_loss_acc:.3f}")
        print(f"🏅 最佳Acc:  Epoch {best_acc_epoch}, Val Acc={best_val_acc:.3f}")
        print(f"💾 已保存: \n   - 最佳损失: {ckpt_best_loss}\n   - 最佳准确: {ckpt_best_acc}")

    # 结束前显式同步/释放，降低退出卡住概率
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    try:
        # 显式释放 DataLoader 引用
        del train_loader
        del val_loader
    except Exception:
        pass

    # 结束分布式
    if is_dist_avail_and_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # 采用常规清理退出，不做硬退出


if __name__ == '__main__':
    main()
