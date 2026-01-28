#!/bin/bash
# 断点续训启动脚本（支持nohup后台运行）
# 用法: ./resume_train.sh [dataset] [num_gpus] [num_workers] [use_nohup]
# 示例: 
#   ./resume_train.sh imagenet 2 4 yes    # 后台运行
#   ./resume_train.sh imagenet 2 4 no     # 前台运行（默认）

set -e  # 遇到错误立即退出

# 默认参数
DATASET=${1:-imagenet}
NUM_GPUS=${2:-2}
NUM_WORKERS=${3:-4}
USE_NOHUP=${4:-yes}  # 默认使用nohup

echo "=========================================="
echo "  断点续训启动脚本"
echo "=========================================="
echo "数据集: $DATASET"
echo "GPU数量: $NUM_GPUS"
echo "DataLoader Workers: $NUM_WORKERS"
echo "后台运行: $USE_NOHUP"
echo ""

# 设置日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${DATASET}_${TIMESTAMP}.log"
echo "📝 日志文件: $LOG_FILE"
echo ""

# 检查配置文件
CONFIG_FILE="STN-Config/${DATASET}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "✅ 配置文件: $CONFIG_FILE"

# 检查是否存在检查点
CKPT_DIR="checkpoints/${DATASET}"
LATEST_CKPT=$(find "$CKPT_DIR" -name "*_latest.pth" 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "🔄 发现检查点: $LATEST_CKPT"
    echo "   将从上次中断处继续训练"
    
    # 显示检查点信息
    python -c "
import torch
import os
ckpt = torch.load('$LATEST_CKPT', map_location='cpu')
print(f'   - Epoch: {ckpt[\"epoch\"]+1}')
print(f'   - 最佳Loss: {ckpt[\"best_val_loss\"]:.6f}')
print(f'   - 最佳Acc: {ckpt.get(\"best_val_accuracy\", ckpt.get(\"best_val_acc\", 0)):.3f}')
" 2>/dev/null || echo "   (无法读取检查点详情)"
else
    echo "🆕 未找到检查点，将从头开始训练"
fi

echo ""
echo "=========================================="
echo "  开始训练"
echo "=========================================="
echo ""

# 清理GPU缓存
if command -v nvidia-smi &> /dev/null; then
    echo "🧹 清理GPU缓存..."
    nvidia-smi --gpu-reset 2>/dev/null || true
fi

# 构建训练命令
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS train_ddp_stn.py --dataset $DATASET --stn_config $DATASET --num_workers $NUM_WORKERS --seed 42"
else
    TRAIN_CMD="python main_stn.py --dataset_name $DATASET --stn_config $DATASET --num_workers $NUM_WORKERS --seed 42 --device cuda"
fi

# 启动训练
if [ "$USE_NOHUP" = "yes" ] || [ "$USE_NOHUP" = "y" ]; then
    echo "🚀 启动后台训练（nohup模式）..."
    echo "   命令: $TRAIN_CMD"
    echo "   日志: $LOG_FILE"
    echo ""
    echo "💡 提示:"
    echo "   - 查看实时日志: tail -f $LOG_FILE"
    echo "   - 查看训练进程: ps aux | grep train"
    echo "   - 停止训练: pkill -f 'train_ddp_stn.py|main_stn.py'"
    echo ""
    
    # 使用nohup在后台运行，输出重定向到日志文件
    nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
    
    # 获取进程ID
    TRAIN_PID=$!
    echo "✅ 训练已在后台启动"
    echo "   进程ID: $TRAIN_PID"
    echo "   日志文件: $LOG_FILE"
    echo ""
    echo "🔍 等待3秒检查进程状态..."
    sleep 3
    
    if ps -p $TRAIN_PID > /dev/null; then
        echo "✅ 训练进程运行正常"
        echo ""
        echo "📊 最新日志（前20行）:"
        echo "----------------------------------------"
        head -20 "$LOG_FILE" 2>/dev/null || echo "日志文件尚未生成"
        echo "----------------------------------------"
        echo ""
        echo "💡 使用以下命令查看实时日志:"
        echo "   tail -f $LOG_FILE"
    else
        echo "❌ 训练进程启动失败，请检查日志:"
        echo "   cat $LOG_FILE"
        exit 1
    fi
else
    echo "🚀 启动前台训练..."
    echo "   命令: $TRAIN_CMD"
    echo ""
    
    # 前台运行，同时输出到终端和日志文件
    $TRAIN_CMD 2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "=========================================="
    echo "  训练完成或中断"
    echo "=========================================="
fi

# 显示最终检查点
if [ -d "$CKPT_DIR" ]; then
    echo ""
    echo "📁 保存的检查点:"
    ls -lh "$CKPT_DIR"/*.pth 2>/dev/null | awk '{printf "   %s (%s)\n", $9, $5}' || echo "   无检查点"
fi
