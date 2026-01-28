#!/bin/bash
# 停止训练脚本
# 安全地停止后台训练进程

echo "=========================================="
echo "  停止训练工具"
echo "=========================================="
echo ""

# 查找训练进程
echo "🔍 查找训练进程..."
TRAIN_PIDS=$(ps aux | grep -E "train_ddp_stn.py|main_stn.py|torchrun.*train" | grep -v grep | awk '{print $2}')

if [ -z "$TRAIN_PIDS" ]; then
    echo "   ❌ 没有运行中的训练进程"
    exit 0
fi

echo "   找到以下训练进程:"
ps aux | grep -E "train_ddp_stn.py|main_stn.py|torchrun.*train" | grep -v grep | \
    awk '{printf "   PID: %s, CPU: %s%%, MEM: %s%%, CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'

echo ""
read -p "❓ 确认停止这些进程? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 正在停止训练进程..."
    
    # 首先尝试优雅地停止（SIGTERM）
    for pid in $TRAIN_PIDS; do
        echo "   发送SIGTERM到进程 $pid..."
        kill -15 $pid 2>/dev/null || true
    done
    
    # 等待5秒
    echo "   等待进程退出..."
    sleep 5
    
    # 检查是否还有进程存活
    REMAINING=$(ps aux | grep -E "train_ddp_stn.py|main_stn.py|torchrun.*train" | grep -v grep | awk '{print $2}')
    
    if [ -n "$REMAINING" ]; then
        echo "   ⚠️  部分进程未响应，强制终止..."
        for pid in $REMAINING; do
            echo "   发送SIGKILL到进程 $pid..."
            kill -9 $pid 2>/dev/null || true
        done
        sleep 2
    fi
    
    # 最终检查
    FINAL_CHECK=$(ps aux | grep -E "train_ddp_stn.py|main_stn.py|torchrun.*train" | grep -v grep)
    
    if [ -z "$FINAL_CHECK" ]; then
        echo "   ✅ 所有训练进程已停止"
        
        # 显示最新检查点
        echo ""
        echo "💾 最新检查点已保存，可以使用以下命令恢复训练:"
        echo "   ./resume_train.sh imagenet 2 4 yes"
    else
        echo "   ❌ 部分进程仍在运行:"
        echo "$FINAL_CHECK"
    fi
else
    echo "   ❌ 已取消"
fi

echo ""
