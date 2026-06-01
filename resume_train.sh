#!/bin/bash
# 超参数调试脚本：一次性测试同一参数的不同值，观察参数对性能的影响
#
# 用法:
#   ./resume_train.sh <dataset> <param_name> <value1,value2,...> [num_gpus] [num_workers]
#
# 示例:
#   ./resume_train.sh cub kl 0.1,0.5,1.0,2.0 2 8
#   ./resume_train.sh oxford_pets dec_t 0.03,0.05,0.07,0.1 2 8
#   ./resume_train.sh dtd lr 1e-5,5e-5,1e-4 2 8
#
# 支持的参数名（短名 → YAML路径）:
#   kl        → stn_config.kl_consistency_weight
#   dec       → stn_config.decorrelation_weight
#   fair      → stn_config.fairness_weight
#   lr        → training.learning_rate
#   dec_t     → stn_config.two_stage.dec_target_temp
#   stu_t     → stn_config.two_stage.dec_student_temp
#   warm_t    → stn_config.two_stage.warmup_student_temp
#   teacher_t → stn_config.two_stage.teacher_temp
#   ema_m     → stn_config.two_stage.ema_momentum
#   warm_ep   → stn_config.two_stage.warmup_epochs
#   也支持直接用 YAML 点路径，如 stn_config.kl_consistency_weight

# ============================================================================
# 参数解析
# ============================================================================
DATASET=${1:?请指定数据集名称}
PARAM_NAME=${2:?请指定要调试的参数名}
VALUES_STR=${3:?请指定参数值列表（逗号分隔）}
NUM_GPUS=${4:-2}
NUM_WORKERS=${5:-8}

# Conda 环境
CONDA_ENV="wca"
CONDA_PYTHON="/mnt/e3319bd7-a0cc-41a8-9825-36b781a06ce8/xzy/anaconda3/envs/wca/bin/python"

# 解析值列表
IFS=',' read -ra VALUES <<< "$VALUES_STR"

# 参数名 → YAML 点路径 映射
declare -A PARAM_MAP=(
    ["kl"]="stn_config.kl_consistency_weight"
    ["dec"]="stn_config.decorrelation_weight"
    ["fair"]="stn_config.fairness_weight"
    ["lr"]="training.learning_rate"
    ["dec_t"]="stn_config.two_stage.dec_target_temp"
    ["stu_t"]="stn_config.two_stage.dec_student_temp"
    ["warm_t"]="stn_config.two_stage.warmup_student_temp"
    ["teacher_t"]="stn_config.two_stage.teacher_temp"
    ["ema_m"]="stn_config.two_stage.ema_momentum"
    ["warm_ep"]="stn_config.two_stage.warmup_epochs"
)

# 解析 YAML 路径
if [[ "$PARAM_NAME" == *"."* ]]; then
    YAML_PATH="$PARAM_NAME"
else
    YAML_PATH="${PARAM_MAP[$PARAM_NAME]:-}"
    if [ -z "$YAML_PATH" ]; then
        echo "❌ 未知参数名: $PARAM_NAME"
        echo "   支持的短名: ${!PARAM_MAP[*]}"
        echo "   也支持直接使用 YAML 点路径"
        exit 1
    fi
fi

# 配置文件路径（默认无监督）
BASE_CONFIG="UN-STN-Config/${DATASET}.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ 配置文件不存在: $BASE_CONFIG"
    exit 1
fi

# 日志目录
LOG_DIR="logs/sweeps"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 参数显示名
PARAM_SHORT=$(echo "$YAML_PATH" | sed 's/.*\.//')
TOTAL_RUNS=${#VALUES[@]}

echo "=============================================="
echo "  超参数调试（无监督训练）"
echo "=============================================="
echo "数据集:      $DATASET"
echo "配置文件:    $BASE_CONFIG"
echo "调试参数:    $YAML_PATH ($PARAM_NAME)"
echo "参数值:      ${VALUES[*]}"
echo "运行次数:    $TOTAL_RUNS"
echo "GPU数量:     $NUM_GPUS"
echo "Workers:     $NUM_WORKERS"
echo "时间戳:      $TIMESTAMP"
echo "=============================================="
echo ""

# 退出时清理临时配置文件
cleanup() { rm -f /tmp/sweep_*_${TIMESTAMP}.yaml; }
trap cleanup EXIT

# 汇总文件
SUMMARY_FILE="${LOG_DIR}/summary_${DATASET}_${PARAM_SHORT}_${TIMESTAMP}.txt"

# ============================================================================
# 主循环
# ============================================================================
RUN_IDX=0
declare -a RESULTS

for VAL in "${VALUES[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))

    VAL=$(echo "$VAL" | xargs)
    VAL_SAFE=$(echo "$VAL" | sed 's/\./-/g' | sed 's/e/E/g')

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$RUN_IDX/$TOTAL_RUNS] $PARAM_SHORT = $VAL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 生成临时配置文件
    TEMP_CONFIG="/tmp/sweep_${DATASET}_${PARAM_SHORT}_${VAL_SAFE}_${TIMESTAMP}.yaml"
    $CONDA_PYTHON -c "
import yaml
with open('$BASE_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

path = '$YAML_PATH'.split('.')
val_str = '$VAL'

try:
    if '.' in val_str or 'e' in val_str.lower():
        val = float(val_str)
    else:
        val = int(val_str)
except ValueError:
    val = val_str

d = config
for key in path[:-1]:
    if key not in d:
        d[key] = {}
    d = d[key]
d[path[-1]] = val

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f'  → $YAML_PATH = {val} ({type(val).__name__})')
"

    LOG_FILE="${LOG_DIR}/sweep_${DATASET}_${PARAM_SHORT}_${VAL_SAFE}_${TIMESTAMP}.log"

    echo "  → 日志: $LOG_FILE"
    echo ""

    # 运行训练（直接用 conda python，避免 conda run 缓冲问题）
    echo "🚀 开始训练..."
    $CONDA_PYTHON -u -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        train_unsupervised_ddp.py \
        --dataset $DATASET \
        --config $TEMP_CONFIG \
        --num_workers $NUM_WORKERS \
        --seed 42 \
        > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    # 提取关键结果
    BEST_LINE=$(grep "新最佳Loss" "$LOG_FILE" | tail -1 | sed 's/.*新最佳Loss: //' | sed 's/,/ /g')
    LAST_EPOCH=$(grep "^Epoch " "$LOG_FILE" | tail -1 | sed 's/  / /g')
    EARLY_STOP=$(grep "早停触发" "$LOG_FILE" | wc -l)

    echo ""
    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️  训练退出码: $EXIT_CODE"
        RESULT="$VAL | EXIT=$EXIT_CODE"
    else
        echo "✅ 训练完成"
        RESULT="$VAL | $LAST_EPOCH"
    fi
    if [ -n "$BEST_LINE" ]; then
        echo "  📊 最佳: $BEST_LINE"
        RESULT="$RESULT | Best: $BEST_LINE"
    fi
    if [ "$EARLY_STOP" -gt 0 ]; then
        echo "  ⏳ 触发早停"
        RESULT="$RESULT | 早停"
    fi
    echo ""

    RESULTS+=("$RESULT")

    # 清理临时配置
    rm -f "$TEMP_CONFIG"
done

# ============================================================================
# 汇总
# ============================================================================
echo "=============================================="
echo "  全部实验完成"
echo "=============================================="
echo ""
echo "📊 结果汇总:"
echo "  $YAML_PATH"
echo ""

for r in "${RESULTS[@]}"; do
    echo "  $r"
done

# 写入汇总文件
{
    echo "超参数调试汇总"
    echo "数据集: $DATASET"
    echo "参数: $YAML_PATH"
    echo "时间: $TIMESTAMP"
    echo "=========================================="
    for r in "${RESULTS[@]}"; do
        echo "$r"
    done
    echo ""
    echo "详细日志: $LOG_DIR/sweep_${DATASET}_${PARAM_SHORT}_*_${TIMESTAMP}.log"
} > "$SUMMARY_FILE"

echo ""
echo "📁 汇总文件: $SUMMARY_FILE"
echo "📁 日志目录: $LOG_DIR/"
echo "📁 检查点目录: checkpoints/unsupervised/$DATASET/"
