#!/bin/bash
# 参数扫描脚本：多数据集 × 多参数 → 训练 → 测试 → 记录
#
# 用法1 — 直接编辑本文件中的 EXPERIMENTS 数组（推荐）
#   修改下面的 EXPERIMENTS 数组，然后直接运行: ./run_sweep.sh
#
# 用法2 — 命令行参数
#   ./run_sweep.sh --datasets oxford_pets --param warmup_epochs --values 8,12
#   ./run_sweep.sh --datasets oxford_pets --params warmup_epochs:8,12 teacher_temp:0.05,0.07
#
# 短名 → YAML路径:
#   warmup_epochs → stn_config.two_stage.warmup_epochs
#   teacher_temp  → stn_config.two_stage.teacher_temp
#   student_temp  → stn_config.two_stage.warmup_student_temp
#   dec_temp      → stn_config.two_stage.dec_student_temp
#   lr            → training.learning_rate
#   也支持直接用 YAML 点路径

set -e

# ============================================================================
# Conda 环境
# ============================================================================
CONDA_ENV="wca"
CONDA_PYTHON="/mnt/e3319bd7-a0cc-41a8-9825-36b781a06ce8/xzy/anaconda3/envs/wca/bin/python"

# ============================================================================
# 实验计划（直接编辑生效，命令行参数会覆盖）
# 格式: "dataset param_name values"
# ============================================================================
EXPERIMENTS=(
    # ============================================================
    # clip_guidance_weight（阶段二 CLIP 约束权重）
    # ============================================================
    "oxford_pets clip_guidance_weight 0.5,1.0,1.5"
    "cub         clip_guidance_weight 0.5,1.0,1.5"
    "dtd         clip_guidance_weight 0.5,1.0,1.5"
)

# ============================================================================
# 参数映射
# ============================================================================
declare -A PARAM_MAP=(
    ["warmup_epochs"]="stn_config.two_stage.warmup_epochs"
    ["teacher_temp"]="stn_config.two_stage.teacher_temp"
    ["student_temp"]="stn_config.two_stage.warmup_student_temp"
    ["dec_temp"]="stn_config.two_stage.dec_student_temp"
    ["decorr"]="stn_config.decorrelation_weight"
    ["fairness"]="stn_config.fairness_weight"
    ["lr"]="training.learning_rate"
    ["batch_size"]="training.batch_size"
    ["clip_guidance_weight"]="stn_config.two_stage.clip_guidance_weight"
)

# ============================================================================
# 构建实验计划（如果 EXPERIMENTS 数组为空，从命令行参数构建）
# ============================================================================
DATASETS=()

if [[ ${#EXPERIMENTS[@]} -gt 0 ]]; then
    # EXPERIMENTS 数组中已有数据，跳过 CLI 解析
    :
else
    # 从 CLI 构建 EXPERIMENTS 数组
    while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
        --dataset)  DATASETS=("$2"); shift 2 ;;
        --param)
            K="$2"
            K="${PARAM_MAP[$K]:-$K}"
            V="$3"
            for ds in "${DATASETS[@]}"; do
                EXPERIMENTS+=("$ds $K $V")
            done
            shift 3
            ;;
        --params)
            shift
            PARAM_KEYS=()
            PARAM_VALS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                K="${1%%:*}"
                V="${1#*:}"
                K="${PARAM_MAP[$K]:-$K}"
                PARAM_KEYS+=("$K")
                PARAM_VALS+=("$V")
                shift
            done
            # 用 Python 做笛卡尔积
            COMBOS=$($CONDA_PYTHON -c "
import itertools
keys = '${PARAM_KEYS[*]}'.split()
val_lists = ['${PARAM_VALS[*]}'.split()[i] for i in range(${#PARAM_VALS[@]})]
for combo in itertools.product(*[v.split(',') for v in val_lists]):
    line = ' '.join(f'{k}:{v}' for k, v in zip(keys, combo))
    print(line)
")
            for ds in "${DATASETS[@]}"; do
                while IFS= read -r combo; do
                    EXPERIMENTS+=("$ds GRID $combo")
                done <<< "$COMBOS"
            done
            ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

fi  # end of CLI arg parsing block

GPUS=${GPUS:-2}
WORKERS=${WORKERS:-8}

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "用法:"
    echo "  ./run_sweep.sh --dataset oxford_pets --param warmup_epochs --values 8,12"
    echo "  ./run_sweep.sh --dataset oxford_pets --params warmup_epochs:8,12 teacher_temp:0.05,0.07"
    exit 1
fi

# ============================================================================
# 初始化
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/sweep/${TIMESTAMP}"
RESULTS_FILE="${LOG_DIR}/results.txt"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  参数扫描"
echo "=============================================="
echo "实验数:   ${#EXPERIMENTS[@]}"
echo "GPU:      $GPUS"
echo "Workers:  $WORKERS"
echo "日志目录: $LOG_DIR"
echo "=============================================="
echo ""

TOTAL_EXPS=0
COMPLETED=0
FAILED=0
declare -a ALL_RESULTS

for EXP in "${EXPERIMENTS[@]}"; do
    # 跳过空行和注释
    [[ -z "$EXP" || "$EXP" == \#* ]] && continue

    read -r DATASET MODE REST <<< "$EXP"

    BASE_CONFIG="UN-STN-Config/${DATASET}.yaml"
    if [ ! -f "$BASE_CONFIG" ]; then
        echo "❌ 配置文件不存在: $BASE_CONFIG"
        continue
    fi

    if [[ "$MODE" == "GRID" ]]; then
        # 多参数模式: REST = "path1:val1 path2:val2 ..."
        EXP_NAME="${DATASET}"

        # 生成临时配置
        TOTAL_EXPS=$((TOTAL_EXPS + 1))
        EXP_LOG="${LOG_DIR}/${EXP_NAME}_grid${TOTAL_EXPS}.log"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$TOTAL_EXPS] $DATASET | 多参数组合"
        for pair in $REST; do
            K="${pair%%:*}"
            V="${pair#*:}"
            echo "    $K = $V"
        done
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        TEMP_CONFIG="/tmp/sweep_${EXP_NAME}_grid${TOTAL_EXPS}_${TIMESTAMP}.yaml"

        # 构建 Python overrides 字典
        OVERRIDES="{"
        for pair in $REST; do
            K="${pair%%:*}"
            V="${pair#*:}"
            OVERRIDES+="'$K': '$V', "
        done
        OVERRIDES+="}"

        $CONDA_PYTHON -c "
import yaml
with open('$BASE_CONFIG', 'r') as f:
    c = yaml.safe_load(f)

overrides = $OVERRIDES
for key_path, val_str in overrides.items():
    parts = key_path.split('.')
    try:
        if '.' in val_str or 'e' in val_str.lower():
            val = float(val_str)
        else:
            val = int(val_str)
    except ValueError:
        val = val_str
    d = c
    for p in parts[:-1]:
        if p not in d: d[p] = {}
        d = d[p]
    d[parts[-1]] = val

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'  Config written')
" 2>&1 | tee -a "$EXP_LOG"

        echo "  [训练] 开始..."
        PYTHONUNBUFFERED=1 $CONDA_PYTHON -u -m torch.distributed.run \
            --nproc_per_node=$GPUS \
            train_unsupervised_ddp.py \
            --dataset $DATASET \
            --config $TEMP_CONFIG \
            --num_workers $WORKERS \
            --seed 42 \
            >> "$EXP_LOG" 2>&1 && TRAIN_EXIT=0 || TRAIN_EXIT=$?

        # 提取
        BEST_LOSS=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Loss: [\d.]+' | sed 's/Loss: //' || true)
        BEST_ACC_RAW=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Acc: [\d.]+' | sed 's/Acc: //' || true)
        BEST_ACC=$(awk "BEGIN {printf \"%.1f\", ${BEST_ACC_RAW:-0} * 100}")
        BEST_EP=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP '第\d+轮' | sed 's/第//;s/轮//' || true)
        TOTAL_EP=$(grep -c "Epoch [0-9]*/100" "$EXP_LOG" || true)

        echo "  [训练] 退出码=$TRAIN_EXIT" | tee -a "$EXP_LOG"

        # 测试
        CKPT_DIR="checkpoints/unsupervised/${DATASET}"
        BEST_CKPT=$(ls -t "$CKPT_DIR"/*_best.pth 2>/dev/null | head -1)
        if [ -n "$BEST_CKPT" ]; then
            echo "  [测试] $(basename "$BEST_CKPT")"
            TEST_OUTPUT=$($CONDA_PYTHON test_unsupervised_stn.py \
                --dataset_name $DATASET \
                --ckpt_path "$BEST_CKPT" \
                --visual_batches 0 \
                --batch_size 64 \
                2>&1)
            TEST_ACC=$(echo "$TEST_OUTPUT" | grep "Top-1 Acc:" | tail -1 | grep -oP '[\d.]+(?=%)')
            echo "$TEST_OUTPUT" >> "$EXP_LOG"
            echo "  [测试] Acc: ${TEST_ACC:-N/A}%" | tee -a "$EXP_LOG"
        else
            TEST_ACC="N/A"
            echo "  [测试] 未找到检查点" | tee -a "$EXP_LOG"
        fi

        RESULT="${DATASET} | Grid | Ep=${TOTAL_EP:-?} | BestEp=${BEST_EP:-?} | ValAcc=${BEST_ACC:-?}% | TestAcc=${TEST_ACC:-N/A}% | BestLoss=${BEST_LOSS:-?}"
        ALL_RESULTS+=("$RESULT")
        echo "$RESULT" >> "$RESULTS_FILE"

        if [ $TRAIN_EXIT -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
        else
            FAILED=$((FAILED + 1))
        fi

        rm -f "$TEMP_CONFIG"
        pkill -f "train_unsupervised_ddp" 2>/dev/null || true
        sleep 2

    else
        # 单参数模式
        # 短名 → 完整 YAML 路径
        if [[ "$MODE" == *"."* ]]; then
            YAML_PATH="$MODE"
        else
            YAML_PATH="${PARAM_MAP[$MODE]:-$MODE}"
        fi
        VALUES_STR="$REST"
        PARAM_SHORT=$(echo "$YAML_PATH" | sed 's/.*\.//')
        IFS=',' read -ra VALUES <<< "$VALUES_STR"

        for VAL in "${VALUES[@]}"; do
            VAL=$(echo "$VAL" | xargs)
            VAL_SAFE=$(echo "$VAL" | sed 's/\./-/g' | sed 's/e/E/g')
            TOTAL_EXPS=$((TOTAL_EXPS + 1))

            EXP_NAME="${DATASET}_${PARAM_SHORT}_${VAL_SAFE}"
            EXP_LOG="${LOG_DIR}/${EXP_NAME}.log"

            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  [$TOTAL_EXPS] $DATASET | $YAML_PATH = $VAL"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""

            # 生成临时配置
            TEMP_CONFIG="/tmp/sweep_${EXP_NAME}_${TIMESTAMP}.yaml"
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
print(f'  Config: $YAML_PATH = {val}')
" 2>&1 | tee -a "$EXP_LOG"

            # 训练
            echo "  [训练] 开始..."
            PYTHONUNBUFFERED=1 $CONDA_PYTHON -u -m torch.distributed.run \
                --nproc_per_node=$GPUS \
                train_unsupervised_ddp.py \
                --dataset $DATASET \
                --config $TEMP_CONFIG \
                --num_workers $WORKERS \
                --seed 42 \
                >> "$EXP_LOG" 2>&1 && TRAIN_EXIT=0 || TRAIN_EXIT=$?

            # 提取训练结果
            TRAIN_BEST_LOSS=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Loss: [\d.]+' | sed 's/Loss: //' || true)
            TRAIN_BEST_ACC_RAW=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Acc: [\d.]+' | sed 's/Acc: //' || true)
            TRAIN_BEST_ACC=$(awk "BEGIN {printf \"%.1f\", ${TRAIN_BEST_ACC_RAW:-0} * 100}")
            TRAIN_BEST_EP=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP '第\d+轮' | sed 's/第//;s/轮//' || true)
            TRAIN_TOTAL_EP=$(grep -c "Epoch [0-9]*/100" "$EXP_LOG" || true)

            echo "  [训练] 退出码=$TRAIN_EXIT" | tee -a "$EXP_LOG"

            # 测试
            CKPT_DIR="checkpoints/unsupervised/${DATASET}"
            BEST_CKPT=$(ls -t "$CKPT_DIR"/*_best.pth 2>/dev/null | head -1)

            if [ -n "$BEST_CKPT" ]; then
                echo "  [测试] $(basename "$BEST_CKPT")"
                TEST_OUTPUT=$($CONDA_PYTHON test_unsupervised_stn.py \
                    --dataset_name $DATASET \
                    --ckpt_path "$BEST_CKPT" \
                    --visual_batches 0 \
                    --batch_size 64 \
                    2>&1)
                TEST_ACC=$(echo "$TEST_OUTPUT" | grep "Top-1 Acc:" | tail -1 | grep -oP '[\d.]+(?=%)')
                echo "$TEST_OUTPUT" >> "$EXP_LOG"
                echo "  [测试] Acc: ${TEST_ACC:-N/A}%" | tee -a "$EXP_LOG"
            else
                TEST_ACC="N/A"
                echo "  [测试] 未找到检查点" | tee -a "$EXP_LOG"
            fi

            RESULT="${DATASET} | ${PARAM_SHORT}=${VAL} | Ep=${TRAIN_TOTAL_EP:-?} | BestEp=${TRAIN_BEST_EP:-?} | ValAcc=${TRAIN_BEST_ACC:-?}% | TestAcc=${TEST_ACC:-N/A}% | BestLoss=${TRAIN_BEST_LOSS:-?}"
            ALL_RESULTS+=("$RESULT")
            echo "$RESULT" >> "$RESULTS_FILE"

            if [ $TRAIN_EXIT -eq 0 ]; then
                COMPLETED=$((COMPLETED + 1))
            else
                FAILED=$((FAILED + 1))
            fi

            rm -f "$TEMP_CONFIG"
            pkill -f "train_unsupervised_ddp" 2>/dev/null || true
            sleep 2
        done
    fi
done

# ============================================================================
# 汇总
# ============================================================================
echo ""
echo "=============================================="
echo "  全部完成"
echo "=============================================="
echo "总计: $TOTAL_EXPS | 成功: $COMPLETED | 失败: $FAILED"
echo ""

# 按数据集分组
for ds in oxford_pets cub dtd food101; do
    count=$(printf '%s\n' "${ALL_RESULTS[@]}" | grep -c "^${ds} |")
    [[ $count -eq 0 ]] && continue
    echo "━━━ $ds ━━━"
    for r in "${ALL_RESULTS[@]}"; do
        if [[ "$r" == "$ds |"* ]]; then
            echo "  $r"
        fi
    done
    echo ""
done

echo "📁 完整日志: $LOG_DIR/"
echo "📁 汇总文件: $RESULTS_FILE"
