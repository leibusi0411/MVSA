#!/bin/bash
# 参数扫描：支持单参数多值 / 多参数网格搜索
#
# 用法:
#   # 单参数
#   ./run_sweep.sh --datasets oxford_pets --param warmup_epochs --values 8,12
#
#   # 多参数网格
#   ./run_sweep.sh --datasets oxford_pets --params warmup_epochs:8,12 teacher_temp:0.05,0.07
#
#   # 多数据集
#   ./run_sweep.sh --datasets oxford_pets,cub --param warmup_epochs --values 8,12
#
# 短名:
#   warmup_epochs, teacher_temp, student_temp, dec_temp, decorr, lr, batch_size

set -e

declare -A PARAM_MAP=(
    ["warmup_epochs"]="stn_config.two_stage.warmup_epochs"
    ["teacher_temp"]="stn_config.two_stage.teacher_temp"
    ["student_temp"]="stn_config.two_stage.warmup_student_temp"
    ["dec_temp"]="stn_config.two_stage.dec_student_temp"
    ["decorr"]="stn_config.decorrelation_weight"
    ["fairness"]="stn_config.fairness_weight"
    ["lr"]="training.learning_rate"
    ["batch_size"]="training.batch_size"
)

# 解析
DATASETS=()
GRID_KEY_VALS=()  # "YAML_path:val1,val2,val3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
        --dataset)  DATASETS=("$2"); shift 2 ;;
        --param)
            K="${PARAM_MAP[$2]:-$2}"
            GRID_KEY_VALS+=("$K:$3")
            shift 3
            ;;
        --values) shift ;;  # consumed by --param
        --params)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                K="${1%%:*}"
                V="${1#*:}"
                K="${PARAM_MAP[$K]:-$K}"
                GRID_KEY_VALS+=("$K:$V")
                shift
            done
            ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

GPUS=${GPUS:-2}
WORKERS=${WORKERS:-8}
CONDA_PYTHON="/mnt/e3319bd7-a0cc-41a8-9825-36b781a06ce8/xzy/anaconda3/envs/wca/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/sweep/${TIMESTAMP}"
RESULTS_FILE="${LOG_DIR}/results.txt"
mkdir -p "$LOG_DIR"

# 生成所有参数组合（用 Python 做笛卡尔积）
COMBO_FILE="/tmp/sweep_combos_${TIMESTAMP}.txt"
$CONDA_PYTHON -c "
import itertools
items = [$(for kv in "${GRID_KEY_VALS[@]}"; do
    k="${kv%%:*}"; v="${kv#*:}"
    printf "('%s', '%s')," "$k" "$v"
done)]
keys = [x[0] for x in items]
val_lists = [x[1].split(',') for x in items]
for combo in itertools.product(*val_lists):
    line = ' '.join(f'{k}:{v}' for k, v in zip(keys, combo))
    print(line)
" > "$COMBO_FILE"

mapfile -t COMBOS < "$COMBO_FILE"
TOTAL_EXPS=$((${#DATASETS[@]} * ${#COMBOS[@]}))
echo "=============================================="
echo "  参数扫描"
echo "=============================================="
echo "数据集: ${DATASETS[*]}"
echo "参数组合: ${#COMBOS[@]}"
echo "总实验数: $TOTAL_EXPS"
echo "GPU: $GPUS | Workers: $WORKERS"
echo "日志: $LOG_DIR"
echo ""

N=0
for ds in "${DATASETS[@]}"; do
    BASE_CONFIG="UN-STN-Config/${ds}.yaml"
    [[ ! -f "$BASE_CONFIG" ]] && { echo "❌ $BASE_CONFIG 不存在"; exit 1; }

    for COMBO in "${COMBOS[@]}"; do
        N=$((N + 1))
        # 构建实验名
        EXP_NAME="${ds}"
        for pair in $COMBO; do
            K="${pair%%:*}"
            V="${pair#*:}"
            KS=$(echo "$K" | sed 's/.*\.//')
            EXP_NAME="${EXP_NAME}_${KS}${V}"
        done

        EXP_LOG="${LOG_DIR}/${EXP_NAME}.log"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$N/$TOTAL_EXPS] $EXP_NAME"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # 生成临时配置
        TEMP_CONFIG="/tmp/sweep_${EXP_NAME}_${TIMESTAMP}.yaml"
        OVERRIDES="{"
        for pair in $COMBO; do
            K="${pair%%:*}"
            V="${pair#*:}"
            # 尝试转数字
            if [[ "$V" =~ ^[0-9]+$ ]]; then
                OVERRIDES+="'$K': $V, "
            elif [[ "$V" =~ ^[0-9]+\.[0-9]+$ ]] || [[ "$V" =~ e ]]; then
                OVERRIDES+="'$K': $V, "
            else
                OVERRIDES+="'$K': '$V', "
            fi
        done
        OVERRIDES+="}"

        $CONDA_PYTHON -c "
import yaml
with open('$BASE_CONFIG', 'r') as f:
    c = yaml.safe_load(f)

overrides = $OVERRIDES
for key_path, val in overrides.items():
    parts = key_path.split('.')
    d = c
    for p in parts[:-1]:
        if p not in d: d[p] = {}
        d = d[p]
    d[parts[-1]] = val

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'  Config written: $TEMP_CONFIG')
" 2>&1 | tee -a "$EXP_LOG"

        # 训练
        echo "  [训练] 开始..."
        $CONDA_PYTHON -u -m torch.distributed.run \
            --nproc_per_node=$GPUS \
            train_unsupervised_ddp.py \
            --dataset $ds \
            --config $TEMP_CONFIG \
            --num_workers $WORKERS \
            --seed 42 \
            >> "$EXP_LOG" 2>&1

        # 提取训练结果
        BEST_ACC_RAW=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Acc: [\d.]+' | sed 's/Acc: //' || echo "0")
        BEST_ACC=$(awk "BEGIN {printf \"%.1f\", ${BEST_ACC_RAW:-0} * 100}")
        BEST_EP=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP '第\d+轮' | sed 's/第//;s/轮//' || echo "?")
        TOTAL_EP=$(grep -c "Epoch [0-9]*/100" "$EXP_LOG" || echo "?")

        # 测试
        BEST_CKPT=$(ls -t "checkpoints/unsupervised/${ds}"/*_best.pth 2>/dev/null | head -1)
        if [[ -n "$BEST_CKPT" ]]; then
            TEST_ACC=$($CONDA_PYTHON test_unsupervised_stn.py \
                --dataset_name $ds --ckpt_path "$BEST_CKPT" \
                --visual_batches 0 --batch_size 64 2>&1 \
                | grep "Top-1 Acc:" | grep -oP '[\d.]+(?=%)')
        else
            TEST_ACC="N/A"
        fi

        RESULT="$EXP_NAME | Ep=${TOTAL_EP} | BestEp=${BEST_EP} | ValAcc=${BEST_ACC}% | TestAcc=${TEST_ACC:-N/A}%"
        echo "  $RESULT" | tee -a "$RESULTS_FILE"

        rm -f "$TEMP_CONFIG"
        pkill -f "train_unsupervised_ddp" 2>/dev/null || true
        sleep 2
    done
done

rm -f "$COMBO_FILE"
echo ""
echo "=============================================="
echo "  全部完成"
echo "=============================================="
cat "$RESULTS_FILE"
echo ""
echo "📁 $LOG_DIR"
