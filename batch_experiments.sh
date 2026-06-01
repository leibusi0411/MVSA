#!/bin/bash
# 批量实验脚本：多数据集 × 多参数 → 训练 → 测试 → 记录
#
# 用法:
#   ./batch_experiments.sh
#
# 工作流程:
#   1. 读取实验计划（本文件中的 EXPERIMENTS 数组）
#   2. 依次执行每个实验：训练 → 测试 → 记录结果
#   3. 所有输出保存到 logs/batch/ 下
#
# 自定义实验：修改下面的 EXPERIMENTS 数组

set -e

# ============================================================================
# Conda 环境
# ============================================================================
CONDA_ENV="wca"
CONDA_PYTHON="/mnt/e3319bd7-a0cc-41a8-9825-36b781a06ce8/xzy/anaconda3/envs/wca/bin/python"

# ============================================================================
# 实验计划（格式: "dataset param_name values"）
# 一行一个 sweep，依次顺序执行
# ============================================================================
EXPERIMENTS=(
    # ============================================================
    # Part 1: warmup_epochs
    # ============================================================
    # 测试两种 update mode 下阶段二是否有效（warmup=10）
    # ============================================================
    "oxford_pets mode periodic,ema"
    "cub         mode periodic,ema"
    "dtd         mode periodic,ema"
   
    
        # # ============================================================
    # # Part 6: teacher_temp (阶段一 teacher 温度)
    # # ============================================================
    # "oxford_pets teacher_t 0.05,0.07,0.09"
    # "cub         teacher_t 0.05,0.07,0.09"
    # "dtd         teacher_t 0.05,0.07,0.09"
    # "food101     teacher_t 0.05,0.07,0.09"

    # # ============================================================
    # # Part 4: warmup_student_temp (阶段一 student 温度)
    # # ============================================================
    # "oxford_pets warm_t 0.07,0.10,0.12"
    # "cub         warm_t 0.07,0.10,0.12"
    # "dtd         warm_t 0.07,0.10,0.12"
    # "food101     warm_t 0.07,0.10,0.12"

    # # ============================================================
    # # Part 5: ema_momentum (EMA teacher 更新动量)
    # # ============================================================
    # "oxford_pets ema_m 0.99,0.995,0.999"
    # "cub         ema_m 0.99,0.995,0.999"
    # "dtd         ema_m 0.99,0.995,0.999"
    # "food101     ema_m 0.99,0.995,0.999"
 

)



        # # ============================================================
    # # Part 2: kl_consistency_weight  第二阶段损失权重
    # # ============================================================
    # "oxford_pets kl 0.7,1.0,1.3"
    # "cub        kl 0.7,1.0,1.3"
    # "dtd        kl 0.7,1.0,1.3"
    # # "food101    kl 0.7,1.0,1.3"

    # ============================================================
    # Part 3: dec_target_temp (阶段二 teacher 温度)
    # ============================================================
    # "oxford_pets dec_t 0.05,0.07,0.09"
    # "cub        dec_t 0.05,0.07,0.09
    # "dtd        dec_t 0.05,0.07,0.09"
    # # "food101    dec_t 0.05,0.07,0.09"





# 默认 GPU 和 workers
NUM_GPUS=2
NUM_WORKERS=8

# ============================================================================
# 初始化
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG_DIR="logs/batch/${TIMESTAMP}"
RESULTS_FILE="${BATCH_LOG_DIR}/results.txt"
mkdir -p "$BATCH_LOG_DIR"

# 参数名 → YAML 路径映射
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
    ["period_u"]="stn_config.two_stage.target_update_interval_epochs"
    ["mode"]="stn_config.two_stage.target_update_mode"
)

echo "=============================================="
echo "  批量实验"
echo "=============================================="
echo "实验数:   ${#EXPERIMENTS[@]}"
echo "GPU:      $NUM_GPUS"
echo "Workers:  $NUM_WORKERS"
echo "日志目录: $BATCH_LOG_DIR"
echo "=============================================="
echo ""

# 统计
TOTAL_EXPS=0
COMPLETED=0
FAILED=0

# 记录每个实验的汇总信息
declare -a ALL_RESULTS

for EXP in "${EXPERIMENTS[@]}"; do
    # 跳过空行和注释
    [[ -z "$EXP" || "$EXP" == \#* ]] && continue

    read -r DATASET PARAM_NAME VALUES_STR <<< "$EXP"
    IFS=',' read -ra VALUES <<< "$VALUES_STR"

    # 解析参数路径
    if [[ "$PARAM_NAME" == *"."* ]]; then
        YAML_PATH="$PARAM_NAME"
    else
        YAML_PATH="${PARAM_MAP[$PARAM_NAME]:-}"
        if [ -z "$YAML_PATH" ]; then
            echo "❌ 未知参数: $PARAM_NAME"
            continue
        fi
    fi
    PARAM_SHORT=$(echo "$YAML_PATH" | sed 's/.*\.//')

    BASE_CONFIG="UN-STN-Config/${DATASET}.yaml"
    if [ ! -f "$BASE_CONFIG" ]; then
        echo "❌ 配置文件不存在: $BASE_CONFIG"
        continue
    fi

    for VAL in "${VALUES[@]}"; do
        VAL=$(echo "$VAL" | xargs)
        VAL_SAFE=$(echo "$VAL" | sed 's/\./-/g' | sed 's/e/E/g')
        TOTAL_EXPS=$((TOTAL_EXPS + 1))

        EXP_NAME="${DATASET}_${PARAM_SHORT}_${VAL_SAFE}"
        EXP_LOG="${BATCH_LOG_DIR}/${EXP_NAME}.log"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$TOTAL_EXPS] $DATASET | $YAML_PATH = $VAL"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # ---- 1) 生成临时配置 ----
        TEMP_CONFIG="/tmp/batch_${EXP_NAME}_${TIMESTAMP}.yaml"
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

        # ---- 2) 训练 ----
        echo "  [训练] 开始..."
        $CONDA_PYTHON -u -m torch.distributed.run \
            --nproc_per_node=$NUM_GPUS \
            train_unsupervised_ddp.py \
            --dataset $DATASET \
            --config $TEMP_CONFIG \
            --num_workers $NUM_WORKERS \
            --seed 42 \
            >> "$EXP_LOG" 2>&1
        TRAIN_EXIT=$?

        # ---- 3) 提取训练关键信息 ----
        TRAIN_FINAL_EPOCH=$(grep "Epoch " "$EXP_LOG" | tail -1 | grep -oP 'Epoch \d+/\d+' | head -1 || true)
        TRAIN_BEST_LOSS=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Loss: [\d.]+' | sed 's/Loss: //' || true)
        TRAIN_BEST_ACC_RAW=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP 'Acc: [\d.]+' | sed 's/Acc: //' || true)
        TRAIN_BEST_ACC=$(awk "BEGIN {printf \"%.1f\", ${TRAIN_BEST_ACC_RAW:-0} * 100}")
        TRAIN_BEST_EPOCH=$(grep "新最佳Loss" "$EXP_LOG" | tail -1 | grep -oP '第\d+轮' | sed 's/第//;s/轮//' || true)
        TRAIN_TOTAL_EPOCHS=$(grep -c "Epoch [0-9]*/100" "$EXP_LOG" || true)
        TRAIN_EARLY=$(grep -c "早停触发" "$EXP_LOG" || true)

        echo "  [训练] 退出码=$TRAIN_EXIT" | tee -a "$EXP_LOG"

        # ---- 4) 测试 ----
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

        # ---- 5) 写入汇总文件（每个实验完成后立即追加一行） ----
        echo "${DATASET} | ${PARAM_SHORT}=${VAL} | TrainEpochs=${TRAIN_TOTAL_EPOCHS:-?} | BestEp=${TRAIN_BEST_EPOCH:-?} | ValAcc=${TRAIN_BEST_ACC:-?}% | TestAcc=${TEST_ACC:-N/A}% | BestLoss=${TRAIN_BEST_LOSS:-?}" >> "$RESULTS_FILE"

        RESULT_SUMMARY="${DATASET} | ${PARAM_SHORT}=${VAL} → Val:${TRAIN_BEST_ACC:-?}% @Ep${TRAIN_BEST_EPOCH:-?} | Test:${TEST_ACC:-N/A}%"
        ALL_RESULTS+=("$RESULT_SUMMARY")

        if [ $TRAIN_EXIT -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
        else
            FAILED=$((FAILED + 1))
        fi

        # 清理
        rm -f "$TEMP_CONFIG"
        pkill -f "train_unsupervised_ddp" 2>/dev/null || true
        sleep 2
    done
done

# ============================================================================
# 最终汇总
# ============================================================================
echo ""
echo "=============================================="
echo "  批量实验完成"
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

echo "📁 完整日志: $BATCH_LOG_DIR/"
echo "📁 汇总文件: $RESULTS_FILE"
