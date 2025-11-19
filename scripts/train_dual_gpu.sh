#!/bin/bash
#############################################
# 双卡分布式训练脚本（NVIDIA RTX 4090×2）
# Linux服务器专用
#############################################

set -e  # 遇错即停

# ============================================
# 1. 环境配置
# ============================================

# 项目根目录（根据实际路径修改）
export PROJECT_ROOT="/home/hjd/PRRSV"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# CUDA环境变量（优化性能）
export CUDA_VISIBLE_DEVICES=0,1          # 使用两块GPU
export NCCL_DEBUG=INFO                   # 调试分布式通信
export NCCL_IB_DISABLE=1                 # 若无InfiniBand则禁用
export OMP_NUM_THREADS=8                 # OpenMP线程数

# PyTorch优化
export TORCH_DISTRIBUTED_DEBUG=DETAIL    # 详细调试信息（生产环境可关闭）

# ============================================
# 2. 检查GPU可用性
# ============================================

echo "=========================================="
echo "检查GPU状态..."
echo "=========================================="
nvidia-smi

# 检测GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 ${GPU_COUNT} 块GPU"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "警告：仅检测到 ${GPU_COUNT} 块GPU，建议使用2块GPU以发挥最佳性能"
fi

# ============================================
# 3. 训练配置
# ============================================

# 选择训练的模型（可选：pl_pair_classifier, pocket_ligand_transformer, pocket_diffusion, equivariant_gnn）
MODEL_TYPE="${1:-pl_pair_classifier}"
# 第二参数为随机种子
SEED="${2:-42}"
# 第三个参数为 K-Fold 折数（>1 启用K折）
KFOLD="${3:-1}"
CONFIG_FILE="${PROJECT_ROOT}/HJD/configs/dual_gpu_config.yaml"

# 读取配置中的日志目录，统一到 train.py 内置目录
CONFIG_LOG_DIR=$(CONFIG_FILE="${CONFIG_FILE}" python - <<'PY'
import os, yaml
with open(os.environ['CONFIG_FILE'], 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg['logging']['log_dir'])
PY
)

LOG_DIR="${CONFIG_LOG_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "训练模型: ${MODEL_TYPE}"
echo "配置文件: ${CONFIG_FILE}"
echo "日志目录: ${LOG_DIR}"
echo "随机种子: ${SEED}"
echo "K-Fold: ${KFOLD}"
echo "=========================================="

# ============================================
# 4. 分布式训练启动（torchrun）
# ============================================

cd "${PROJECT_ROOT}/HJD" || exit 1

# 使用PyTorch原生分布式启动器（若启用K折，避免多进程写同一 results，改为单进程）
NPROC=2
if [ "${KFOLD}" -gt 1 ]; then
  NPROC=1
fi

torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    train.py \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_TYPE}" \
    --seed "${SEED}" \
    --kfold "${KFOLD}" \
    2>&1 | tee "${LOG_DIR}/training_seed${SEED}.log"

# ============================================
# 5. 训练完成后处理
# ============================================

echo "=========================================="
echo "训练完成！"
echo "最佳模型: ${LOG_DIR}/best_model.pth"
echo "训练日志: ${LOG_DIR}/training_seed${SEED}.log"
echo "=========================================="

# 显示GPU最终状态
nvidia-smi

# ============================================
# 6. 收集评估结果
# ============================================

if [ "${KFOLD}" -gt 1 ]; then
    # 复制K折汇总与折别文件
    SUM_FILE="results/kfold_summary_seed${SEED}_k${KFOLD}.json"
    if [ -f "${SUM_FILE}" ]; then
        cp "${SUM_FILE}" "${LOG_DIR}/kfold_summary_seed${SEED}_k${KFOLD}.json"
        echo "已复制K折汇总: ${LOG_DIR}/kfold_summary_seed${SEED}_k${KFOLD}.json"
    else
        echo "未找到 ${SUM_FILE}"
    fi
    for f in results/test_eval_seed${SEED}_fold*of${KFOLD}.json; do
        if [ -f "$f" ]; then
            cp "$f" "${LOG_DIR}/"
            echo "已复制折别评估: ${LOG_DIR}/$(basename "$f")"
        fi
    done
else
    if [ -f "results/test_eval.json" ]; then
        cp "results/test_eval.json" "${LOG_DIR}/test_eval_seed${SEED}.json"
        echo "已复制测试评估JSON到: ${LOG_DIR}/test_eval_seed${SEED}.json"
    else
        echo "未找到 results/test_eval.json（可能训练提前结束或发生异常）"
    fi
fi

echo "所有任务完成！"
