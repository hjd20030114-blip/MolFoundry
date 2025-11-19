#!/bin/bash

echo "=========================================="
echo "PRRSV抑制剂设计项目训练启动脚本"
echo "=========================================="

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装或未在PATH中"
    exit 1
fi

# 激活conda环境
echo "🔧 激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prrsv

if [ $? -ne 0 ]; then
    echo "❌ 无法激活prrsv环境"
    exit 1
fi

echo "✅ 环境激活成功"

# 检查Python包
echo "🔍 检查关键依赖..."
python -c "
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
print('✅ 核心包导入成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
"

if [ $? -ne 0 ]; then
    echo "❌ 依赖检查失败"
    exit 1
fi

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 检查数据目录
echo "📁 检查数据目录..."
if [ ! -d "data/P-L" ]; then
    echo "❌ P-L数据目录不存在: data/P-L"
    exit 1
fi

# 统计数据文件
pdb_count=$(find data/P-L -name "*.pdb" | wc -l)
sdf_count=$(find data/P-L -name "*.sdf" | wc -l)
echo "✅ 找到 $pdb_count 个PDB文件, $sdf_count 个SDF文件"

# 创建必要目录
mkdir -p logs/$(date +%Y%m%d_%H%M%S)
mkdir -p results

# 检查配置文件
if [ ! -f "config/train_config.yaml" ]; then
    echo "❌ 配置文件不存在: config/train_config.yaml"
    exit 1
fi

echo "✅ 配置文件检查通过"

# 选择模型类型
MODEL_TYPE=${1:-"pl_pair_classifier"}
echo "🤖 使用模型: $MODEL_TYPE"

# 启动训练
echo "=========================================="
echo "🚀 开始训练..."
echo "=========================================="

python train.py \
    --config config/train_config.yaml \
    --model $MODEL_TYPE \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ 训练完成!"
    echo "=========================================="
    echo "📊 日志文件: logs/training_$(date +%Y%m%d_%H%M%S).log"
    echo "💾 模型文件: logs/final_model.pth"
else
    echo "=========================================="
    echo "❌ 训练失败!"
    echo "=========================================="
    exit 1
fi
