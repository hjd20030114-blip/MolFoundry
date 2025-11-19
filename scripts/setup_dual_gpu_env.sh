#!/bin/bash
#############################################
# 双卡服务器环境初始化脚本
# 配置PyTorch、CUDA优化、依赖安装
#############################################

set -e

echo "=========================================="
echo "初始化双卡训练环境..."
echo "=========================================="

# ============================================
# 1. 检查CUDA版本
# ============================================

if ! command -v nvcc &> /dev/null; then
    echo "错误：未检测到CUDA，请先安装CUDA toolkit"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "CUDA版本: ${CUDA_VERSION}"

# ============================================
# 2. 检查Python环境
# ============================================

if ! command -v python &> /dev/null; then
    echo "错误：未检测到Python，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version | awk '{print $2}')
echo "Python版本: ${PYTHON_VERSION}"

# ============================================
# 3. 安装/升级PyTorch（CUDA 11.8+）
# ============================================

echo "检查PyTorch安装..."

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "PyTorch已安装: ${TORCH_VERSION}"
    
    # 检查CUDA支持
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "✓ CUDA支持正常"
    else
        echo "✗ PyTorch未启用CUDA，请重新安装"
        read -p "是否重新安装PyTorch? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        fi
    fi
else
    echo "PyTorch未安装，开始安装..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# ============================================
# 4. 安装深度学习依赖
# ============================================

echo "安装项目依赖..."

pip install --upgrade \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install --upgrade \
    e3nn \
    rdkit \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    pyyaml \
    tqdm \
    wandb \
    tensorboard

# ============================================
# 5. 配置NCCL（分布式训练优化）
# ============================================

echo "配置NCCL环境变量..."

# 添加到bashrc/zshrc
ENV_FILE="${HOME}/.bashrc"
if [ -f "${HOME}/.zshrc" ]; then
    ENV_FILE="${HOME}/.zshrc"
fi

if ! grep -q "NCCL_DEBUG" "${ENV_FILE}"; then
    cat >> "${ENV_FILE}" << 'EOF'

# PRRSV项目 - NCCL分布式训练优化
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网卡名修改
export OMP_NUM_THREADS=8
EOF
    echo "✓ 环境变量已添加到 ${ENV_FILE}"
    echo "  请运行: source ${ENV_FILE}"
fi

# ============================================
# 6. 测试分布式训练
# ============================================

echo "=========================================="
echo "测试分布式训练..."
echo "=========================================="

cat > /tmp/test_distributed.py << 'PYEOF'
import torch
import torch.distributed as dist
import os

def test_distributed():
    # 初始化进程组
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        
        # 测试GPU
        device = torch.device(f'cuda:{rank}')
        x = torch.randn(1000, 1000).to(device)
        y = torch.matmul(x, x.T)
        
        print(f"[GPU {rank}] 测试成功！张量形状: {y.shape}")
        
        dist.destroy_process_group()
    else:
        print("单GPU模式测试...")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.matmul(x, x.T)
            print(f"✓ GPU测试成功！可用GPU数: {torch.cuda.device_count()}")
        else:
            print("✗ CUDA不可用")

if __name__ == '__main__':
    test_distributed()
PYEOF

# 运行单GPU测试
python /tmp/test_distributed.py

# 运行双GPU测试
echo "测试双GPU通信..."
torchrun --nproc_per_node=2 /tmp/test_distributed.py

rm /tmp/test_distributed.py

# ============================================
# 7. 完成
# ============================================

echo "=========================================="
echo "✓ 双卡环境初始化完成！"
echo ""
echo "下一步："
echo "1. source ${ENV_FILE}  # 加载环境变量"
echo "2. cd /home/hjd/PRRSV/HJD"
echo "3. bash scripts/train_dual_gpu.sh pl_pair_classifier"
echo "=========================================="
