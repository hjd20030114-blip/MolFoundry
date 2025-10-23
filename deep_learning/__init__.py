"""
MolFoundry - 物理知情的分子生成与多级验证平台
基于扩散/Transformer/等变GNN等模型的分子生成与 Docking → DFT → MD 验证流水线
"""

__version__ = "1.0.0"
__author__ = "PRRSV Team"
__model_name__ = "MolFoundry"

# 避免在导入包时就加载所有子模块（可能引入如 pandas 等重依赖），
# 仅暴露子包名称，由使用方按需导入：
#   from deep_learning.training import Trainer
#   from deep_learning.data.pl_pair_dataset import PLPairDataset
# 等。
__all__ = [
    'models',
    'data',
    'training',
    'evaluation',
    'utils',
    '__version__',
    '__author__',
    '__model_name__'
]
