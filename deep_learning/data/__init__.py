"""
数据处理模块
包含数据集、特征化器、数据加载器等
"""

# 注意：为避免在导入 data 包时就加载重依赖（如 pandas），此处不导入具体子模块。
# 使用方请按需显式导入，例如：
#   from deep_learning.data.pl_pair_dataset import PLPairDataset
# 如需其它数据模块，请直接从对应子模块导入。

__all__ = [
    'pl_pair_dataset',
    'dataset',
    'featurizers',
    'data_loaders',
    'augmentation',
    'preprocessing'
]
