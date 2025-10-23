"""
深度学习模型模块
包含SE(3)-Equivariant GNN、Diffusion Model、Transformer等核心模型
"""

MODEL_NAME = "MolFoundry"

from .equivariant_gnn import EquivariantGNN, SE3TransformerLayer
from .diffusion_model import PocketConditionedDiffusion, DiffusionScheduler
from .transformer import PocketLigandTransformer, CrossAttentionLayer
from .discriminator import MultiTaskDiscriminator, BindingAffinityHead
from .pl_pair_classifier import PLPairClassifier  # 确保注册 P-L 二分类模型
from .base_model import BaseModel, ModelConfig, ModelRegistry

def create_model(model_type: str, config):
    """
    模型工厂函数（通过注册器创建）

    Args:
        model_type: 模型类型（需与注册名一致，例如：
            'equivariant_gnn', 'pocket_diffusion',
            'pocket_ligand_transformer', 'multitask_discriminator'）
        config: 模型配置，可以是 ModelConfig 或 dict

    Returns:
        模型实例
    """
    # 兼容 dict 与 ModelConfig
    if isinstance(config, dict):
        cfg_dict = dict(config)
        cfg_dict.setdefault('model_type', model_type)
        cfg = ModelConfig.from_dict(cfg_dict)
    elif isinstance(config, ModelConfig):
        cfg = config
    else:
        raise TypeError(f"config 必须是 dict 或 ModelConfig，当前类型: {type(config)}")

    # 通过注册器创建对应模型
    model_class = ModelRegistry.get_model(model_type)
    return model_class(cfg)

__all__ = [
    'MODEL_NAME',
    'EquivariantGNN',
    'SE3TransformerLayer',
    'PocketConditionedDiffusion',
    'DiffusionScheduler',
    'PocketLigandTransformer',
    'CrossAttentionLayer',
    'MultiTaskDiscriminator',
    'BindingAffinityHead',
    'PLPairClassifier',
    'BaseModel',
    'ModelConfig',
    'create_model'
]
