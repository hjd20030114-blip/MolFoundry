#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器
包含数据加载配置和创建函数
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .dataset import (
    MolecularDataset, ProteinDataset, PocketLigandDataset,
    collate_molecular_batch, collate_pocket_ligand_batch
)

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """数据配置类"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    def __post_init__(self):
        """验证配置"""
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "训练、验证和测试集比例之和必须为1"

def create_data_loaders(dataset, 
                       config: DataConfig,
                       dataset_type: str = 'molecular') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        dataset: 数据集实例
        config: 数据配置
        dataset_type: 数据集类型，用于选择合适的collate函数
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 计算分割大小
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 选择合适的collate函数
    if dataset_type in ['molecular', 'protein']:
        collate_fn = collate_molecular_batch
    elif dataset_type in ['pocket_ligand', 'prrsv']:
        collate_fn = collate_pocket_ligand_batch
    else:
        collate_fn = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"创建数据加载器: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}, 测试集{len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def create_inference_loader(dataset, 
                           config: DataConfig,
                           dataset_type: str = 'molecular') -> DataLoader:
    """
    创建推理数据加载器
    
    Args:
        dataset: 数据集实例
        config: 数据配置
        dataset_type: 数据集类型
        
    Returns:
        DataLoader实例
    """
    # 选择合适的collate函数
    if dataset_type in ['molecular', 'protein']:
        collate_fn = collate_molecular_batch
    elif dataset_type in ['pocket_ligand', 'prrsv']:
        collate_fn = collate_pocket_ligand_batch
    else:
        collate_fn = None
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"创建推理数据加载器: {len(dataset)}个样本")
    
    return loader

class DataLoaderManager:
    """数据加载器管理器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.loaders = {}
        
    def register_loader(self, name: str, loader: DataLoader):
        """注册数据加载器"""
        self.loaders[name] = loader
        logger.info(f"注册数据加载器: {name}")
        
    def get_loader(self, name: str) -> DataLoader:
        """获取数据加载器"""
        if name not in self.loaders:
            raise KeyError(f"未找到数据加载器: {name}")
        return self.loaders[name]
        
    def list_loaders(self) -> List[str]:
        """列出所有数据加载器名称"""
        return list(self.loaders.keys())
        
    def create_and_register_loaders(self, 
                                   dataset, 
                                   dataset_type: str = 'molecular',
                                   prefix: str = ''):
        """创建并注册训练、验证、测试数据加载器"""
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset, self.config, dataset_type
        )
        
        self.register_loader(f'{prefix}train', train_loader)
        self.register_loader(f'{prefix}val', val_loader)
        self.register_loader(f'{prefix}test', test_loader)
        
        return train_loader, val_loader, test_loader
