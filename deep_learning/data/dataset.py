#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集类定义
包含分子、蛋白质和口袋-配体相互作用数据集
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """分子数据集"""
    
    def __init__(self, molecules: List[str], properties: Optional[Dict] = None):
        """
        初始化分子数据集
        
        Args:
            molecules: SMILES字符串列表
            properties: 分子性质字典
        """
        self.molecules = molecules
        self.properties = properties or {}
        
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        mol_data = {
            'smiles': self.molecules[idx],
            'idx': idx
        }
        
        # 添加性质数据
        for prop_name, prop_values in self.properties.items():
            if idx < len(prop_values):
                mol_data[prop_name] = prop_values[idx]
                
        return mol_data

class ProteinDataset(Dataset):
    """蛋白质数据集"""
    
    def __init__(self, proteins: List[str], sequences: Optional[List[str]] = None):
        """
        初始化蛋白质数据集
        
        Args:
            proteins: 蛋白质PDB文件路径列表
            sequences: 蛋白质序列列表
        """
        self.proteins = proteins
        self.sequences = sequences or []
        
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_data = {
            'pdb_path': self.proteins[idx],
            'idx': idx
        }
        
        if idx < len(self.sequences):
            protein_data['sequence'] = self.sequences[idx]
            
        return protein_data

class PocketLigandDataset(Dataset):
    """口袋-配体相互作用数据集"""
    
    def __init__(self, 
                 pocket_data: List[Dict],
                 ligand_data: List[Dict],
                 interaction_data: Optional[List[Dict]] = None):
        """
        初始化口袋-配体数据集
        
        Args:
            pocket_data: 口袋数据列表
            ligand_data: 配体数据列表  
            interaction_data: 相互作用数据列表
        """
        assert len(pocket_data) == len(ligand_data), "口袋和配体数据长度必须相同"
        
        self.pocket_data = pocket_data
        self.ligand_data = ligand_data
        self.interaction_data = interaction_data or [{}] * len(pocket_data)
        
    def __len__(self):
        return len(self.pocket_data)
    
    def __getitem__(self, idx):
        return {
            'pocket': self.pocket_data[idx],
            'ligand': self.ligand_data[idx],
            'interaction': self.interaction_data[idx],
            'idx': idx
        }

class PRRSVDataset(PocketLigandDataset):
    """PRRSV特化数据集"""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        初始化PRRSV数据集
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        
        # 加载PRRSV特定数据
        pocket_data, ligand_data, interaction_data = self._load_prrsv_data()
        
        super().__init__(pocket_data, ligand_data, interaction_data)
        
    def _load_prrsv_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """加载PRRSV数据"""
        pocket_data = []
        ligand_data = []
        interaction_data = []
        
        # 这里应该实现实际的数据加载逻辑
        # 目前返回空数据作为占位符
        logger.warning("PRRSV数据加载功能尚未实现，返回空数据集")
        
        return pocket_data, ligand_data, interaction_data

def create_dataset(dataset_type: str, **kwargs) -> Dataset:
    """
    创建数据集的工厂函数
    
    Args:
        dataset_type: 数据集类型 ('molecular', 'protein', 'pocket_ligand', 'prrsv')
        **kwargs: 数据集参数
        
    Returns:
        Dataset实例
    """
    if dataset_type == 'molecular':
        return MolecularDataset(**kwargs)
    elif dataset_type == 'protein':
        return ProteinDataset(**kwargs)
    elif dataset_type == 'pocket_ligand':
        return PocketLigandDataset(**kwargs)
    elif dataset_type == 'prrsv':
        return PRRSVDataset(**kwargs)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")

def collate_molecular_batch(batch: List[Dict]) -> Dict:
    """分子数据批处理函数"""
    collated = {}
    
    # 收集所有键
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    # 对每个键进行批处理
    for key in all_keys:
        values = [item.get(key) for item in batch]
        
        if key == 'smiles':
            collated[key] = values
        elif key == 'idx':
            collated[key] = torch.tensor(values)
        else:
            # 尝试转换为tensor
            try:
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    collated[key] = torch.tensor([v for v in values if v is not None])
                else:
                    collated[key] = values
            except:
                collated[key] = values
                
    return collated

def collate_pocket_ligand_batch(batch: List[Dict]) -> Dict:
    """口袋-配体数据批处理函数"""
    collated = {
        'pocket': [],
        'ligand': [],
        'interaction': [],
        'idx': torch.tensor([item['idx'] for item in batch])
    }
    
    for item in batch:
        collated['pocket'].append(item['pocket'])
        collated['ligand'].append(item['ligand'])
        collated['interaction'].append(item['interaction'])
        
    return collated
