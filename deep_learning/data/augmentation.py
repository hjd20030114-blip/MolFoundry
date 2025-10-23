#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
包含分子和蛋白质数据增强方法
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """数据增强配置"""
    rotation_prob: float = 0.5
    noise_prob: float = 0.3
    noise_std: float = 0.1
    flip_prob: float = 0.2
    scale_prob: float = 0.2
    scale_range: Tuple[float, float] = (0.9, 1.1)

class MolecularAugmentation:
    """分子数据增强"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
    def augment_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        增强分子坐标
        
        Args:
            coords: 分子坐标张量 [N, 3]
            
        Returns:
            增强后的坐标
        """
        augmented_coords = coords.clone()
        
        # 随机旋转
        if np.random.random() < self.config.rotation_prob:
            augmented_coords = self._random_rotation(augmented_coords)
            
        # 添加噪声
        if np.random.random() < self.config.noise_prob:
            augmented_coords = self._add_noise(augmented_coords)
            
        # 随机缩放
        if np.random.random() < self.config.scale_prob:
            augmented_coords = self._random_scale(augmented_coords)
            
        return augmented_coords
    
    def _random_rotation(self, coords: torch.Tensor) -> torch.Tensor:
        """随机旋转"""
        # 生成随机旋转矩阵
        angles = torch.rand(3) * 2 * np.pi
        
        # 绕X轴旋转
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])]
        ], dtype=coords.dtype)
        
        # 绕Y轴旋转
        Ry = torch.tensor([
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
        ], dtype=coords.dtype)
        
        # 绕Z轴旋转
        Rz = torch.tensor([
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1]
        ], dtype=coords.dtype)
        
        # 组合旋转矩阵
        R = torch.mm(torch.mm(Rz, Ry), Rx)
        
        return torch.mm(coords, R.T)
    
    def _add_noise(self, coords: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(coords) * self.config.noise_std
        return coords + noise
    
    def _random_scale(self, coords: torch.Tensor) -> torch.Tensor:
        """随机缩放"""
        scale_min, scale_max = self.config.scale_range
        scale = np.random.uniform(scale_min, scale_max)
        return coords * scale
    
    def augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        增强分子特征
        
        Args:
            features: 分子特征张量
            
        Returns:
            增强后的特征
        """
        augmented_features = features.clone()
        
        # 特征噪声
        if np.random.random() < self.config.noise_prob:
            noise = torch.randn_like(features) * self.config.noise_std * 0.1
            augmented_features = augmented_features + noise
            
        return augmented_features

class ProteinAugmentation:
    """蛋白质数据增强"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
    def augment_structure(self, coords: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强蛋白质结构
        
        Args:
            coords: 蛋白质坐标 [N, 3]
            features: 蛋白质特征 [N, F]
            
        Returns:
            (增强后的坐标, 增强后的特征)
        """
        augmented_coords = coords.clone()
        augmented_features = features.clone()
        
        # 随机旋转
        if np.random.random() < self.config.rotation_prob:
            augmented_coords = self._random_rotation(augmented_coords)
            
        # 添加结构噪声
        if np.random.random() < self.config.noise_prob:
            augmented_coords = self._add_structural_noise(augmented_coords)
            
        # 特征增强
        if np.random.random() < self.config.noise_prob:
            augmented_features = self._augment_features(augmented_features)
            
        return augmented_coords, augmented_features
    
    def _random_rotation(self, coords: torch.Tensor) -> torch.Tensor:
        """随机旋转（与分子增强相同）"""
        angles = torch.rand(3) * 2 * np.pi
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])]
        ], dtype=coords.dtype)
        
        Ry = torch.tensor([
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
        ], dtype=coords.dtype)
        
        Rz = torch.tensor([
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1]
        ], dtype=coords.dtype)
        
        R = torch.mm(torch.mm(Rz, Ry), Rx)
        return torch.mm(coords, R.T)
    
    def _add_structural_noise(self, coords: torch.Tensor) -> torch.Tensor:
        """添加结构噪声（较小的噪声以保持蛋白质结构完整性）"""
        noise = torch.randn_like(coords) * self.config.noise_std * 0.5
        return coords + noise
    
    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """增强蛋白质特征"""
        augmented_features = features.clone()
        
        # 添加特征噪声
        noise = torch.randn_like(features) * self.config.noise_std * 0.05
        augmented_features = augmented_features + noise
        
        return augmented_features

class InteractionAugmentation:
    """相互作用数据增强"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.molecular_aug = MolecularAugmentation(config)
        self.protein_aug = ProteinAugmentation(config)
        
    def augment_complex(self, 
                       protein_coords: torch.Tensor,
                       protein_features: torch.Tensor,
                       ligand_coords: torch.Tensor,
                       ligand_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        增强蛋白质-配体复合物
        
        Args:
            protein_coords: 蛋白质坐标
            protein_features: 蛋白质特征
            ligand_coords: 配体坐标
            ligand_features: 配体特征
            
        Returns:
            (增强后的蛋白质坐标, 蛋白质特征, 配体坐标, 配体特征)
        """
        # 对整个复合物应用相同的旋转
        if np.random.random() < self.config.rotation_prob:
            # 生成旋转矩阵
            angles = torch.rand(3) * 2 * np.pi
            R = self._create_rotation_matrix(angles, protein_coords.dtype)
            
            # 应用旋转
            protein_coords = torch.mm(protein_coords, R.T)
            ligand_coords = torch.mm(ligand_coords, R.T)
        
        # 分别增强蛋白质和配体
        protein_coords, protein_features = self.protein_aug.augment_structure(
            protein_coords, protein_features
        )
        ligand_coords = self.molecular_aug.augment_coordinates(ligand_coords)
        ligand_features = self.molecular_aug.augment_features(ligand_features)
        
        return protein_coords, protein_features, ligand_coords, ligand_features
    
    def _create_rotation_matrix(self, angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """创建旋转矩阵"""
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])]
        ], dtype=dtype)
        
        Ry = torch.tensor([
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
        ], dtype=dtype)
        
        Rz = torch.tensor([
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1]
        ], dtype=dtype)
        
        return torch.mm(torch.mm(Rz, Ry), Rx)
