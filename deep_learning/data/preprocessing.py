#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
包含数据清洗、标准化、特征选择等功能
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """标准化配置"""
    method: str = 'standard'  # 'standard', 'minmax', 'robust'
    feature_range: Tuple[float, float] = (0, 1)
    with_mean: bool = True
    with_std: bool = True
    quantile_range: Tuple[float, float] = (25.0, 75.0)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.scalers = {}
        self.fitted = False
        
    def fit(self, data: Dict[str, torch.Tensor]):
        """
        拟合预处理器
        
        Args:
            data: 包含各种特征的数据字典
        """
        logger.info("开始拟合数据预处理器...")
        
        for feature_name, feature_data in data.items():
            if feature_data.dtype in [torch.float32, torch.float64]:
                scaler = self._create_scaler()
                
                # 转换为numpy数组进行拟合
                if feature_data.dim() > 1:
                    # 多维特征，reshape为2D
                    reshaped_data = feature_data.view(-1, feature_data.shape[-1]).numpy()
                else:
                    # 1D特征
                    reshaped_data = feature_data.view(-1, 1).numpy()
                
                scaler.fit(reshaped_data)
                self.scalers[feature_name] = scaler
                
                logger.info(f"拟合特征 {feature_name}: shape={feature_data.shape}")
        
        self.fitted = True
        logger.info("数据预处理器拟合完成")
    
    def transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        转换数据
        
        Args:
            data: 输入数据字典
            
        Returns:
            转换后的数据字典
        """
        if not self.fitted:
            raise RuntimeError("预处理器尚未拟合，请先调用fit方法")
        
        transformed_data = {}
        
        for feature_name, feature_data in data.items():
            if feature_name in self.scalers:
                scaler = self.scalers[feature_name]
                original_shape = feature_data.shape
                
                # 转换数据
                if feature_data.dim() > 1:
                    reshaped_data = feature_data.view(-1, feature_data.shape[-1]).numpy()
                    transformed = scaler.transform(reshaped_data)
                    transformed_tensor = torch.from_numpy(transformed).view(original_shape)
                else:
                    reshaped_data = feature_data.view(-1, 1).numpy()
                    transformed = scaler.transform(reshaped_data)
                    transformed_tensor = torch.from_numpy(transformed).view(original_shape)
                
                transformed_data[feature_name] = transformed_tensor.to(feature_data.dtype)
            else:
                # 非数值特征直接复制
                transformed_data[feature_name] = feature_data
        
        return transformed_data
    
    def fit_transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """拟合并转换数据"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        逆转换数据
        
        Args:
            data: 转换后的数据字典
            
        Returns:
            原始尺度的数据字典
        """
        if not self.fitted:
            raise RuntimeError("预处理器尚未拟合")
        
        inverse_data = {}
        
        for feature_name, feature_data in data.items():
            if feature_name in self.scalers:
                scaler = self.scalers[feature_name]
                original_shape = feature_data.shape
                
                # 逆转换
                if feature_data.dim() > 1:
                    reshaped_data = feature_data.view(-1, feature_data.shape[-1]).numpy()
                    inverse_transformed = scaler.inverse_transform(reshaped_data)
                    inverse_tensor = torch.from_numpy(inverse_transformed).view(original_shape)
                else:
                    reshaped_data = feature_data.view(-1, 1).numpy()
                    inverse_transformed = scaler.inverse_transform(reshaped_data)
                    inverse_tensor = torch.from_numpy(inverse_transformed).view(original_shape)
                
                inverse_data[feature_name] = inverse_tensor.to(feature_data.dtype)
            else:
                inverse_data[feature_name] = feature_data
        
        return inverse_data
    
    def _create_scaler(self):
        """创建标准化器"""
        if self.config.method == 'standard':
            return StandardScaler(
                with_mean=self.config.with_mean,
                with_std=self.config.with_std
            )
        elif self.config.method == 'minmax':
            return MinMaxScaler(feature_range=self.config.feature_range)
        elif self.config.method == 'robust':
            return RobustScaler(
                quantile_range=self.config.quantile_range,
                with_centering=self.config.with_mean,
                with_scaling=self.config.with_std
            )
        else:
            raise ValueError(f"未知的标准化方法: {self.config.method}")
    
    def get_feature_stats(self) -> Dict[str, Dict]:
        """获取特征统计信息"""
        stats = {}
        
        for feature_name, scaler in self.scalers.items():
            feature_stats = {}
            
            if hasattr(scaler, 'mean_'):
                feature_stats['mean'] = scaler.mean_
            if hasattr(scaler, 'scale_'):
                feature_stats['scale'] = scaler.scale_
            if hasattr(scaler, 'center_'):
                feature_stats['center'] = scaler.center_
            if hasattr(scaler, 'data_min_'):
                feature_stats['min'] = scaler.data_min_
            if hasattr(scaler, 'data_max_'):
                feature_stats['max'] = scaler.data_max_
                
            stats[feature_name] = feature_stats
        
        return stats

class MolecularPreprocessor(DataPreprocessor):
    """分子数据预处理器"""
    
    def __init__(self, config: NormalizationConfig):
        super().__init__(config)
        
    def preprocess_smiles(self, smiles_list: List[str]) -> List[str]:
        """
        预处理SMILES字符串
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            清洗后的SMILES列表
        """
        cleaned_smiles = []
        
        for smiles in smiles_list:
            # 基本清洗
            cleaned = smiles.strip()
            
            # 移除无效字符
            if cleaned and len(cleaned) > 0:
                cleaned_smiles.append(cleaned)
            else:
                logger.warning(f"跳过无效SMILES: {smiles}")
        
        logger.info(f"SMILES预处理: {len(smiles_list)} -> {len(cleaned_smiles)}")
        return cleaned_smiles
    
    def filter_by_properties(self, 
                           molecules: List[str], 
                           properties: Dict[str, List[float]],
                           filters: Dict[str, Tuple[float, float]]) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        根据分子性质过滤分子
        
        Args:
            molecules: 分子列表
            properties: 性质字典
            filters: 过滤条件 {property_name: (min_val, max_val)}
            
        Returns:
            (过滤后的分子列表, 过滤后的性质字典)
        """
        valid_indices = []
        
        for i, mol in enumerate(molecules):
            is_valid = True
            
            for prop_name, (min_val, max_val) in filters.items():
                if prop_name in properties and i < len(properties[prop_name]):
                    prop_val = properties[prop_name][i]
                    if not (min_val <= prop_val <= max_val):
                        is_valid = False
                        break
            
            if is_valid:
                valid_indices.append(i)
        
        # 过滤数据
        filtered_molecules = [molecules[i] for i in valid_indices]
        filtered_properties = {}
        
        for prop_name, prop_values in properties.items():
            filtered_properties[prop_name] = [prop_values[i] for i in valid_indices if i < len(prop_values)]
        
        logger.info(f"分子过滤: {len(molecules)} -> {len(filtered_molecules)}")
        return filtered_molecules, filtered_properties

class ProteinPreprocessor(DataPreprocessor):
    """蛋白质数据预处理器"""
    
    def __init__(self, config: NormalizationConfig):
        super().__init__(config)
        
    def preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """
        预处理蛋白质序列
        
        Args:
            sequences: 蛋白质序列列表
            
        Returns:
            清洗后的序列列表
        """
        cleaned_sequences = []
        
        for seq in sequences:
            # 转换为大写并移除空白字符
            cleaned = seq.upper().replace(' ', '').replace('\n', '').replace('\t', '')
            
            # 验证序列只包含标准氨基酸
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            if all(aa in valid_aa for aa in cleaned):
                cleaned_sequences.append(cleaned)
            else:
                logger.warning(f"跳过包含非标准氨基酸的序列: {seq[:50]}...")
        
        logger.info(f"蛋白质序列预处理: {len(sequences)} -> {len(cleaned_sequences)}")
        return cleaned_sequences
    
    def filter_by_length(self, 
                        sequences: List[str], 
                        min_length: int = 50, 
                        max_length: int = 1000) -> List[str]:
        """
        根据长度过滤蛋白质序列
        
        Args:
            sequences: 蛋白质序列列表
            min_length: 最小长度
            max_length: 最大长度
            
        Returns:
            过滤后的序列列表
        """
        filtered_sequences = []
        
        for seq in sequences:
            if min_length <= len(seq) <= max_length:
                filtered_sequences.append(seq)
        
        logger.info(f"蛋白质序列长度过滤: {len(sequences)} -> {len(filtered_sequences)}")
        return filtered_sequences
