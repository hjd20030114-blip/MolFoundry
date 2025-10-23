#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据工具函数
包含数据集分割、平衡、增强等功能
"""

import numpy as np
import torch
import pickle
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter

logger = logging.getLogger(__name__)

def split_dataset(data: Union[List, np.ndarray, torch.Tensor],
                 labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 stratify: bool = False,
                 random_state: int = 42) -> Tuple:
    """
    分割数据集
    
    Args:
        data: 输入数据
        labels: 标签（可选）
        train_size: 训练集比例
        val_size: 验证集比例
        test_size: 测试集比例
        stratify: 是否分层采样
        random_state: 随机种子
        
    Returns:
        分割后的数据集
    """
    # 验证比例
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "比例之和必须为1"
    
    # 转换为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    data = np.array(data)
    if labels is not None:
        labels = np.array(labels)
    
    # 第一次分割：分离出测试集
    if test_size > 0:
        if stratify and labels is not None:
            train_val_data, test_data, train_val_labels, test_labels = train_test_split(
                data, labels, test_size=test_size, stratify=labels, random_state=random_state
            )
        else:
            if labels is not None:
                train_val_data, test_data, train_val_labels, test_labels = train_test_split(
                    data, labels, test_size=test_size, random_state=random_state
                )
            else:
                train_val_data, test_data = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
                train_val_labels, test_labels = None, None
    else:
        train_val_data, test_data = data, None
        train_val_labels, test_labels = labels, None
    
    # 第二次分割：分离训练集和验证集
    if val_size > 0:
        val_ratio = val_size / (train_size + val_size)
        
        if stratify and train_val_labels is not None:
            train_data, val_data, train_labels, val_labels = train_test_split(
                train_val_data, train_val_labels, test_size=val_ratio, 
                stratify=train_val_labels, random_state=random_state
            )
        else:
            if train_val_labels is not None:
                train_data, val_data, train_labels, val_labels = train_test_split(
                    train_val_data, train_val_labels, test_size=val_ratio, random_state=random_state
                )
            else:
                train_data, val_data = train_test_split(
                    train_val_data, test_size=val_ratio, random_state=random_state
                )
                train_labels, val_labels = None, None
    else:
        train_data, val_data = train_val_data, None
        train_labels, val_labels = train_val_labels, None
    
    # 构建返回结果
    result = []
    
    # 训练集
    if train_labels is not None:
        result.extend([train_data, train_labels])
    else:
        result.append(train_data)
    
    # 验证集
    if val_data is not None:
        if val_labels is not None:
            result.extend([val_data, val_labels])
        else:
            result.append(val_data)
    
    # 测试集
    if test_data is not None:
        if test_labels is not None:
            result.extend([test_data, test_labels])
        else:
            result.append(test_data)
    
    logger.info(f"数据集分割完成: 训练集{len(train_data)}, 验证集{len(val_data) if val_data is not None else 0}, 测试集{len(test_data) if test_data is not None else 0}")
    
    return tuple(result)

def balance_dataset(data: Union[List, np.ndarray],
                   labels: Union[List, np.ndarray],
                   method: str = 'oversample',
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    平衡数据集
    
    Args:
        data: 输入数据
        labels: 标签
        method: 平衡方法 ('oversample', 'undersample', 'smote')
        random_state: 随机种子
        
    Returns:
        平衡后的数据和标签
    """
    data = np.array(data)
    labels = np.array(labels)
    
    # 统计类别分布
    label_counts = Counter(labels)
    logger.info(f"原始类别分布: {label_counts}")
    
    if method == 'oversample':
        # 过采样：增加少数类样本
        max_count = max(label_counts.values())
        balanced_data = []
        balanced_labels = []
        
        for label in label_counts.keys():
            label_mask = labels == label
            label_data = data[label_mask]
            label_labels = labels[label_mask]
            
            current_count = len(label_data)
            if current_count < max_count:
                # 随机重复采样
                np.random.seed(random_state)
                indices = np.random.choice(current_count, max_count - current_count, replace=True)
                additional_data = label_data[indices]
                additional_labels = label_labels[indices]
                
                balanced_data.append(np.vstack([label_data, additional_data]))
                balanced_labels.append(np.hstack([label_labels, additional_labels]))
            else:
                balanced_data.append(label_data)
                balanced_labels.append(label_labels)
        
        balanced_data = np.vstack(balanced_data)
        balanced_labels = np.hstack(balanced_labels)
        
    elif method == 'undersample':
        # 欠采样：减少多数类样本
        min_count = min(label_counts.values())
        balanced_data = []
        balanced_labels = []
        
        for label in label_counts.keys():
            label_mask = labels == label
            label_data = data[label_mask]
            label_labels = labels[label_mask]
            
            if len(label_data) > min_count:
                # 随机选择样本
                np.random.seed(random_state)
                indices = np.random.choice(len(label_data), min_count, replace=False)
                balanced_data.append(label_data[indices])
                balanced_labels.append(label_labels[indices])
            else:
                balanced_data.append(label_data)
                balanced_labels.append(label_labels)
        
        balanced_data = np.vstack(balanced_data)
        balanced_labels = np.hstack(balanced_labels)
        
    elif method == 'smote':
        # SMOTE算法
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            balanced_data, balanced_labels = smote.fit_resample(data, labels)
        except ImportError:
            logger.warning("imblearn不可用，使用过采样方法")
            return balance_dataset(data, labels, method='oversample', random_state=random_state)
    
    else:
        raise ValueError(f"未知的平衡方法: {method}")
    
    # 打乱数据
    np.random.seed(random_state)
    indices = np.random.permutation(len(balanced_data))
    balanced_data = balanced_data[indices]
    balanced_labels = balanced_labels[indices]
    
    new_label_counts = Counter(balanced_labels)
    logger.info(f"平衡后类别分布: {new_label_counts}")
    
    return balanced_data, balanced_labels

def augment_data(data: Union[List, np.ndarray],
                labels: Optional[Union[List, np.ndarray]] = None,
                augmentation_factor: float = 2.0,
                noise_level: float = 0.1,
                random_state: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    数据增强
    
    Args:
        data: 输入数据
        labels: 标签（可选）
        augmentation_factor: 增强倍数
        noise_level: 噪声水平
        random_state: 随机种子
        
    Returns:
        增强后的数据和标签
    """
    data = np.array(data)
    if labels is not None:
        labels = np.array(labels)
    
    np.random.seed(random_state)
    
    original_size = len(data)
    target_size = int(original_size * augmentation_factor)
    additional_size = target_size - original_size
    
    if additional_size <= 0:
        return data, labels
    
    # 生成增强数据
    augmented_data = []
    augmented_labels = []
    
    for _ in range(additional_size):
        # 随机选择一个原始样本
        idx = np.random.randint(0, original_size)
        original_sample = data[idx]
        
        # 添加噪声
        noise = np.random.normal(0, noise_level, original_sample.shape)
        augmented_sample = original_sample + noise
        
        augmented_data.append(augmented_sample)
        
        if labels is not None:
            augmented_labels.append(labels[idx])
    
    # 合并原始数据和增强数据
    augmented_data = np.vstack([data, np.array(augmented_data)])
    
    if labels is not None:
        augmented_labels = np.hstack([labels, np.array(augmented_labels)])
    else:
        augmented_labels = None
    
    # 打乱数据
    indices = np.random.permutation(len(augmented_data))
    augmented_data = augmented_data[indices]
    
    if augmented_labels is not None:
        augmented_labels = augmented_labels[indices]
    
    logger.info(f"数据增强完成: {original_size} -> {len(augmented_data)}")
    
    return augmented_data, augmented_labels

def save_dataset(data: Dict[str, Any], 
                filepath: Union[str, Path],
                format: str = 'pickle') -> None:
    """
    保存数据集
    
    Args:
        data: 数据字典
        filepath: 文件路径
        format: 保存格式 ('pickle', 'json', 'npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        # 转换numpy数组为列表
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                json_data[key] = value.numpy().tolist()
            else:
                json_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    elif format == 'npz':
        # 只保存numpy数组
        np_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            elif isinstance(value, torch.Tensor):
                np_data[key] = value.numpy()
        
        np.savez(filepath, **np_data)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"数据集已保存到: {filepath}")

def load_dataset(filepath: Union[str, Path],
                format: Optional[str] = None) -> Dict[str, Any]:
    """
    加载数据集
    
    Args:
        filepath: 文件路径
        format: 文件格式（自动检测如果为None）
        
    Returns:
        数据字典
    """
    filepath = Path(filepath)
    
    if format is None:
        # 根据文件扩展名自动检测格式
        if filepath.suffix == '.pkl' or filepath.suffix == '.pickle':
            format = 'pickle'
        elif filepath.suffix == '.json':
            format = 'json'
        elif filepath.suffix == '.npz':
            format = 'npz'
        else:
            raise ValueError(f"无法识别文件格式: {filepath.suffix}")
    
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    elif format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 转换列表为numpy数组（如果适用）
        for key, value in data.items():
            if isinstance(value, list):
                try:
                    data[key] = np.array(value)
                except:
                    pass  # 保持原始格式
    elif format == 'npz':
        npz_data = np.load(filepath)
        data = {key: npz_data[key] for key in npz_data.files}
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"数据集已从 {filepath} 加载")
    return data

def create_cross_validation_splits(data: Union[List, np.ndarray],
                                  labels: Optional[Union[List, np.ndarray]] = None,
                                  n_splits: int = 5,
                                  stratify: bool = False,
                                  random_state: int = 42) -> List[Tuple]:
    """
    创建交叉验证分割
    
    Args:
        data: 输入数据
        labels: 标签（可选）
        n_splits: 分割数量
        stratify: 是否分层
        random_state: 随机种子
        
    Returns:
        分割索引列表
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    
    data = np.array(data)
    if labels is not None:
        labels = np.array(labels)
    
    if stratify and labels is not None:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kfold.split(data, labels))
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kfold.split(data))
    
    logger.info(f"创建了 {n_splits} 折交叉验证分割")
    return splits

def normalize_features(data: np.ndarray,
                      method: str = 'standard',
                      fit_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    特征标准化
    
    Args:
        data: 输入数据
        method: 标准化方法 ('standard', 'minmax', 'robust')
        fit_data: 用于拟合的数据（如果为None则使用data）
        
    Returns:
        (标准化后的数据, 标准化参数)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if fit_data is None:
        fit_data = data
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"未知的标准化方法: {method}")
    
    # 拟合并转换
    scaler.fit(fit_data)
    normalized_data = scaler.transform(data)
    
    # 保存标准化参数
    params = {
        'method': method,
        'scaler': scaler
    }
    
    return normalized_data, params
