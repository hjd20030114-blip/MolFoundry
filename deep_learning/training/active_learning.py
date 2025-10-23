#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主动学习模块
包含主动学习循环和不确定性采样策略
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ActiveLearningConfig:
    """主动学习配置"""
    initial_pool_size: int = 1000
    query_size: int = 100
    max_iterations: int = 10
    uncertainty_threshold: float = 0.1
    diversity_weight: float = 0.3
    acquisition_function: str = 'uncertainty'  # 'uncertainty', 'diversity', 'hybrid'

class UncertaintySampler:
    """不确定性采样器"""
    
    def __init__(self, method: str = 'entropy'):
        """
        初始化不确定性采样器
        
        Args:
            method: 不确定性计算方法 ('entropy', 'variance', 'margin')
        """
        self.method = method
        
    def compute_uncertainty(self, predictions: torch.Tensor, 
                          model_outputs: Optional[Dict] = None) -> torch.Tensor:
        """
        计算预测的不确定性
        
        Args:
            predictions: 模型预测 [N, ...]
            model_outputs: 模型额外输出（如方差）
            
        Returns:
            不确定性分数 [N]
        """
        if self.method == 'entropy':
            return self._compute_entropy(predictions)
        elif self.method == 'variance':
            return self._compute_variance(predictions, model_outputs)
        elif self.method == 'margin':
            return self._compute_margin(predictions)
        else:
            raise ValueError(f"未知的不确定性方法: {self.method}")
    
    def _compute_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算熵不确定性"""
        if predictions.dim() == 1:
            # 回归任务，使用预测方差作为不确定性
            return torch.var(predictions, dim=0, keepdim=True).expand(predictions.shape[0])
        else:
            # 分类任务，计算预测概率的熵
            probs = torch.softmax(predictions, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return entropy
    
    def _compute_variance(self, predictions: torch.Tensor, 
                         model_outputs: Optional[Dict] = None) -> torch.Tensor:
        """计算方差不确定性"""
        if model_outputs and 'variance' in model_outputs:
            return model_outputs['variance'].squeeze()
        else:
            # 如果没有显式方差，使用预测的标准差
            if predictions.dim() > 1:
                return torch.std(predictions, dim=-1)
            else:
                return torch.ones_like(predictions) * 0.1  # 默认不确定性
    
    def _compute_margin(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算边际不确定性（分类任务）"""
        if predictions.dim() == 1:
            return torch.ones_like(predictions) * 0.1  # 回归任务默认值
        else:
            probs = torch.softmax(predictions, dim=-1)
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            return 1.0 - margin  # 边际越小，不确定性越大
    
    def select_samples(self, uncertainties: torch.Tensor, 
                      query_size: int) -> torch.Tensor:
        """
        根据不确定性选择样本
        
        Args:
            uncertainties: 不确定性分数 [N]
            query_size: 查询样本数量
            
        Returns:
            选中样本的索引 [query_size]
        """
        _, indices = torch.topk(uncertainties, k=min(query_size, len(uncertainties)))
        return indices

class DiversitySampler:
    """多样性采样器"""
    
    def __init__(self, method: str = 'kmeans'):
        """
        初始化多样性采样器
        
        Args:
            method: 多样性采样方法 ('kmeans', 'farthest_point')
        """
        self.method = method
        
    def select_diverse_samples(self, features: torch.Tensor, 
                             query_size: int,
                             selected_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        选择多样性样本
        
        Args:
            features: 样本特征 [N, D]
            query_size: 查询样本数量
            selected_indices: 已选择的样本索引
            
        Returns:
            选中样本的索引 [query_size]
        """
        if self.method == 'kmeans':
            return self._kmeans_sampling(features, query_size, selected_indices)
        elif self.method == 'farthest_point':
            return self._farthest_point_sampling(features, query_size, selected_indices)
        else:
            raise ValueError(f"未知的多样性采样方法: {self.method}")
    
    def _kmeans_sampling(self, features: torch.Tensor, 
                        query_size: int,
                        selected_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """K-means聚类采样"""
        try:
            from sklearn.cluster import KMeans
            
            # 转换为numpy
            features_np = features.detach().cpu().numpy()
            
            # 如果有已选择的样本，排除它们
            if selected_indices is not None:
                mask = torch.ones(len(features), dtype=torch.bool)
                mask[selected_indices] = False
                available_features = features_np[mask.cpu().numpy()]
                available_indices = torch.arange(len(features))[mask]
            else:
                available_features = features_np
                available_indices = torch.arange(len(features))
            
            if len(available_features) <= query_size:
                return available_indices
            
            # K-means聚类
            kmeans = KMeans(n_clusters=query_size, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(available_features)
            
            # 从每个聚类中选择最接近中心的样本
            selected = []
            for i in range(query_size):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    cluster_features = available_features[cluster_mask]
                    cluster_indices = available_indices[cluster_mask]
                    
                    # 计算到聚类中心的距离
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    closest_idx = np.argmin(distances)
                    selected.append(cluster_indices[closest_idx].item())
            
            return torch.tensor(selected)
            
        except ImportError:
            logger.warning("scikit-learn不可用，使用随机采样")
            return self._random_sampling(features, query_size, selected_indices)
    
    def _farthest_point_sampling(self, features: torch.Tensor, 
                                query_size: int,
                                selected_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """最远点采样"""
        device = features.device
        n_samples = len(features)
        
        # 如果有已选择的样本，排除它们
        if selected_indices is not None:
            mask = torch.ones(n_samples, dtype=torch.bool, device=device)
            mask[selected_indices] = False
            available_indices = torch.arange(n_samples, device=device)[mask]
            available_features = features[mask]
        else:
            available_indices = torch.arange(n_samples, device=device)
            available_features = features
        
        if len(available_features) <= query_size:
            return available_indices
        
        selected = []
        remaining_indices = available_indices.clone()
        remaining_features = available_features.clone()
        
        # 随机选择第一个点
        first_idx = torch.randint(0, len(remaining_features), (1,), device=device)
        selected.append(remaining_indices[first_idx].item())
        
        # 移除已选择的点
        mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=device)
        mask[first_idx] = False
        remaining_indices = remaining_indices[mask]
        remaining_features = remaining_features[mask]
        
        # 迭代选择最远的点
        for _ in range(query_size - 1):
            if len(remaining_features) == 0:
                break
                
            # 计算到所有已选择点的最小距离
            selected_features = features[torch.tensor(selected, device=device)]
            distances = torch.cdist(remaining_features, selected_features)
            min_distances = torch.min(distances, dim=1)[0]
            
            # 选择距离最远的点
            farthest_idx = torch.argmax(min_distances)
            selected.append(remaining_indices[farthest_idx].item())
            
            # 移除已选择的点
            mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=device)
            mask[farthest_idx] = False
            remaining_indices = remaining_indices[mask]
            remaining_features = remaining_features[mask]
        
        return torch.tensor(selected, device=device)
    
    def _random_sampling(self, features: torch.Tensor, 
                        query_size: int,
                        selected_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """随机采样（备用方法）"""
        device = features.device
        n_samples = len(features)
        
        if selected_indices is not None:
            mask = torch.ones(n_samples, dtype=torch.bool, device=device)
            mask[selected_indices] = False
            available_indices = torch.arange(n_samples, device=device)[mask]
        else:
            available_indices = torch.arange(n_samples, device=device)
        
        if len(available_indices) <= query_size:
            return available_indices
        
        # 随机选择
        perm = torch.randperm(len(available_indices), device=device)
        return available_indices[perm[:query_size]]

class ActiveLearningLoop:
    """主动学习循环"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 config: ActiveLearningConfig,
                 uncertainty_sampler: Optional[UncertaintySampler] = None,
                 diversity_sampler: Optional[DiversitySampler] = None):
        """
        初始化主动学习循环
        
        Args:
            model: 要训练的模型
            config: 主动学习配置
            uncertainty_sampler: 不确定性采样器
            diversity_sampler: 多样性采样器
        """
        self.model = model
        self.config = config
        self.uncertainty_sampler = uncertainty_sampler or UncertaintySampler()
        self.diversity_sampler = diversity_sampler or DiversitySampler()
        
        self.labeled_indices = set()
        self.iteration = 0
        
    def run(self, 
            unlabeled_data: torch.Tensor,
            unlabeled_features: torch.Tensor,
            oracle_function: Callable[[torch.Tensor], torch.Tensor],
            trainer: Optional[object] = None) -> Dict[str, List]:
        """
        运行主动学习循环
        
        Args:
            unlabeled_data: 未标记数据
            unlabeled_features: 未标记数据特征
            oracle_function: 标注函数（模拟专家标注）
            trainer: 训练器对象
            
        Returns:
            主动学习历史记录
        """
        history = {
            'iterations': [],
            'labeled_sizes': [],
            'uncertainties': [],
            'performance': []
        }
        
        logger.info("开始主动学习循环")
        
        # 初始化标记池
        if not self.labeled_indices:
            initial_indices = self._initialize_labeled_pool(unlabeled_features)
            self.labeled_indices.update(initial_indices.tolist())
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            logger.info(f"主动学习迭代 {iteration + 1}/{self.config.max_iterations}")
            
            # 获取当前标记和未标记数据
            labeled_indices = torch.tensor(list(self.labeled_indices))
            unlabeled_indices = self._get_unlabeled_indices(len(unlabeled_data))
            
            if len(unlabeled_indices) == 0:
                logger.info("没有更多未标记数据，停止主动学习")
                break
            
            # 训练模型（如果提供了训练器）
            if trainer is not None:
                labeled_data = unlabeled_data[labeled_indices]
                labeled_targets = oracle_function(labeled_indices)
                trainer.train_epoch(labeled_data, labeled_targets)
            
            # 选择查询样本
            query_indices = self._select_query_samples(
                unlabeled_data[unlabeled_indices],
                unlabeled_features[unlabeled_indices],
                unlabeled_indices
            )
            
            # 获取标注
            new_labels = oracle_function(query_indices)
            
            # 更新标记池
            self.labeled_indices.update(query_indices.tolist())
            
            # 记录历史
            history['iterations'].append(iteration)
            history['labeled_sizes'].append(len(self.labeled_indices))
            
            # 计算性能（如果可能）
            if trainer is not None:
                performance = self._evaluate_performance(trainer, unlabeled_data, oracle_function)
                history['performance'].append(performance)
            
            logger.info(f"迭代 {iteration + 1} 完成，标记样本数: {len(self.labeled_indices)}")
        
        logger.info("主动学习循环完成")
        return history
    
    def _initialize_labeled_pool(self, features: torch.Tensor) -> torch.Tensor:
        """初始化标记池"""
        if len(features) <= self.config.initial_pool_size:
            return torch.arange(len(features))
        
        # 使用多样性采样初始化
        return self.diversity_sampler.select_diverse_samples(
            features, self.config.initial_pool_size
        )
    
    def _get_unlabeled_indices(self, total_size: int) -> torch.Tensor:
        """获取未标记样本索引"""
        all_indices = set(range(total_size))
        unlabeled_indices = all_indices - self.labeled_indices
        return torch.tensor(list(unlabeled_indices))
    
    def _select_query_samples(self, 
                             unlabeled_data: torch.Tensor,
                             unlabeled_features: torch.Tensor,
                             unlabeled_indices: torch.Tensor) -> torch.Tensor:
        """选择查询样本"""
        if self.config.acquisition_function == 'uncertainty':
            return self._uncertainty_based_selection(unlabeled_data, unlabeled_indices)
        elif self.config.acquisition_function == 'diversity':
            return self._diversity_based_selection(unlabeled_features, unlabeled_indices)
        elif self.config.acquisition_function == 'hybrid':
            return self._hybrid_selection(unlabeled_data, unlabeled_features, unlabeled_indices)
        else:
            raise ValueError(f"未知的获取函数: {self.config.acquisition_function}")
    
    def _uncertainty_based_selection(self, 
                                   unlabeled_data: torch.Tensor,
                                   unlabeled_indices: torch.Tensor) -> torch.Tensor:
        """基于不确定性的选择"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(unlabeled_data)
            uncertainties = self.uncertainty_sampler.compute_uncertainty(predictions)
        
        selected_local = self.uncertainty_sampler.select_samples(uncertainties, self.config.query_size)
        return unlabeled_indices[selected_local]
    
    def _diversity_based_selection(self, 
                                 unlabeled_features: torch.Tensor,
                                 unlabeled_indices: torch.Tensor) -> torch.Tensor:
        """基于多样性的选择"""
        selected_local = self.diversity_sampler.select_diverse_samples(
            unlabeled_features, self.config.query_size
        )
        return unlabeled_indices[selected_local]
    
    def _hybrid_selection(self, 
                         unlabeled_data: torch.Tensor,
                         unlabeled_features: torch.Tensor,
                         unlabeled_indices: torch.Tensor) -> torch.Tensor:
        """混合选择策略"""
        # 分配查询预算
        uncertainty_budget = int(self.config.query_size * (1 - self.config.diversity_weight))
        diversity_budget = self.config.query_size - uncertainty_budget
        
        selected_indices = []
        
        # 不确定性选择
        if uncertainty_budget > 0:
            uncertainty_selected = self._uncertainty_based_selection(unlabeled_data, unlabeled_indices)
            selected_indices.extend(uncertainty_selected[:uncertainty_budget].tolist())
        
        # 多样性选择（排除已选择的样本）
        if diversity_budget > 0:
            remaining_mask = torch.ones(len(unlabeled_indices), dtype=torch.bool)
            if selected_indices:
                selected_local = torch.tensor([
                    (unlabeled_indices == idx).nonzero(as_tuple=True)[0].item() 
                    for idx in selected_indices
                ])
                remaining_mask[selected_local] = False
            
            remaining_features = unlabeled_features[remaining_mask]
            remaining_indices = unlabeled_indices[remaining_mask]
            
            if len(remaining_features) > 0:
                diversity_selected = self.diversity_sampler.select_diverse_samples(
                    remaining_features, diversity_budget
                )
                selected_indices.extend(remaining_indices[diversity_selected].tolist())
        
        return torch.tensor(selected_indices)
    
    def _evaluate_performance(self, trainer, data: torch.Tensor, oracle_function: Callable) -> Dict:
        """评估当前模型性能"""
        # 这里应该实现性能评估逻辑
        # 返回性能指标字典
        return {'accuracy': 0.0, 'loss': 0.0}
