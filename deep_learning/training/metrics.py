#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练指标计算
包含各种评估指标的计算函数
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置累积的指标"""
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: Optional[float] = None):
        """
        更新指标
        
        Args:
            predictions: 预测值
            targets: 真实值
            loss: 损失值
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_batch_metrics(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算单个batch的简单指标（与Trainer接口兼容）"""
        metrics: Dict[str, float] = {}
        try:
            if 'binding_affinity' in outputs and 'binding_affinity' in targets:
                y_pred = outputs['binding_affinity'].detach().cpu().numpy().reshape(-1)
                y_true = targets['binding_affinity'].detach().cpu().numpy().reshape(-1)
                # 简单MSE/MAE
                mse = float(((y_pred - y_true) ** 2).mean())
                mae = float(np.abs(y_pred - y_true).mean())
                metrics['mse'] = mse
                metrics['mae'] = mae
        except Exception:
            pass
        return metrics
    
    def compute_epoch_metrics(self, all_outputs: List[Dict[str, torch.Tensor]], all_targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """整合一个epoch的输出/标签，返回回归指标（供Trainer验证阶段使用）"""
        try:
            if not all_outputs or not all_targets:
                return {}
            y_pred_list = []
            y_true_list = []
            for out, tgt in zip(all_outputs, all_targets):
                if 'binding_affinity' not in out or 'binding_affinity' not in tgt:
                    continue
                y_pred_list.append(out['binding_affinity'].detach().cpu().numpy().reshape(-1))
                y_true_list.append(tgt['binding_affinity'].detach().cpu().numpy().reshape(-1))
            if not y_pred_list:
                return {}
            y_pred = np.concatenate(y_pred_list, axis=0)
            y_true = np.concatenate(y_true_list, axis=0)
            # 复用回归指标
            self.predictions = [y_pred]
            self.targets = [y_true]
            return self.compute_regression_metrics()
        except Exception:
            return {}
            
    def compute_regression_metrics(self) -> Dict[str, float]:
        """计算回归指标"""
        if not self.predictions:
            return {}
            
        # 合并所有预测和目标
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        # 确保是1D数组
        if all_predictions.ndim > 1:
            all_predictions = all_predictions.flatten()
        if all_targets.ndim > 1:
            all_targets = all_targets.flatten()
            
        metrics = {}
        
        try:
            # 均方误差
            metrics['mse'] = mean_squared_error(all_targets, all_predictions)
            
            # 均方根误差
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # 平均绝对误差
            metrics['mae'] = mean_absolute_error(all_targets, all_predictions)
            
            # R²分数
            metrics['r2'] = r2_score(all_targets, all_predictions)
            
            # 平均绝对百分比误差
            non_zero_mask = all_targets != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((all_targets[non_zero_mask] - all_predictions[non_zero_mask]) / all_targets[non_zero_mask])) * 100
                metrics['mape'] = mape
                
            # 皮尔逊相关系数
            correlation = np.corrcoef(all_targets, all_predictions)[0, 1]
            if not np.isnan(correlation):
                metrics['pearson'] = correlation
                
        except Exception as e:
            logger.warning(f"计算回归指标时出错: {e}")
            
        return metrics
        
    def compute_classification_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """计算分类指标"""
        if not self.predictions:
            return {}
            
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        metrics = {}
        
        try:
            # 二分类指标
            if all_predictions.ndim == 1 or all_predictions.shape[1] == 1:
                # 二分类
                if all_predictions.ndim > 1:
                    all_predictions = all_predictions.flatten()
                if all_targets.ndim > 1:
                    all_targets = all_targets.flatten()
                    
                # 预测类别
                pred_classes = (all_predictions > threshold).astype(int)
                
                # 准确率
                metrics['accuracy'] = accuracy_score(all_targets, pred_classes)
                
                # 精确率、召回率、F1分数
                metrics['precision'] = precision_score(all_targets, pred_classes, zero_division=0)
                metrics['recall'] = recall_score(all_targets, pred_classes, zero_division=0)
                metrics['f1'] = f1_score(all_targets, pred_classes, zero_division=0)
                
                # AUC
                try:
                    metrics['auc'] = roc_auc_score(all_targets, all_predictions)
                    metrics['ap'] = average_precision_score(all_targets, all_predictions)
                except ValueError:
                    # 如果只有一个类别，AUC无法计算
                    pass
                    
            else:
                # 多分类
                pred_classes = np.argmax(all_predictions, axis=1)
                true_classes = np.argmax(all_targets, axis=1) if all_targets.ndim > 1 else all_targets
                
                # 准确率
                metrics['accuracy'] = accuracy_score(true_classes, pred_classes)
                
                # 宏平均指标
                metrics['precision_macro'] = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
                
                # 微平均指标
                metrics['precision_micro'] = precision_score(true_classes, pred_classes, average='micro', zero_division=0)
                metrics['recall_micro'] = recall_score(true_classes, pred_classes, average='micro', zero_division=0)
                metrics['f1_micro'] = f1_score(true_classes, pred_classes, average='micro', zero_division=0)
                
        except Exception as e:
            logger.warning(f"计算分类指标时出错: {e}")
            
        return metrics
        
    def compute_loss_metrics(self) -> Dict[str, float]:
        """计算损失指标"""
        if not self.losses:
            return {}
            
        losses = np.array(self.losses)
        
        return {
            'loss': np.mean(losses),
            'loss_std': np.std(losses),
            'loss_min': np.min(losses),
            'loss_max': np.max(losses)
        }
        
    def compute_all_metrics(self, task_type: str = 'regression', threshold: float = 0.5) -> Dict[str, float]:
        """
        计算所有相关指标
        
        Args:
            task_type: 任务类型 ('regression' 或 'classification')
            threshold: 分类阈值
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 损失指标
        metrics.update(self.compute_loss_metrics())
        
        # 任务特定指标
        if task_type == 'regression':
            metrics.update(self.compute_regression_metrics())
        elif task_type == 'classification':
            metrics.update(self.compute_classification_metrics(threshold))
            
        return metrics

def compute_molecular_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    计算分子性质预测的特定指标
    
    Args:
        predictions: 预测的分子性质
        targets: 真实的分子性质
        
    Returns:
        分子指标字典
    """
    metrics = {}
    
    # 基本回归指标
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(targets, predictions)
    metrics['r2'] = r2_score(targets, predictions)
    
    # 分子特定指标
    # 预测准确性（在一定误差范围内的预测比例）
    error_thresholds = [0.1, 0.2, 0.5, 1.0]
    for threshold in error_thresholds:
        accurate_predictions = np.abs(predictions - targets) <= threshold
        metrics[f'accuracy_{threshold}'] = np.mean(accurate_predictions)
    
    # 相对误差
    non_zero_mask = targets != 0
    if np.any(non_zero_mask):
        relative_errors = np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])
        metrics['mean_relative_error'] = np.mean(relative_errors)
        metrics['median_relative_error'] = np.median(relative_errors)
    
    return metrics

def compute_binding_affinity_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    计算结合亲和力预测的特定指标
    
    Args:
        predictions: 预测的结合亲和力
        targets: 真实的结合亲和力
        
    Returns:
        结合亲和力指标字典
    """
    metrics = {}
    
    # 基本回归指标
    metrics.update(compute_molecular_metrics(predictions, targets))
    
    # 结合亲和力特定指标
    # 排序相关性（Spearman相关系数）
    from scipy.stats import spearmanr, kendalltau
    
    try:
        spearman_corr, spearman_p = spearmanr(targets, predictions)
        metrics['spearman'] = spearman_corr
        metrics['spearman_p'] = spearman_p
    except:
        pass
    
    try:
        kendall_corr, kendall_p = kendalltau(targets, predictions)
        metrics['kendall'] = kendall_corr
        metrics['kendall_p'] = kendall_p
    except:
        pass
    
    # 分类性能（将连续值转换为高/低亲和力分类）
    median_affinity = np.median(targets)
    target_classes = (targets > median_affinity).astype(int)
    pred_classes = (predictions > median_affinity).astype(int)
    
    metrics['classification_accuracy'] = accuracy_score(target_classes, pred_classes)
    
    return metrics

def compute_generation_metrics(generated_molecules: List[str], 
                             reference_molecules: List[str]) -> Dict[str, float]:
    """
    计算分子生成的指标
    
    Args:
        generated_molecules: 生成的分子SMILES列表
        reference_molecules: 参考分子SMILES列表
        
    Returns:
        生成指标字典
    """
    metrics = {}
    
    # 有效性（生成的分子中有效的比例）
    try:
        from rdkit import Chem
        valid_count = 0
        for smiles in generated_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
        metrics['validity'] = valid_count / len(generated_molecules) if generated_molecules else 0
    except ImportError:
        logger.warning("RDKit不可用，无法计算分子有效性")
    
    # 唯一性（生成的分子中唯一的比例）
    unique_molecules = set(generated_molecules)
    metrics['uniqueness'] = len(unique_molecules) / len(generated_molecules) if generated_molecules else 0
    
    # 新颖性（生成的分子中不在参考集合中的比例）
    reference_set = set(reference_molecules)
    novel_count = sum(1 for mol in unique_molecules if mol not in reference_set)
    metrics['novelty'] = novel_count / len(unique_molecules) if unique_molecules else 0
    
    return metrics

class ValidationMetrics:
    """验证指标管理器"""

    def __init__(self):
        self.metrics_history = []
        self.best_metrics = {}
        self.best_epoch = 0

    def update(self, epoch: int, metrics: Dict[str, float]):
        """更新验证指标"""
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history.append(metrics_with_epoch)

        # 更新最佳指标（基于验证损失）
        if 'val_loss' in metrics:
            if not self.best_metrics or metrics['val_loss'] < self.best_metrics.get('val_loss', float('inf')):
                self.best_metrics = metrics_with_epoch.copy()
                self.best_epoch = epoch

    def get_best_metrics(self) -> Dict[str, float]:
        """获取最佳指标"""
        return self.best_metrics

    def get_history(self) -> List[Dict[str, float]]:
        """获取指标历史"""
        return self.metrics_history

    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新指标"""
        return self.metrics_history[-1] if self.metrics_history else {}

    def has_improved(self, patience: int = 10) -> bool:
        """检查指标是否在指定patience内有改善"""
        if len(self.metrics_history) < patience:
            return True

        current_epoch = self.metrics_history[-1]['epoch']
        return (current_epoch - self.best_epoch) < patience
