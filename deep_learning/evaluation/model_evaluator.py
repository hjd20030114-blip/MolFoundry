#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估器
包含模型性能评估和分析功能
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device = torch.device('cpu')):
        """
        初始化模型评估器
        
        Args:
            model: 要评估的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
    def evaluate_regression(self, 
                          data_loader,
                          criterion: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """
        评估回归模型
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 获取输入和目标
                if isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('x'))
                    targets = batch.get('target', batch.get('y'))
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                # 收集预测和目标
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                num_batches += 1
        
        # 合并所有预测和目标
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        from ..training.metrics import compute_molecular_metrics
        metrics = compute_molecular_metrics(predictions, targets)
        
        if criterion is not None:
            metrics['loss'] = total_loss / num_batches
            
        return metrics
    
    def evaluate_classification(self, 
                              data_loader,
                              criterion: Optional[torch.nn.Module] = None,
                              num_classes: Optional[int] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        评估分类模型
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数
            num_classes: 类别数量
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 获取输入和目标
                if isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('x'))
                    targets = batch.get('target', batch.get('y'))
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                # 获取预测概率和类别
                probabilities = torch.softmax(outputs, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)
                
                # 收集结果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                num_batches += 1
        
        # 合并所有结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)
        
        # 计算指标
        from ..training.metrics import MetricsCalculator
        calculator = MetricsCalculator()
        calculator.predictions = [predictions]
        calculator.targets = [targets]
        
        metrics = calculator.compute_classification_metrics()
        
        if criterion is not None:
            metrics['loss'] = total_loss / num_batches
        
        # 添加混淆矩阵
        if num_classes:
            cm = confusion_matrix(targets, predictions, labels=range(num_classes))
            metrics['confusion_matrix'] = cm
        
        return metrics
    
    def evaluate_binding_affinity(self, 
                                 data_loader,
                                 criterion: Optional[torch.nn.Module] = None) -> Dict[str, float]:
        """
        评估结合亲和力预测模型
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # 处理复杂的输入格式（蛋白质-配体对）
                if isinstance(batch, dict):
                    # 假设输入包含蛋白质和配体特征
                    protein_features = batch.get('protein_features')
                    ligand_features = batch.get('ligand_features')
                    targets = batch.get('binding_affinity')
                    
                    if protein_features is not None and ligand_features is not None:
                        protein_features = protein_features.to(self.device)
                        ligand_features = ligand_features.to(self.device)
                        inputs = (protein_features, ligand_features)
                    else:
                        inputs = batch.get('input', batch.get('x')).to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                
                targets = targets.to(self.device)
                
                # 前向传播
                if isinstance(inputs, tuple):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                # 计算损失
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                # 收集预测和目标
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                num_batches += 1
        
        # 合并所有预测和目标
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算结合亲和力特定指标
        from ..training.metrics import compute_binding_affinity_metrics
        metrics = compute_binding_affinity_metrics(predictions, targets)
        
        if criterion is not None:
            metrics['loss'] = total_loss / num_batches
            
        return metrics
    
    def cross_validate(self, 
                      dataset,
                      k_folds: int = 5,
                      task_type: str = 'regression',
                      criterion: Optional[torch.nn.Module] = None) -> Dict[str, List[float]]:
        """
        执行k折交叉验证
        
        Args:
            dataset: 数据集
            k_folds: 折数
            task_type: 任务类型
            criterion: 损失函数
            
        Returns:
            交叉验证结果
        """
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"交叉验证折 {fold + 1}/{k_folds}")
            
            # 创建验证数据加载器
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # 评估当前折
            if task_type == 'regression':
                metrics = self.evaluate_regression(val_loader, criterion)
            elif task_type == 'classification':
                metrics = self.evaluate_classification(val_loader, criterion)
            elif task_type == 'binding_affinity':
                metrics = self.evaluate_binding_affinity(val_loader, criterion)
            else:
                raise ValueError(f"未知的任务类型: {task_type}")
            
            fold_metrics.append(metrics)
        
        # 计算平均指标和标准差
        for metric_name in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][metric_name], (int, float)):
                values = [fold[metric_name] for fold in fold_metrics]
                cv_results[f'{metric_name}_mean'] = np.mean(values)
                cv_results[f'{metric_name}_std'] = np.std(values)
                cv_results[f'{metric_name}_folds'] = values
        
        return cv_results
    
    def generate_evaluation_report(self, 
                                 metrics: Dict[str, float],
                                 save_path: Optional[str] = None) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        report_lines = [
            "=" * 50,
            "模型评估报告",
            "=" * 50,
            ""
        ]
        
        # 基本指标
        if 'loss' in metrics:
            report_lines.append(f"损失: {metrics['loss']:.4f}")
        
        if 'mse' in metrics:
            report_lines.append(f"均方误差 (MSE): {metrics['mse']:.4f}")
        
        if 'rmse' in metrics:
            report_lines.append(f"均方根误差 (RMSE): {metrics['rmse']:.4f}")
        
        if 'mae' in metrics:
            report_lines.append(f"平均绝对误差 (MAE): {metrics['mae']:.4f}")
        
        if 'r2' in metrics:
            report_lines.append(f"R² 分数: {metrics['r2']:.4f}")
        
        if 'pearson' in metrics:
            report_lines.append(f"皮尔逊相关系数: {metrics['pearson']:.4f}")
        
        # 分类指标
        if 'accuracy' in metrics:
            report_lines.append(f"准确率: {metrics['accuracy']:.4f}")
        
        if 'precision' in metrics:
            report_lines.append(f"精确率: {metrics['precision']:.4f}")
        
        if 'recall' in metrics:
            report_lines.append(f"召回率: {metrics['recall']:.4f}")
        
        if 'f1' in metrics:
            report_lines.append(f"F1分数: {metrics['f1']:.4f}")
        
        if 'auc' in metrics:
            report_lines.append(f"AUC: {metrics['auc']:.4f}")
        
        # 结合亲和力特定指标
        if 'spearman' in metrics:
            report_lines.append(f"Spearman相关系数: {metrics['spearman']:.4f}")
        
        if 'kendall' in metrics:
            report_lines.append(f"Kendall相关系数: {metrics['kendall']:.4f}")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"评估报告已保存到: {save_path}")
        
        return report_text
    
    def plot_predictions(self, 
                        predictions: np.ndarray,
                        targets: np.ndarray,
                        save_path: Optional[str] = None,
                        title: str = "预测 vs 真实值") -> None:
        """
        绘制预测值与真实值的对比图
        
        Args:
            predictions: 预测值
            targets: 真实值
            save_path: 保存路径
            title: 图标题
        """
        plt.figure(figsize=(8, 6))
        
        # 散点图
        plt.scatter(targets, predictions, alpha=0.6)
        
        # 对角线（完美预测线）
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加R²分数
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测图已保存到: {save_path}")
        
        plt.show()
