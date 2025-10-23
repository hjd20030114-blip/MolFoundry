#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标
包含各种评估指标的计算和分析功能
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging

logger = logging.getLogger(__name__)

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基本回归指标
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # 相关系数
    try:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        metrics['pearson'] = pearson_corr
        metrics['pearson_p'] = pearson_p
    except:
        pass
    
    try:
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        metrics['spearman'] = spearman_corr
        metrics['spearman_p'] = spearman_p
    except:
        pass
    
    # 平均绝对百分比误差
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        metrics['mape'] = mape
    
    return metrics

def compute_classification_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None,
                                 average: str = 'binary') -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率
        average: 平均方式
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基本分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # 如果提供了概率，计算AUC相关指标
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # 二分类
                metrics['auc'] = roc_auc_score(y_true, y_prob)
                metrics['ap'] = average_precision_score(y_true, y_prob)
            else:  # 多分类
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
        except ValueError as e:
            logger.warning(f"无法计算AUC: {e}")
    
    return metrics

def compute_molecular_property_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算分子性质预测的特定指标
    
    Args:
        y_true: 真实分子性质
        y_pred: 预测分子性质
        
    Returns:
        指标字典
    """
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # 分子特定指标
    # 预测准确性（在一定误差范围内的预测比例）
    error_thresholds = [0.1, 0.2, 0.5, 1.0]
    for threshold in error_thresholds:
        accurate_predictions = np.abs(y_pred - y_true) <= threshold
        metrics[f'accuracy_{threshold}'] = np.mean(accurate_predictions)
    
    # 相对误差统计
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        relative_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        metrics['mean_relative_error'] = np.mean(relative_errors)
        metrics['median_relative_error'] = np.median(relative_errors)
        metrics['q75_relative_error'] = np.percentile(relative_errors, 75)
        metrics['q95_relative_error'] = np.percentile(relative_errors, 95)
    
    return metrics

def compute_binding_affinity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算结合亲和力预测的特定指标
    
    Args:
        y_true: 真实结合亲和力
        y_pred: 预测结合亲和力
        
    Returns:
        指标字典
    """
    metrics = compute_molecular_property_metrics(y_true, y_pred)
    
    # 排序相关性
    try:
        kendall_corr, kendall_p = kendalltau(y_true, y_pred)
        metrics['kendall'] = kendall_corr
        metrics['kendall_p'] = kendall_p
    except:
        pass
    
    # 分类性能（将连续值转换为高/低亲和力分类）
    median_affinity = np.median(y_true)
    y_true_binary = (y_true > median_affinity).astype(int)
    y_pred_binary = (y_pred > median_affinity).astype(int)
    
    classification_metrics = compute_classification_metrics(y_true_binary, y_pred_binary)
    for key, value in classification_metrics.items():
        metrics[f'binary_{key}'] = value
    
    # Top-k准确性
    for k in [1, 5, 10]:
        if len(y_true) >= k:
            # 获取真实值的top-k索引
            true_top_k = np.argsort(y_true)[-k:]
            # 获取预测值的top-k索引
            pred_top_k = np.argsort(y_pred)[-k:]
            # 计算重叠
            overlap = len(set(true_top_k) & set(pred_top_k))
            metrics[f'top_{k}_accuracy'] = overlap / k
    
    return metrics

def compute_generation_metrics(generated_smiles: List[str], 
                             reference_smiles: Optional[List[str]] = None) -> Dict[str, float]:
    """
    计算分子生成的指标
    
    Args:
        generated_smiles: 生成的SMILES列表
        reference_smiles: 参考SMILES列表
        
    Returns:
        生成指标字典
    """
    metrics = {}
    
    if not generated_smiles:
        return {'validity': 0.0, 'uniqueness': 0.0, 'novelty': 0.0}
    
    # 有效性
    valid_smiles = []
    try:
        from rdkit import Chem
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        metrics['validity'] = len(valid_smiles) / len(generated_smiles)
    except ImportError:
        logger.warning("RDKit不可用，无法计算分子有效性")
        valid_smiles = generated_smiles
        metrics['validity'] = 1.0
    
    # 唯一性
    unique_smiles = list(set(valid_smiles))
    metrics['uniqueness'] = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0
    
    # 新颖性
    if reference_smiles is not None:
        reference_set = set(reference_smiles)
        novel_count = sum(1 for smiles in unique_smiles if smiles not in reference_set)
        metrics['novelty'] = novel_count / len(unique_smiles) if unique_smiles else 0.0
    
    # 多样性（基于Tanimoto相似性）
    if len(unique_smiles) > 1:
        try:
            from rdkit import Chem
            from rdkit.Chem import DataStructs
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
            
            fps = []
            for smiles in unique_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            
            if len(fps) > 1:
                similarities = []
                for i in range(len(fps)):
                    for j in range(i + 1, len(fps)):
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        similarities.append(sim)
                
                metrics['diversity'] = 1.0 - np.mean(similarities)
            else:
                metrics['diversity'] = 0.0
                
        except ImportError:
            logger.warning("RDKit不可用，无法计算分子多样性")
    
    return metrics

def compute_generation_binding_energy_metrics(
    energies: List[float],
    thresholds: Optional[List[float]] = None,
    k_list: Optional[List[int]] = None,
    minimize: bool = True
) -> Dict[str, float]:
    """计算生成分子的结合能分布指标（通常越小越好，即能量更负更优）。

    Args:
        energies: 结合能/打分列表（单位自定；若为对接能量，通常为负值，越小越好）
        thresholds: 阈值列表，用于计算比例（默认 [-6.0, -7.0, -8.0]）
        k_list: Top-k 列表（默认 [1, 5, 10]），用于计算前k个最佳能量的统计
        minimize: 是否为“越小越好”（True：排序升序；False：排序降序）

    Returns:
        包含最佳能量、均值/中位数、分位数、Top-k 统计与阈值达成率的字典
    """
    metrics: Dict[str, float] = {}
    if thresholds is None:
        thresholds = [-6.0, -7.0, -8.0]
    if k_list is None:
        k_list = [1, 5, 10]

    if energies is None or len(energies) == 0:
        return {'count': 0}

    arr = np.asarray(energies, dtype=float)
    valid_mask = np.isfinite(arr)
    valid = arr[valid_mask]

    metrics['count'] = int(arr.size)
    metrics['count_valid'] = int(valid.size)
    if valid.size == 0:
        return metrics

    # 排序：若为越小越好（如结合能越负越优），升序；否则降序
    order = np.argsort(valid)
    if not minimize:
        order = order[::-1]
    sorted_vals = valid[order]

    # 基本统计
    metrics['best_energy'] = float(sorted_vals[0])
    metrics['mean_energy'] = float(np.mean(valid))
    metrics['median_energy'] = float(np.median(valid))
    # 分位数
    for q in [5, 10, 25, 50, 75, 90, 95]:
        try:
            metrics[f'quantile_{q}'] = float(np.percentile(valid, q))
        except Exception:
            pass

    # Top-k 统计
    n = sorted_vals.size
    for k in k_list:
        if k <= 0:
            continue
        kk = min(k, n)
        topk = sorted_vals[:kk]
        metrics[f'top{kk}_mean_energy'] = float(np.mean(topk))
        metrics[f'top{kk}_best_energy'] = float(topk[0])
        metrics[f'top{kk}_cutoff_energy'] = float(topk[-1])  # 前k名中的最差者

    # 阈值达成率（若 minimize=True，则统计 <= 阈值 的比例；否则统计 >= 阈值）
    for thr in thresholds:
        if minimize:
            rate = float(np.mean(valid <= thr))
        else:
            rate = float(np.mean(valid >= thr))
        # 将阈值中的小数点与负号规整为可读键名
        thr_key = str(thr).replace('.', 'p').replace('-', 'neg')
        metrics[f'rate_better_than_{thr_key}'] = rate

    return metrics

def compute_drug_likeness_metrics(smiles_list: List[str]) -> Dict[str, float]:
    """
    计算类药性指标
    
    Args:
        smiles_list: SMILES列表
        
    Returns:
        类药性指标字典
    """
    metrics = {}
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        valid_mols = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
        
        if not valid_mols:
            return metrics
        
        # Lipinski规则
        lipinski_violations = []
        qed_scores = []
        
        for mol in valid_mols:
            # 分子量
            mw = Descriptors.MolWt(mol)
            # LogP
            logp = Descriptors.MolLogP(mol)
            # 氢键供体
            hbd = Descriptors.NumHDonors(mol)
            # 氢键受体
            hba = Descriptors.NumHAcceptors(mol)
            
            # Lipinski违规计数
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            lipinski_violations.append(violations)
            
            # QED分数（如果可用）
            try:
                from rdkit.Chem import QED
                qed = QED.qed(mol)
                qed_scores.append(qed)
            except:
                pass
        
        # 统计指标
        metrics['lipinski_pass_rate'] = np.mean([v == 0 for v in lipinski_violations])
        metrics['mean_lipinski_violations'] = np.mean(lipinski_violations)
        
        if qed_scores:
            metrics['mean_qed'] = np.mean(qed_scores)
            metrics['median_qed'] = np.median(qed_scores)
        
    except ImportError:
        logger.warning("RDKit不可用，无法计算类药性指标")
    
    return metrics

class EvaluationSuite:
    """评估套件"""
    
    def __init__(self):
        self.results = {}
        
    def add_regression_evaluation(self, name: str, y_true: np.ndarray, y_pred: np.ndarray):
        """添加回归评估"""
        self.results[name] = compute_regression_metrics(y_true, y_pred)
        
    def add_classification_evaluation(self, name: str, y_true: np.ndarray, 
                                    y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None):
        """添加分类评估"""
        self.results[name] = compute_classification_metrics(y_true, y_pred, y_prob)
        
    def add_generation_evaluation(self, name: str, generated_smiles: List[str],
                                reference_smiles: Optional[List[str]] = None):
        """添加生成评估"""
        self.results[name] = compute_generation_metrics(generated_smiles, reference_smiles)
    
    def add_generation_binding_energy(self,
                                      name: str,
                                      energies: List[float],
                                      thresholds: Optional[List[float]] = None,
                                      k_list: Optional[List[int]] = None,
                                      minimize: bool = True):
        """添加生成分子结合能分布评估（默认越小越好）。"""
        self.results[name] = compute_generation_binding_energy_metrics(
            energies=energies,
            thresholds=thresholds,
            k_list=k_list,
            minimize=minimize
        )
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取评估摘要"""
        return self.results
        
    def print_summary(self):
        """打印评估摘要"""
        for name, metrics in self.results.items():
            print(f"\n{name}:")
            print("-" * 40)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

# 别名，保持向后兼容
MolecularEvaluator = EvaluationSuite
