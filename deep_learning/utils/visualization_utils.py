#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具函数
包含训练曲线、分子性质等的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def plot_training_curves(history: Dict[str, List[float]], 
                        save_path: Optional[str] = None,
                        title: str = "训练曲线") -> None:
    """
    绘制训练曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        title: 图标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # 损失曲线
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='训练损失')
        axes[0, 0].plot(history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线（如果存在）
    if 'train_acc' in history and 'val_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='训练准确率')
        axes[0, 1].plot(history['val_acc'], label='验证准确率')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # R²分数曲线（如果存在）
    if 'train_r2' in history and 'val_r2' in history:
        axes[1, 0].plot(history['train_r2'], label='训练R²')
        axes[1, 0].plot(history['val_r2'], label='验证R²')
        axes[1, 0].set_title('R²分数曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线（如果存在）
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], label='学习率')
        axes[1, 1].set_title('学习率曲线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练曲线已保存到: {save_path}")
    
    plt.show()

def plot_molecular_properties(properties: Dict[str, List[float]],
                            save_path: Optional[str] = None,
                            title: str = "分子性质分布") -> None:
    """
    绘制分子性质分布
    
    Args:
        properties: 分子性质字典
        save_path: 保存路径
        title: 图标题
    """
    n_properties = len(properties)
    if n_properties == 0:
        return
    
    # 计算子图布局
    n_cols = min(3, n_properties)
    n_rows = (n_properties + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(title, fontsize=16)
    
    if n_properties == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (prop_name, values) in enumerate(properties.items()):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row, col]
        
        # 绘制直方图
        ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{prop_name}分布')
        ax.set_xlabel(prop_name)
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        ax.legend()
    
    # 隐藏多余的子图
    for i in range(n_properties, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row, col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"分子性质图已保存到: {save_path}")
    
    plt.show()

def visualize_attention_weights(attention_weights: np.ndarray,
                              labels: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              title: str = "注意力权重") -> None:
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重矩阵
        labels: 标签列表
        save_path: 保存路径
        title: 图标题
    """
    plt.figure(figsize=(10, 8))
    
    # 使用热图显示注意力权重
    sns.heatmap(attention_weights, 
                annot=True if attention_weights.shape[0] <= 20 else False,
                cmap='Blues',
                xticklabels=labels if labels else False,
                yticklabels=labels if labels else False,
                cbar_kws={'label': '注意力权重'})
    
    plt.title(title)
    plt.xlabel('键 (Key)')
    plt.ylabel('查询 (Query)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"注意力权重图已保存到: {save_path}")
    
    plt.show()

def create_molecular_plot(smiles: str, 
                         properties: Optional[Dict[str, float]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    创建分子结构图
    
    Args:
        smiles: SMILES字符串
        properties: 分子性质
        save_path: 保存路径
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"无效的SMILES: {smiles}")
            return
        
        # 生成分子图像
        img = Draw.MolToImage(mol, size=(400, 400))
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示分子结构
        ax1.imshow(img)
        ax1.set_title(f'分子结构\nSMILES: {smiles}')
        ax1.axis('off')
        
        # 显示性质
        if properties:
            prop_names = list(properties.keys())
            prop_values = list(properties.values())
            
            ax2.barh(prop_names, prop_values)
            ax2.set_title('分子性质')
            ax2.set_xlabel('数值')
            
            # 添加数值标签
            for i, v in enumerate(prop_values):
                ax2.text(v, i, f' {v:.2f}', va='center')
        else:
            ax2.text(0.5, 0.5, '无性质数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('分子性质')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分子图已保存到: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("RDKit不可用，无法绘制分子结构")
    except Exception as e:
        logger.warning(f"分子图创建失败: {e}")

def plot_correlation_matrix(data: np.ndarray,
                          labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          title: str = "相关性矩阵") -> None:
    """
    绘制相关性矩阵
    
    Args:
        data: 数据矩阵
        labels: 特征标签
        save_path: 保存路径
        title: 图标题
    """
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(data.T)
    
    plt.figure(figsize=(10, 8))
    
    # 使用热图显示相关性
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                xticklabels=labels if labels else False,
                yticklabels=labels if labels else False,
                cbar_kws={'label': '相关系数'})
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"相关性矩阵已保存到: {save_path}")
    
    plt.show()

def plot_prediction_scatter(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: Optional[str] = None,
                          title: str = "预测 vs 真实值") -> None:
    """
    绘制预测值与真实值的散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        title: 图标题
    """
    plt.figure(figsize=(8, 6))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 对角线（完美预测线）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
    
    # 计算R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{title}\nR² = {r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'样本数: {len(y_true)}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测散点图已保存到: {save_path}")
    
    plt.show()

def plot_loss_landscape(loss_values: np.ndarray,
                       param1_range: np.ndarray,
                       param2_range: np.ndarray,
                       param1_name: str = "参数1",
                       param2_name: str = "参数2",
                       save_path: Optional[str] = None) -> None:
    """
    绘制损失函数地形图
    
    Args:
        loss_values: 损失值矩阵
        param1_range: 第一个参数的范围
        param2_range: 第二个参数的范围
        param1_name: 第一个参数名称
        param2_name: 第二个参数名称
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 2D等高线图
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(param1_range, param2_range, loss_values, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title('损失函数等高线图')
    
    # 3D表面图
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    surf = ax2.plot_surface(X, Y, loss_values, cmap='viridis', alpha=0.8)
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_zlabel('损失值')
    ax2.set_title('损失函数3D图')
    
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"损失地形图已保存到: {save_path}")
    
    plt.show()
