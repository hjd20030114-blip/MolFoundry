# -*- coding: utf-8 -*-
"""
P-L 口袋-配体匹配深度分类模型（深度残差 MLP + 注意力池化）
输入维度：配体指纹(2048) + 口袋统计特征(12) = 2060
输出：logit（未过sigmoid） + 可选的结合能预测（多任务学习）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .base_model import BaseModel, ModelConfig, ModelRegistry

class ResidualMLPBlock(nn.Module):
    """残差 MLP 块（含 BatchNorm + Dropout）"""
    
    def __init__(self, dim: int, dropout: float = 0.2, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(dim * 2)
            self.bn2 = nn.BatchNorm1d(dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        
        # 残差连接
        return x + residual

class AttentionPooling(nn.Module):
    """注意力池化层（学习特征权重）"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, dim]
        Returns:
            [batch, dim] 加权特征
        """
        # 计算注意力分数
        attn_scores = self.attention_weights(x)  # [batch, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # 跨batch归一化
        
        # 加权求和
        weighted = x * attn_weights
        
        return weighted


@ModelRegistry.register("pl_pair_classifier")
class PLPairClassifier(BaseModel):
    """深度增强版 P-L 分类器（残差 MLP + 注意力 + 多任务）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # 默认输入维度（与 deep_learning/data/pl_pair_dataset.py 一致）
        self.input_dim = getattr(config, 'input_dim', 2060)
        self.hidden_dim = config.hidden_dim
        self.dropout_p = config.dropout
        self.num_layers = getattr(config, 'num_layers', 6)  # 增加到6层
        self.use_batch_norm = getattr(config, 'use_batch_norm', True)
        self.multi_task = getattr(config, 'multi_task', False)  # 是否启用多任务
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim) if self.use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(self.dropout_p)
        )
        
        # 深度残差 MLP 块（6层）
        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(
                dim=self.hidden_dim,
                dropout=self.dropout_p,
                use_batch_norm=self.use_batch_norm
            )
            for _ in range(self.num_layers)
        ])
        
        # 注意力池化
        self.attention_pooling = AttentionPooling(self.hidden_dim)
        
        # 分类头（主任务）
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2) if self.use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim // 2, 1)  # 输出logit
        )
        
        # 多任务头（预测结合能，辅助任务）
        if self.multi_task:
            self.affinity_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2) if self.use_batch_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.hidden_dim // 2, 1)  # 预测结合能
            )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 2060] 输入特征
            return_features: 是否返回中间特征
        Returns:
            {'logit': [B,1], 'affinity': [B,1] (可选), 'features': [B, hidden_dim] (可选)}
        """
        # 输入投影
        h = self.input_proj(x)  # [B, hidden_dim]
        
        # 通过残差块
        for block in self.residual_blocks:
            h = block(h)
        
        # 注意力池化（可选）
        h_pooled = self.attention_pooling(h)
        
        # 分类主任务
        logit = self.classifier_head(h_pooled)
        
        outputs = {'logit': logit}
        
        # 多任务：结合能预测
        if self.multi_task:
            affinity = self.affinity_head(h_pooled)
            outputs['affinity'] = affinity
        
        # 返回中间特征（用于可视化/分析）
        if return_features:
            outputs['features'] = h_pooled
        
        return outputs
