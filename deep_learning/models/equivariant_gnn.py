"""
SE(3)-Equivariant Graph Neural Network
基于e3nn实现的等变图神经网络，用于蛋白质-配体相互作用建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Dict, Any
import math

try:
    from e3nn import o3
    from e3nn.nn import BatchNorm
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False
    print("Warning: e3nn not available. SE(3)-Equivariant features will be limited.")

from .base_model import BaseModel, ModelConfig, ModelRegistry

class RadialBasisFunction(nn.Module):
    """径向基函数编码距离信息"""
    
    def __init__(self, num_radial: int = 50, cutoff: float = 5.0):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # 高斯基函数参数
        self.register_buffer('centers', torch.linspace(0, cutoff, num_radial))
        self.register_buffer('widths', torch.ones(num_radial) * (cutoff / num_radial))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [num_edges] 边的距离
        Returns:
            [num_edges, num_radial] 径向基函数编码
        """
        # 截断函数
        cutoff_mask = distances <= self.cutoff

        # 高斯基函数
        distances = distances.unsqueeze(-1)  # [num_edges, 1]
        rbf = torch.exp(-((distances - self.centers) / self.widths) ** 2)
        
        # 应用截断
        rbf = rbf * cutoff_mask.unsqueeze(-1).float()
        
        return rbf

class SE3TransformerLayer(MessagePassing):
    """SE(3)-等变Transformer层"""
    
    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        irreps_edge_attr: str,
        num_heads: int = 4,
        fc_neurons: list = [64, 64],
        use_attention: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.irreps_in = o3.Irreps(irreps_in) if HAS_E3NN else None
        self.irreps_out = o3.Irreps(irreps_out) if HAS_E3NN else None
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr) if HAS_E3NN else None
        self.num_heads = num_heads
        self.use_attention = use_attention
        
        if not HAS_E3NN:
            # 简化版本，不使用e3nn
            self.linear = nn.Linear(64, 64)  # 假设特征维度
            return
        
        # 自注意力机制
        if use_attention:
            self.attention = o3.Linear(
                irreps_in=self.irreps_in,
                irreps_out=f"{num_heads}x0e",
                internal_weights=True,
                shared_weights=True
            )
        
        # 消息传递网络
        self.message_net = o3.Linear(
            irreps_in=self.irreps_in + self.irreps_edge_attr,
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True
        )
        
        # 更新网络
        self.update_net = o3.Linear(
            irreps_in=self.irreps_in + self.irreps_out,
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True
        )
        
        # 层归一化
        self.norm = BatchNorm(self.irreps_out)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, irreps_in] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, irreps_edge_attr] 边特征
            batch: [num_nodes] 批次索引
        """
        if not HAS_E3NN:
            # 简化版本
            return self.linear(x)
        
        # 消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 残差连接和归一化
        if self.irreps_in == self.irreps_out:
            out = out + x
        
        out = self.norm(out)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """消息函数"""
        if not HAS_E3NN:
            return x_j
        
        # 拼接节点特征和边特征
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        
        # 通过消息网络
        message = self.message_net(message_input)
        
        return message
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """更新函数"""
        if not HAS_E3NN:
            return aggr_out
        
        # 拼接原始特征和聚合消息
        update_input = torch.cat([x, aggr_out], dim=-1)
        
        # 通过更新网络
        updated = self.update_net(update_input)
        
        return updated

@ModelRegistry.register("equivariant_gnn")
class EquivariantGNN(BaseModel):
    """SE(3)-等变图神经网络"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.max_radius = config.max_radius
        self.num_neighbors = config.num_neighbors
        
        # 原子类型嵌入
        self.atom_embedding = nn.Embedding(100, config.hidden_dim)  # 支持100种原子类型
        
        # 径向基函数
        self.rbf = RadialBasisFunction(num_radial=50, cutoff=config.max_radius)
        
        # 边特征处理
        self.edge_embedding = nn.Linear(50, config.hidden_dim)  # RBF输出维度
        
        if HAS_E3NN:
            # SE(3)-等变层
            irreps_hidden = config.irreps_hidden
            irreps_edge = f"{config.hidden_dim}x0e"
            
            self.layers = nn.ModuleList([
                SE3TransformerLayer(
                    irreps_in=irreps_hidden if i > 0 else f"{config.hidden_dim}x0e",
                    irreps_out=irreps_hidden,
                    irreps_edge_attr=irreps_edge,
                    num_heads=4
                )
                for i in range(config.num_layers)
            ])
        else:
            # 简化版本的GNN层
            self.layers = nn.ModuleList([
                nn.Linear(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_layers)
            ])
        
        # 节点/图特征维度：e3nn下为irreps的总维度；否则为hidden_dim
        if HAS_E3NN:
            try:
                self.feature_dim = self.layers[-1].irreps_out.dim  # type: ignore[attr-defined]
            except Exception:
                # 回退：使用hidden_dim，尽量避免维度不一致
                self.feature_dim = config.hidden_dim
        else:
            self.feature_dim = config.hidden_dim
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_dim),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)  # 结合亲和力预测
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            atom_types: [num_atoms] 原子类型
            positions: [num_atoms, 3] 原子坐标
            edge_index: [2, num_edges] 边索引
            batch: [num_atoms] 批次索引
        
        Returns:
            Dict包含预测结果和中间特征
        """
        # 原子特征嵌入
        x = self.atom_embedding(atom_types)  # [num_atoms, hidden_dim]
        
        # 计算边特征
        edge_attr = self._compute_edge_features(positions, edge_index)
        
        # 通过SE(3)-等变层
        for layer in self.layers:
            if HAS_E3NN:
                x = layer(x, edge_index, edge_attr, batch)
            else:
                x = F.relu(layer(x))
                x = self.dropout(x)
        
        # 图级别池化
        if batch is not None:
            graph_features = global_mean_pool(x, batch)
        else:
            graph_features = x.mean(dim=0, keepdim=True)
        
        # 预测结合亲和力
        binding_affinity = self.output_projection(graph_features)
        
        return {
            'binding_affinity': binding_affinity,
            'node_features': x,
            'graph_features': graph_features
        }
    
    def _compute_edge_features(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """计算边特征"""
        # 计算边向量和距离
        row, col = edge_index
        edge_vec = positions[row] - positions[col]  # [num_edges, 3]
        edge_dist = torch.norm(edge_vec, dim=-1)  # [num_edges]
        
        # 径向基函数编码
        rbf_features = self.rbf(edge_dist)  # [num_edges, num_radial]
        
        # 边特征嵌入
        edge_attr = self.edge_embedding(rbf_features)  # [num_edges, hidden_dim]
        
        return edge_attr
    
    def predict_binding_affinity(
        self,
        protein_data: Dict[str, torch.Tensor],
        ligand_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """预测蛋白质-配体结合亲和力"""
        # 合并蛋白质和配体数据
        combined_data = self._combine_protein_ligand(protein_data, ligand_data)
        
        # 前向传播
        outputs = self.forward(**combined_data)
        
        return outputs['binding_affinity']
    
    def _combine_protein_ligand(
        self,
        protein_data: Dict[str, torch.Tensor],
        ligand_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """合并蛋白质和配体数据"""
        # 简化实现，实际应该更复杂
        num_protein_atoms = protein_data['atom_types'].size(0)
        num_ligand_atoms = ligand_data['atom_types'].size(0)
        
        # 合并原子类型和坐标
        atom_types = torch.cat([protein_data['atom_types'], ligand_data['atom_types']])
        positions = torch.cat([protein_data['positions'], ligand_data['positions']])
        
        # 合并边索引（需要调整配体的索引）
        ligand_edge_index = ligand_data['edge_index'] + num_protein_atoms
        edge_index = torch.cat([protein_data['edge_index'], ligand_edge_index], dim=1)
        
        return {
            'atom_types': atom_types,
            'positions': positions,
            'edge_index': edge_index
        }
