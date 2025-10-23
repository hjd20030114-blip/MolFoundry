"""
Pocket-Ligand Cross-attention Transformer
用于蛋白质口袋和配体之间的交叉注意力建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from .base_model import BaseModel, ModelConfig, ModelRegistry

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class CrossAttentionLayer(nn.Module):
    """增强交叉注意力层（Pre-LN + GELU + 门控FFN）"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_gated_ffn: bool = True
    ):
        super().__init__()
        self.use_gated_ffn = use_gated_ffn
        
        # Pre-LayerNorm（训练更稳定）
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 门控前馈网络（GLU变体，参数翻倍但性能更好）
        self.norm3 = nn.LayerNorm(d_model)
        if use_gated_ffn:
            # SwiGLU: x * swish(gate)
            self.ffn_gate = nn.Linear(d_model, d_ff)
            self.ffn_value = nn.Linear(d_model, d_ff)
            self.ffn_proj = nn.Linear(d_ff, d_model)
        else:
            # 标准FFN（GELU激活）
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model] 查询序列
            context: [batch_size, context_len, d_model] 上下文序列
            self_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
        """
        # Pre-LN 自注意力
        x_norm = self.norm1(x)
        attn_output, self_attn_weights = self.self_attention(x_norm, x_norm, x_norm, self_mask)
        x = x + self.dropout(attn_output)
        
        # Pre-LN 交叉注意力
        x_norm = self.norm2(x)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x_norm, context, context, cross_mask
        )
        x = x + self.dropout(cross_attn_output)
        
        # Pre-LN 前馈网络（门控或标准）
        x_norm = self.norm3(x)
        if self.use_gated_ffn:
            # SwiGLU: x * swish(gate)
            gate = F.silu(self.ffn_gate(x_norm))
            value = self.ffn_value(x_norm)
            ff_output = self.ffn_proj(gate * value)
        else:
            ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        
        return {
            'output': x,
            'self_attention_weights': self_attn_weights,
            'cross_attention_weights': cross_attn_weights
        }

@ModelRegistry.register("pocket_ligand_transformer")
class PocketLigandTransformer(BaseModel):
    """蛋白质口袋-配体交叉注意力Transformer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.d_model = config.hidden_dim
        self.num_heads = getattr(config, 'num_heads', 8)
        self.num_layers = config.num_layers
        # FFN维度扩大4倍（标准Transformer做法）
        self.d_ff = getattr(config, 'ff_dim', config.hidden_dim * 4)
        self.use_gated_ffn = getattr(config, 'use_gated_ffn', True)
        # 原始输入维度（与数据集对齐：配体2048位，口袋12维统计）
        self.ligand_input_dim = getattr(config, 'ligand_input_dim', 2048)
        self.pocket_input_dim = getattr(config, 'pocket_input_dim', 12)
        # 供 Trainer 读取以从batch['x']切分
        self.d_ligand_in = self.ligand_input_dim
        self.d_pocket_in = self.pocket_input_dim
        
        # 输入嵌入（从原始维度投影到 d_model）
        self.ligand_embedding = nn.Linear(self.ligand_input_dim, self.d_model)
        self.pocket_embedding = nn.Linear(self.pocket_input_dim, self.d_model)
        
        # 位置编码
        self.ligand_pos_encoding = PositionalEncoding(self.d_model)
        self.pocket_pos_encoding = PositionalEncoding(self.d_model)
        
        # 增强Transformer层（Pre-LN + 门控FFN）
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=config.dropout,
                use_gated_ffn=self.use_gated_ffn
            )
            for _ in range(self.num_layers)
        ])
        
        # 最终层归一化（Pre-LN架构需要）
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.d_model, 1)  # 结合亲和力预测
        )
        
        # 分子生成头
        self.generation_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, config.hidden_dim)  # 生成分子特征
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        ligand_features: Optional[torch.Tensor] = None,
        pocket_features: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
        pocket_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        x_input: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ligand_features: [batch_size, ligand_len, feature_dim]
            pocket_features: [batch_size, pocket_len, feature_dim]
            ligand_mask: [batch_size, ligand_len]
            pocket_mask: [batch_size, pocket_len]
        """
        # 若未显式提供两路特征，则从融合输入 x_input 自动切分
        if (ligand_features is None or pocket_features is None) and (x_input is not None):
            if x_input.dim() == 2:
                # [B, D]，按最后一维切分
                ligand_features = x_input[:, :self.d_ligand_in]
                pocket_features = x_input[:, self.d_ligand_in:self.d_ligand_in + self.d_pocket_in]
            elif x_input.dim() == 3:
                # [B, L, D]
                ligand_features = x_input[..., :self.d_ligand_in]
                pocket_features = x_input[..., self.d_ligand_in:self.d_ligand_in + self.d_pocket_in]
            else:
                raise ValueError(f"Unsupported x_input shape: {tuple(x_input.shape)}")

        if ligand_features is None or pocket_features is None:
            raise ValueError("PocketLigandTransformer.forward requires ligand_features & pocket_features, or a fused x_input to split.")

        # 兼容2D输入：[B, D] -> [B, 1, D]
        if ligand_features.dim() == 2:
            ligand_features = ligand_features.unsqueeze(1)
        if pocket_features.dim() == 2:
            pocket_features = pocket_features.unsqueeze(1)
        
        # 输入嵌入
        ligand_emb = self.ligand_embedding(ligand_features)
        pocket_emb = self.pocket_embedding(pocket_features)
        
        # 位置编码
        ligand_emb = ligand_emb.transpose(0, 1)  # [ligand_len, batch_size, d_model]
        pocket_emb = pocket_emb.transpose(0, 1)  # [pocket_len, batch_size, d_model]
        
        ligand_emb = self.ligand_pos_encoding(ligand_emb)
        pocket_emb = self.pocket_pos_encoding(pocket_emb)
        
        ligand_emb = ligand_emb.transpose(0, 1)  # [batch_size, ligand_len, d_model]
        pocket_emb = pocket_emb.transpose(0, 1)  # [batch_size, pocket_len, d_model]
        
        # Dropout
        ligand_emb = self.dropout(ligand_emb)
        pocket_emb = self.dropout(pocket_emb)
        
        # 通过Transformer层
        attention_weights = []
        x = ligand_emb
        
        for layer in self.layers:
            layer_output = layer(
                x=x,
                context=pocket_emb,
                self_mask=ligand_mask,
                cross_mask=pocket_mask
            )
            x = layer_output['output']
            
            if return_attention:
                attention_weights.append({
                    'self_attention': layer_output['self_attention_weights'],
                    'cross_attention': layer_output['cross_attention_weights']
                })
        
        # 全局池化
        if ligand_mask is not None:
            # 掩码池化
            mask_expanded = ligand_mask.unsqueeze(-1).expand_as(x)
            x_masked = x * mask_expanded.float()
            pooled = x_masked.sum(dim=1) / mask_expanded.float().sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # 预测结合亲和力
        binding_affinity = self.output_projection(pooled)
        
        # 生成分子特征
        generated_features = self.generation_head(x)
        
        outputs = {
            'binding_affinity': binding_affinity,
            'logit': binding_affinity,  # 兼容二分类训练（与数据集的 'y' 搭配）
            'ligand_features': x,
            'generated_features': generated_features,
            'pooled_features': pooled
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def generate_molecules(
        self,
        pocket_features: torch.Tensor,
        num_molecules: int = 1,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """基于口袋特征生成分子"""
        batch_size = pocket_features.size(0)
        device = pocket_features.device
        
        # 初始化生成序列
        generated = torch.zeros(
            batch_size, max_length, self.d_model, device=device
        )
        
        # 自回归生成
        for i in range(max_length):
            # 当前序列
            current_seq = generated[:, :i+1, :]
            
            # 前向传播
            outputs = self.forward(
                ligand_features=current_seq,
                pocket_features=pocket_features
            )
            
            # 获取下一个token的特征
            next_features = outputs['generated_features'][:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_features = next_features / temperature
            
            # 更新生成序列
            if i < max_length - 1:
                generated[:, i+1, :] = next_features
        
        return generated
    
    def compute_interaction_map(
        self,
        ligand_features: torch.Tensor,
        pocket_features: torch.Tensor
    ) -> torch.Tensor:
        """计算配体-口袋相互作用图"""
        outputs = self.forward(
            ligand_features=ligand_features,
            pocket_features=pocket_features,
            return_attention=True
        )
        
        # 提取交叉注意力权重
        cross_attention_weights = []
        for layer_attn in outputs['attention_weights']:
            cross_attention_weights.append(layer_attn['cross_attention'])
        
        # 平均所有层的注意力权重
        interaction_map = torch.stack(cross_attention_weights).mean(dim=0)
        
        return interaction_map
