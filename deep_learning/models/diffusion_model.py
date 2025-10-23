"""
Pocket-conditioned Diffusion Model
基于扩散模型的分子生成，以蛋白口袋为条件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from .base_model import BaseModel, ModelConfig, ModelRegistry

class AdaLNZero(nn.Module):
    """自适应层归一化（AdaLN-Zero）- DiT风格时间条件注入"""
    
    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        # 生成 scale, shift, gate 三个参数
        self.ada_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 3)
        )
        # zero初始化gate，确保训练初期接近恒等映射
        nn.init.zeros_(self.ada_linear[1].weight)
        nn.init.zeros_(self.ada_linear[1].bias)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            time_emb: [batch, time_emb_dim]
        """
        # LayerNorm
        x_norm = self.layer_norm(x)
        
        # 从时间嵌入生成 scale, shift, gate
        ada_params = self.ada_linear(time_emb).unsqueeze(1)  # [batch, 1, dim*3]
        scale, shift, gate = ada_params.chunk(3, dim=-1)
        
        # 自适应调制
        x_modulated = x_norm * (1 + scale) + shift
        
        # 门控残差连接（zero初始化，训练初期接近x）
        return x + gate * x_modulated

class GatedCrossAttention(nn.Module):
    """门控交叉注意力 - 选择性融合口袋特征"""
    
    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # 门控网络（学习融合权重）
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, dim] 分子特征
            key_value: [batch, kv_len, dim] 口袋特征
        """
        # 交叉注意力
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value
        )
        
        # 学习门控权重（基于原始特征和注意力输出）
        gate_input = torch.cat([query, attn_out], dim=-1)
        gate = self.gate_net(gate_input)
        
        # 门控融合
        gated_out = query + gate * attn_out
        return self.layer_norm(gated_out)

class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块（替代部分残差块，增强全局建模）"""
    
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0, dropout: float = 0.1, time_emb_dim: int = 2048):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, time_emb_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.adaln2 = AdaLNZero(dim, time_emb_dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            time_emb: [batch, time_emb_dim]
        """
        # 自注意力分支（含AdaLN）
        x_norm1 = self.adaln1(x, time_emb)
        attn_out, _ = self.self_attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_out
        
        # MLP分支（含AdaLN）
        x_norm2 = self.adaln2(x, time_emb)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        
        return x

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络（多尺度特征融合）"""
    
    def __init__(self, dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # 自顶向下路径（上采样+融合）
        self.lateral_convs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU()
            ) for _ in range(num_scales)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of [batch, seq_len, dim] 多尺度特征（从深到浅）
        Returns:
            融合后的多尺度特征
        """
        assert len(features) == self.num_scales
        
        # 自顶向下融合
        results = []
        prev = None
        for i in range(self.num_scales):
            lateral = self.lateral_convs[i](features[i])
            if prev is not None:
                lateral = lateral + prev
            output = self.output_convs[i](lateral)
            results.append(output)
            prev = lateral
        
        return results

class DiffusionScheduler:
    """扩散过程调度器"""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # 生成beta序列
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 计算相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于逆向过程的参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦beta调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """向原始数据添加噪声"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # 广播到正确的形状
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (
            sqrt_alphas_cumprod_t * x_start +
            sqrt_one_minus_alphas_cumprod_t * noise
        )
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码用于时间步嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """增强残差块（双层+中间特征扩展+AdaLN）"""
    
    def __init__(self, dim: int, time_emb_dim: int, dropout: float = 0.1, expand_ratio: float = 2.0, use_adaln: bool = True):
        super().__init__()
        self.use_adaln = use_adaln
        
        if use_adaln:
            self.adaln = AdaLNZero(dim, time_emb_dim)
        else:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, dim),
                nn.SiLU()
            )
        
        expanded_dim = int(dim * expand_ratio)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(32, dim), dim),
            nn.SiLU(),
            nn.Linear(dim, expanded_dim)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(32, expanded_dim), expanded_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, dim)
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # 时间条件（AdaLN或MLP）
        if self.use_adaln:
            x_cond = self.adaln(x, time_emb)
            h = self.block1(x_cond)
        else:
            h = self.block1(x)
            time_emb_proj = self.time_mlp(time_emb)
            h = h + time_emb_proj
        
        h = self.block2(h)
        
        return x + h

class PocketConditioningModule(nn.Module):
    """增强蛋白口袋条件模块（更深编码+双重注意力）"""
    
    def __init__(self, pocket_dim: int, hidden_dim: int, num_heads: int = 16):
        super().__init__()
        # 更深的口袋编码器（5层MLP）
        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 自注意力（分子内部）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 门控交叉注意力（分子→口袋，选择性融合）
        self.gated_cross_attention = GatedCrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        molecule_features: torch.Tensor,
        pocket_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            molecule_features: [batch, seq_len, hidden_dim]
            pocket_features: [batch, pocket_len, pocket_dim]
        """
        # 编码口袋特征
        pocket_encoded = self.pocket_encoder(pocket_features)  # [batch, pocket_len, hidden_dim]
        
        # 自注意力（分子内部交互）
        self_attn_out, _ = self.self_attention(
            query=molecule_features,
            key=molecule_features,
            value=molecule_features
        )
        molecule_features = self.layer_norm1(molecule_features + self_attn_out)
        
        # 门控交叉注意力（分子→口袋，选择性融合）
        conditioned_features = self.gated_cross_attention(
            query=molecule_features,
            key_value=pocket_encoded
        )
        
        return conditioned_features

@ModelRegistry.register("pocket_diffusion")
class PocketConditionedDiffusion(BaseModel):
    """基于口袋条件的扩散模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.hidden_dim = config.hidden_dim
        self.num_timesteps = config.num_timesteps
        
        # 扩散调度器
        self.scheduler = DiffusionScheduler(
            num_timesteps=config.num_timesteps,
            beta_schedule=config.beta_schedule
        )
        
        # 时间嵌入
        time_emb_dim = config.hidden_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 分子特征嵌入
        self.molecule_embedding = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 口袋条件模块（增强版，16头注意力）
        self.pocket_conditioning = PocketConditioningModule(
            pocket_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_heads=16
        )
        
        # 混合架构去噪网络（Transformer + ResBlock + FPN）
        num_layers = config.num_layers  # 默认6层
        num_down = num_layers // 2
        num_up = num_layers // 2
        
        # 下采样路径（混合ResBlock和Transformer）
        self.down_blocks = nn.ModuleList()
        for i in range(num_down):
            # 前半部分用ResBlock，后半部分用Transformer（全局建模）
            if i < num_down // 2:
                self.down_blocks.append(
                    ResidualBlock(config.hidden_dim, time_emb_dim, config.dropout, expand_ratio=2.0, use_adaln=True)
                )
            else:
                self.down_blocks.append(
                    TransformerEncoderBlock(config.hidden_dim, num_heads=16, mlp_ratio=4.0, dropout=config.dropout, time_emb_dim=time_emb_dim)
                )
        
        # 中间层（最深层，2个Transformer块）
        self.mid_blocks = nn.ModuleList([
            TransformerEncoderBlock(config.hidden_dim, num_heads=16, mlp_ratio=4.0, dropout=config.dropout, time_emb_dim=time_emb_dim)
            for _ in range(2)
        ])
        
        # 上采样路径（跳跃连接）
        self.up_blocks = nn.ModuleList()
        for i in range(num_up):
            # 前半部分用Transformer，后半部分用ResBlock
            if i < num_up // 2:
                self.up_blocks.append(
                    TransformerEncoderBlock(config.hidden_dim * 2, num_heads=16, mlp_ratio=4.0, dropout=config.dropout, time_emb_dim=time_emb_dim)
                )
            else:
                self.up_blocks.append(
                    ResidualBlock(config.hidden_dim * 2, time_emb_dim, config.dropout, expand_ratio=2.0, use_adaln=True)
                )
        
        # 跳跃连接投影层
        self.skip_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            for _ in range(num_up)
        ])
        
        # 特征金字塔网络（多尺度融合）
        self.fpn = FeaturePyramidNetwork(config.hidden_dim, num_scales=num_down)
        
        # 输出层
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
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
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        pocket_features: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_t: [batch, seq_len, hidden_dim] 噪声分子特征
            timesteps: [batch] 时间步
            pocket_features: [batch, pocket_len, hidden_dim] 口袋特征
        """
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)  # [batch, time_emb_dim]
        
        # 分子特征嵌入
        h = self.molecule_embedding(x_t)  # [batch, seq_len, hidden_dim]
        
        # 口袋条件
        h = self.pocket_conditioning(h, pocket_features)
        
        # 混合架构前向传播：下采样+FPN+跳跃连接
        skip_connections = []
        
        # 下采样（收集多尺度特征）
        for down_block in self.down_blocks:
            h = down_block(h, time_emb)
            skip_connections.append(h)
        
        # 特征金字塔融合（增强多尺度特征）
        skip_connections = self.fpn(skip_connections)
        
        # 中间层（最深）
        for mid_block in self.mid_blocks:
            h = mid_block(h, time_emb)
        
        # 上采样（融合增强后的跳跃连接）
        for up_block, skip_proj, skip_h in zip(self.up_blocks, self.skip_projections, reversed(skip_connections)):
            h = torch.cat([h, skip_h], dim=-1)  # 拼接跳跃连接
            h = skip_proj(h)  # 降维回 hidden_dim
            h = up_block(h, time_emb)
        
        # 输出预测的噪声
        noise_pred = self.output_projection(h)
        
        if return_dict:
            return {
                'noise_pred': noise_pred,
                'features': h
            }
        return noise_pred
    
    def training_step(
        self,
        x_0: torch.Tensor,
        pocket_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """训练步骤"""
        batch_size = x_0.size(0)
        device = x_0.device
        
        # 随机采样时间步
        timesteps = self.scheduler.sample_timesteps(batch_size, device)
        
        # 生成噪声
        noise = torch.randn_like(x_0)
        
        # 添加噪声
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)
        
        # 预测噪声
        outputs = self.forward(x_t, timesteps, pocket_features)
        noise_pred = outputs['noise_pred']
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise_target': noise
        }
    
    @torch.no_grad()
    def sample(
        self,
        pocket_features: torch.Tensor,
        num_samples: int = 1,
        sample_steps: Optional[int] = None
    ) -> torch.Tensor:
        """从噪声生成分子"""
        device = pocket_features.device
        batch_size = pocket_features.size(0)
        seq_len = 50  # 假设分子序列长度
        
        if sample_steps is None:
            sample_steps = self.num_timesteps
        
        # 从纯噪声开始
        x = torch.randn(batch_size, seq_len, self.hidden_dim, device=device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, sample_steps, dtype=torch.long, device=device
        )
        
        for t in timesteps:
            t_batch = t.repeat(batch_size)
            
            # 预测噪声
            outputs = self.forward(x, t_batch, pocket_features)
            noise_pred = outputs['noise_pred']
            
            # 去噪步骤
            x = self._denoise_step(x, noise_pred, t)
        
        return x
    
    def _denoise_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """单步去噪"""
        alpha_t = self.scheduler.alphas[t]
        alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
        beta_t = self.scheduler.betas[t]
        
        # 预测x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        if t > 0:
            # 添加噪声（除了最后一步）
            noise = torch.randn_like(x_t)
            posterior_variance_t = self.scheduler.posterior_variance[t]
            x_t_minus_1 = (
                torch.sqrt(alpha_t) * x_0_pred +
                torch.sqrt(1 - alpha_t - posterior_variance_t) * noise_pred +
                torch.sqrt(posterior_variance_t) * noise
            )
        else:
            x_t_minus_1 = x_0_pred
        
        return x_t_minus_1
