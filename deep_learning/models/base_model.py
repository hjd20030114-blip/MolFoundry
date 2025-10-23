"""
基础模型类和配置
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置基类"""
    model_type: str
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    input_dim: int = 2060
    activation: str = "relu"
    use_batch_norm: bool = True
    use_residual: bool = True
    
    # SE(3) Equivariant GNN 特定参数
    irreps_hidden: str = "128x0e + 64x1o + 32x2e"
    max_radius: float = 5.0
    num_neighbors: int = 32
    
    # Diffusion Model 特定参数
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    
    # Transformer 特定参数
    num_heads: int = 8
    ff_dim: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)

class BaseModel(nn.Module):
    """基础模型类"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        
        # 通用组件
        self.dropout = nn.Dropout(config.dropout)
        
        # 激活函数
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, *args, **kwargs):
        """前向传播 - 子类需要实现"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'model_type': self.model_type
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> 'BaseModel':
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig.from_dict(checkpoint['config'])
        
        # 根据模型类型创建相应的模型
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model checkpoint loaded from {path}")
        return model, checkpoint.get('epoch', 0)
    
    def freeze_parameters(self, freeze: bool = True):
        """冻结/解冻模型参数"""
        for param in self.parameters():
            param.requires_grad = not freeze
    
    def get_device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.parameters()).device
    
    def count_parameters_by_layer(self) -> Dict[str, int]:
        """按层统计参数数量"""
        param_count = {}
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    param_count[name] = num_params
        return param_count

class ModelRegistry:
    """模型注册器"""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        """注册模型装饰器"""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str):
        """获取注册的模型"""
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered. Available models: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls):
        """列出所有注册的模型"""
        return list(cls._models.keys())

# 模型工厂函数
def create_model(model_type: str, config: ModelConfig) -> BaseModel:
    """创建模型的工厂函数"""
    try:
        model_class = ModelRegistry.get_model(model_type)
        return model_class(config)
    except ValueError as e:
        logger.error(f"Failed to create model: {e}")
        raise

# 模型初始化工具
def init_weights(module: nn.Module):
    """权重初始化"""
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)
