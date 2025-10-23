"""
深度学习训练器
支持多任务学习、对比学习、强化学习等训练策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import json
import inspect

from ..models.base_model import BaseModel
from .callbacks import Callback, CallbackList
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    num_epochs: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 优化器设置
    optimizer_type: str = "adam"  # adam, adamw, sgd
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    eta_min: float = 0.0
    
    # 训练策略
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # 多任务学习
    task_weights: Dict[str, float] = None
    adaptive_task_weights: bool = True
    
    # 对比学习
    use_contrastive_learning: bool = False
    contrastive_temperature: float = 0.1
    
    # 验证和保存
    validation_frequency: int = 1
    save_frequency: int = 10
    early_stopping_patience: int = 20
    use_early_stopping: bool = True
    
    # 设备和并行
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'binding_affinity': 1.0,
                'interaction_fingerprint': 0.5,
                'admet': 0.3,
                'synthesizability': 0.2,
                'drug_likeness': 0.2
            }

class Trainer:
    """深度学习训练器"""
    
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设备设置
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 混合精度（PyTorch>=2.6 推荐API）。仅在CUDA可用时启用。
        try:
            from torch.amp import GradScaler as _GradScaler
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.scaler = _GradScaler(device_type) if (config.mixed_precision and device_type == "cuda") else None
        except Exception:
            # 回退旧API，且仍仅在CUDA可用时启用
            self.scaler = torch.cuda.amp.GradScaler() if (config.mixed_precision and torch.cuda.is_available()) else None
        
        # 回调函数
        self.callbacks = CallbackList(callbacks or [])
        # 将训练器引用传递给所有回调（用于访问model/optimizer等）
        self.callbacks.set_trainer(self)
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config.scheduler_type.lower() == "cosine":
            if getattr(self.config, 'warmup_epochs', 0) and self.config.warmup_epochs > 0:
                try:
                    # 线性warmup从较小比例逐步增至1.0
                    warmup = optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.1,
                        end_factor=1.0,
                        total_iters=self.config.warmup_epochs
                    )
                    cosine = optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=max(1, self.config.num_epochs - self.config.warmup_epochs),
                        eta_min=self.config.eta_min
                    )
                    scheduler = optim.lr_scheduler.SequentialLR(
                        self.optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[self.config.warmup_epochs]
                    )
                    return scheduler
                except Exception:
                    # 回退：若当前PyTorch版本无LinearLR/SequentialLR，则使用纯cosine
                    return optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.config.num_epochs,
                        eta_min=self.config.eta_min
                    )
            else:
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, self.config.num_epochs),
                    eta_min=self.config.eta_min
                )
        elif self.config.scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        else:
            return None
    
    def train(self) -> Dict[str, Any]:
        """开始训练"""
        logger.info("Starting training...")
        
        # 训练开始回调
        self.callbacks.on_train_begin()
        
        try:
            # 当无训练数据时，直接跳过训练过程，保持流程可运行
            if self.train_loader is None:
                logger.warning("No train_loader provided. Skipping training loop.")
                return self.training_history

            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Epoch开始回调
                self.callbacks.on_epoch_begin(epoch)
                
                # 训练一个epoch
                train_metrics = self._train_epoch()
                
                # 验证
                val_metrics = None
                if self.val_loader and epoch % self.config.validation_frequency == 0:
                    val_metrics = self._validate_epoch()
                
                # 更新学习率
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if val_metrics:
                            self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # 记录历史
                self.training_history['train_loss'].append(train_metrics['loss'])
                if val_metrics:
                    self.training_history['val_loss'].append(val_metrics['loss'])
                    self.training_history['metrics'].append(val_metrics)
                
                # Epoch结束回调：提供扁平化的关键信息（供回调监控/保存）
                epoch_logs = {
                    'epoch': epoch,
                    'train_loss': train_metrics.get('loss', None)
                }
                if val_metrics:
                    epoch_logs['val_loss'] = val_metrics.get('loss', None)
                    # 可按需添加更多验证指标到日志平面键
                    for k, v in val_metrics.items():
                        if k != 'loss' and isinstance(v, (int, float)):
                            epoch_logs[f'val_{k}'] = v
                self.callbacks.on_epoch_end(epoch, epoch_logs)
                
                # 日志输出
                self._log_epoch_results(epoch, train_metrics, val_metrics)
                
                # 早停检查（可关闭）
                if val_metrics and self.config.use_early_stopping and self._should_stop_early(val_metrics['loss']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        # 训练结束回调
        self.callbacks.on_train_end()
        
        logger.info("Training completed")
        return self.training_history
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Batch开始回调
            self.callbacks.on_batch_begin(batch_idx)
            
            # 前向传播和损失计算
            loss, batch_metrics = self._train_step(batch)
            
            # 反向传播
            if self.config.mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Batch结束回调
            batch_logs = {
                'batch': batch_idx,
                'loss': loss.item(),
                'metrics': batch_metrics
            }
            self.callbacks.on_batch_end(batch_idx, batch_logs)
        
        return {'loss': total_loss / num_batches}
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """单步训练"""
        # 将数据移到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        # 拆分模型输入与监督目标
        model_inputs, targets = self._split_inputs_targets(batch)

        if self.config.mixed_precision and self.scaler:
            try:
                from torch import amp as _amp
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                # 仅在CUDA时启用autocast
                with _amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                    outputs = self.model(**model_inputs)
                    loss = self._compute_loss(outputs, targets)
            except Exception:
                # 回退旧API
                with torch.cuda.amp.autocast():
                    outputs = self.model(**model_inputs)
                    loss = self._compute_loss(outputs, targets)
        else:
            outputs = self.model(**model_inputs)
            loss = self._compute_loss(outputs, targets)
        
        # 计算指标
        batch_metrics = self.metrics_calculator.compute_batch_metrics(outputs, targets)
        
        return loss, batch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 将数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                model_inputs, targets = self._split_inputs_targets(batch)
                outputs = self.model(**model_inputs)
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # 计算验证指标
        val_metrics = self.metrics_calculator.compute_epoch_metrics(all_outputs, all_targets)
        val_metrics['loss'] = total_loss / len(self.val_loader)
        
        return val_metrics

    def _split_inputs_targets(self, batch: Dict[str, Any]) -> tuple:
        """根据模型forward签名，将batch拆分为模型输入和监督目标"""
        try:
            sig = inspect.signature(self.model.forward)
            allowed = set(sig.parameters.keys())
        except (ValueError, TypeError):
            # 回退：若无法获取签名，假设全部为输入
            allowed = set(batch.keys())
        model_inputs = {k: v for k, v in batch.items() if k in allowed}
        # 适配：若模型接受 x_input 而 batch 提供了 x，则自动映射
        if ('x_input' in allowed) and ('x' in batch) and ('x_input' not in model_inputs):
            model_inputs['x_input'] = batch['x']
        targets = {k: v for k, v in batch.items() if k not in allowed}
        return model_inputs, targets
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算损失"""
        if hasattr(self.model, 'compute_multitask_loss'):
            # 多任务模型
            loss_dict = self.model.compute_multitask_loss(outputs, targets, self.config.task_weights)
            return loss_dict['total_loss']
        # 单任务分支
        # 1) 二分类：期望 outputs 包含 'logit'，targets 包含 'y' 或 'label'
        if 'logit' in outputs and ('y' in targets or 'label' in targets):
            y_key = 'y' if 'y' in targets else 'label'
            criterion = getattr(self, 'criterion', None)
            if criterion is None:
                criterion = nn.BCEWithLogitsLoss()
            return criterion(outputs['logit'], targets[y_key])
        # 2) 回归：结合亲和力
        if 'binding_affinity' in outputs and 'binding_affinity' in targets:
            return nn.MSELoss()(outputs['binding_affinity'], targets['binding_affinity'])
        # 3) 其它情况：给出可诊断信息
        raise ValueError(
            f"Cannot compute loss: missing required keys. Outputs keys: {list(outputs.keys())}, Targets keys: {list(targets.keys())}"
        )
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """早停检查"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter = getattr(self, 'patience_counter', 0) + 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ):
        """记录epoch结果"""
        log_msg = f"Epoch {epoch:3d}: Train Loss: {train_metrics['loss']:.4f}"
        
        if val_metrics:
            log_msg += f", Val Loss: {val_metrics['loss']:.4f}"
            
            # 添加其他指标
            for key, value in val_metrics.items():
                if key != 'loss' and isinstance(value, (int, float)):
                    log_msg += f", {key}: {value:.4f}"
        
        # 添加学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        log_msg += f", LR: {current_lr:.6e}"
        
        logger.info(log_msg)
    
    def save_checkpoint(self, path: str, include_optimizer: bool = True):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """预测"""
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 将数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                # 仅传入模型 forward 需要的键，避免将 'y' 等标签传入
                model_inputs, _ = self._split_inputs_targets(batch)
                outputs = self.model(**model_inputs)
                all_predictions.append(outputs)
        
        # 合并预测结果
        combined_predictions = {}
        for key in all_predictions[0].keys():
            if isinstance(all_predictions[0][key], torch.Tensor):
                combined_predictions[key] = torch.cat([pred[key] for pred in all_predictions]).cpu().numpy()
        
        return combined_predictions

    def evaluate(self, data_loader: DataLoader, return_predictions: bool = False) -> Dict[str, Any]:
        """在给定数据集上评估模型，返回指标，必要时返回预测。

        Args:
            data_loader: 待评估的数据加载器
            return_predictions: 若为 True，则返回拼接后的预测与目标

        Returns:
            若 return_predictions=False: Dict[str, float] (metrics)
            若 return_predictions=True: Dict 包含 'metrics', 'predictions', 'targets'
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs: List[Dict[str, torch.Tensor]] = []
        all_targets: List[Dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                model_inputs, targets = self._split_inputs_targets(batch)
                outputs = self.model(**model_inputs)
                # 若可计算损失，则一并累计
                try:
                    loss = self._compute_loss(outputs, targets)
                    total_loss += loss.item()
                except Exception:
                    pass
                all_outputs.append(outputs)
                all_targets.append(targets)

        # 判定任务类型：优先检测二分类（logit + y/label），否则尝试回归（binding_affinity）
        is_classification = False
        y_key = None
        for out, tgt in zip(all_outputs, all_targets):
            if 'logit' in out and (('y' in tgt) or ('label' in tgt)):
                is_classification = True
                y_key = 'y' if 'y' in tgt else 'label'
                break

        metrics: Dict[str, Any] = {}
        combined_predictions: Dict[str, np.ndarray] = {}
        combined_targets: Dict[str, np.ndarray] = {}

        if is_classification:
            # 汇总logit与y
            logits_list = []
            ys_list = []
            for out, tgt in zip(all_outputs, all_targets):
                if 'logit' in out and (y_key in tgt):
                    logits_list.append(out['logit'].detach().cpu())
                    ys_list.append(tgt[y_key].detach().cpu())
            if logits_list:
                logits = torch.cat(logits_list).view(-1)
                ys = torch.cat(ys_list).view(-1).to(dtype=torch.float32)
                probs = torch.sigmoid(logits)
                # 使用 MetricsCalculator 计算分类指标
                self.metrics_calculator.reset()
                self.metrics_calculator.predictions = [probs.numpy()]
                self.metrics_calculator.targets = [ys.numpy()]
                metrics = self.metrics_calculator.compute_classification_metrics(threshold=0.5)
                if len(data_loader) > 0 and total_loss > 0:
                    metrics['loss'] = total_loss / max(1, len(data_loader))
                # 预测返回
                combined_predictions['logit'] = logits.numpy().reshape(-1, 1)
                combined_predictions['prob'] = probs.numpy().reshape(-1, 1)
                combined_targets[y_key] = ys.numpy().reshape(-1, 1)
        else:
            # 回归路径：沿用原 compute_epoch_metrics（针对 binding_affinity）
            metrics = self.metrics_calculator.compute_epoch_metrics(all_outputs, all_targets)
            if len(data_loader) > 0 and total_loss > 0:
                metrics['loss'] = total_loss / max(1, len(data_loader))
            # 预测返回：拼接所有tensor输出
            if all_outputs:
                first_out = all_outputs[0]
                for key, val in first_out.items():
                    if isinstance(val, torch.Tensor):
                        combined_predictions[key] = torch.cat([o[key] for o in all_outputs]).cpu().numpy()
            if all_targets and isinstance(all_targets[0], dict) and all_targets[0]:
                sample_targets = all_targets[0]
                for key, val in sample_targets.items():
                    if isinstance(val, torch.Tensor):
                        combined_targets[key] = torch.cat([t[key] for t in all_targets]).cpu().numpy()

        if not return_predictions:
            return metrics

        return {
            'metrics': metrics,
            'predictions': combined_predictions,
            'targets': combined_targets
        }
