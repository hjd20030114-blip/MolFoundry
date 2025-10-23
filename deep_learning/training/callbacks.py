#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练回调函数
包含各种训练过程中的回调机制
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import time
import json
import shutil

logger = logging.getLogger(__name__)

class Callback(ABC):
    """回调函数基类"""
    
    def __init__(self):
        self.trainer = None
        
    def set_trainer(self, trainer):
        """设置训练器引用"""
        self.trainer = trainer
        
    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始时调用"""
        pass
        
    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束时调用"""
        pass
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """每个epoch开始时调用"""
        pass
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """每个epoch结束时调用"""
        pass
        
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """每个batch开始时调用"""
        pass
        
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """每个batch结束时调用"""
        pass

class CallbackList:
    """回调函数列表管理器"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        
    def append(self, callback: Callback):
        """添加回调函数"""
        self.callbacks.append(callback)
        
    def set_trainer(self, trainer):
        """为所有回调函数设置训练器引用"""
        for callback in self.callbacks:
            callback.set_trainer(trainer)
            
    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始"""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
            
    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束"""
        for callback in self.callbacks:
            callback.on_train_end(logs)
            
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """epoch开始"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
            
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """epoch结束"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
            
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """batch开始"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
            
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """batch结束"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

class ModelCheckpoint(Callback):
    """模型检查点回调"""
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = 'val_loss',
                 save_best_only: bool = True,
                 mode: str = 'min',
                 save_freq: int = 1,
                 best_save_freq: int = 0,
                 final_best_path: Optional[str] = None):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        # 每隔 best_save_freq 个epoch，保存“当前最佳”模型（若>0）
        self.best_save_freq = best_save_freq
        # 训练结束时，将最佳检查点拷贝到该稳定路径（若提供）
        self.final_best_path: Optional[Path] = Path(final_best_path) if final_best_path else None
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epochs_since_last_save = 0
        self.epochs_since_last_best_snapshot = 0
        self.best_checkpoint_path: Optional[str] = None  # 兼容保留（不再依赖）
        # 内存缓存最佳权重（避免每次提升都落盘）
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None
        self.best_scheduler_state_dict = None
        self.best_epoch: Optional[int] = None
        self.best_logs: Optional[Dict[str, Any]] = None
        
        # 确保目录存在
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """epoch结束时检查是否保存模型"""
        if logs is None:
            return
            
        current_value = logs.get(self.monitor)
        if current_value is None:
            # 兼容旧版：从嵌套日志中回退提取
            if self.monitor == 'val_loss' and isinstance(logs.get('val_metrics'), dict):
                current_value = logs['val_metrics'].get('loss')
            elif self.monitor == 'train_loss' and isinstance(logs.get('train_metrics'), dict):
                current_value = logs['train_metrics'].get('loss')
            
        if current_value is None:
            logger.warning(f"监控指标 {self.monitor} 不存在于日志中")
            return
            
        self.epochs_since_last_save += 1
        self.epochs_since_last_best_snapshot += 1
        
        # 检查是否需要保存
        should_save = False
        improved = False
        
        if self.save_best_only:
            if self.mode == 'min' and current_value < self.best_value:
                self.best_value = current_value
                improved = True
            elif self.mode == 'max' and current_value > self.best_value:
                self.best_value = current_value
                improved = True
        else:
            if self.epochs_since_last_save >= self.save_freq:
                should_save = True
                
        # 如果取得了更优成绩：仅缓存，不立刻落盘（当 save_best_only=True 时）
        if self.save_best_only and improved and self.trainer is not None:
            try:
                # 缓存最佳的模型/优化器/调度器权重和相关信息
                self.best_model_state_dict = {k: v.cpu() for k, v in self.trainer.model.state_dict().items()}
                self.best_optimizer_state_dict = self.trainer.optimizer.state_dict() if hasattr(self.trainer, 'optimizer') else None
                self.best_scheduler_state_dict = self.trainer.scheduler.state_dict() if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None else None
                self.best_epoch = epoch
                self.best_logs = dict(logs) if logs else {'epoch': epoch, self.monitor: current_value}
            except Exception as e:
                logger.warning(f"缓存最佳权重失败: {e}")
            # 重置周期计数器，让下一个 best_save_freq 触发尽快保存一次快照
            self.epochs_since_last_best_snapshot = 0

        # 当 save_best_only=False 时，保持原有周期性保存当前权重
        if should_save:
            _ = self._save_model(epoch, logs)
            self.epochs_since_last_save = 0

        # 若设置了best_save_freq，则每隔一定epoch保存一次“当前最佳”的快照
        if (
            self.save_best_only
            and self.best_save_freq > 0
            and self.epochs_since_last_best_snapshot >= self.best_save_freq
            and self.best_model_state_dict is not None
        ):
            # 构建目标文件名（与常规命名一致），用当前epoch的上下文格式化
            logs_for_format = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float, str))}
            if 'epoch' in logs_for_format:
                logs_for_format.pop('epoch')
            periodic_path = str(self.filepath).format(epoch=epoch, **logs_for_format)
            # 将缓存的“最佳权重”写到周期性文件
            try:
                checkpoint = {
                    'epoch': self.best_epoch if self.best_epoch is not None else epoch,
                    'model_state_dict': self.best_model_state_dict,
                    'optimizer_state_dict': self.best_optimizer_state_dict,
                    'logs': self.best_logs or logs or {},
                    'best_value': self.best_value
                }
                if self.best_scheduler_state_dict is not None:
                    checkpoint['scheduler_state_dict'] = self.best_scheduler_state_dict
                torch.save(checkpoint, periodic_path)
                logger.info(f"保存最佳快照: {periodic_path}")
            except Exception as e:
                logger.warning(f"保存最佳快照失败: {e}")
            finally:
                self.epochs_since_last_best_snapshot = 0
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束时，输出一个稳定文件名的最佳模型副本"""
        # 若缓存了最佳权重，则在训练结束时写入稳定文件名
        if self.best_model_state_dict is None:
            logger.warning("训练结束但未缓存到最佳权重，不进行最终最佳模型保存")
            return
        target_path = self.final_best_path or (self.filepath.parent / 'best_model.pth')
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': self.best_epoch if self.best_epoch is not None else 0,
                'model_state_dict': self.best_model_state_dict,
                'optimizer_state_dict': self.best_optimizer_state_dict,
                'logs': self.best_logs or {},
                'best_value': self.best_value
            }
            if self.best_scheduler_state_dict is not None:
                checkpoint['scheduler_state_dict'] = self.best_scheduler_state_dict
            torch.save(checkpoint, str(target_path))
            logger.info(f"保存最终最佳模型: {target_path}")
        except Exception as e:
            logger.warning(f"保存最终最佳模型失败: {e}")
            
    def _save_model(self, epoch: int, logs: Dict) -> str:
        """保存模型"""
        if self.trainer is None:
            logger.error("训练器引用未设置")
            return ""
            
        # 构建文件名
        # 避免与 logs 中的 'epoch' 重复传参导致 format 冲突
        logs_for_format = {k: v for k, v in logs.items() if isinstance(v, (int, float, str))}
        if 'epoch' in logs_for_format:
            logs_for_format.pop('epoch')
        filename = str(self.filepath).format(epoch=epoch, **logs_for_format)
        
        # 保存模型状态
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'logs': logs,
            'best_value': self.best_value
        }
        
        if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.trainer.scheduler.state_dict()
            
        torch.save(checkpoint, filename)
        logger.info(f"保存模型检查点: {filename}")
        return filename

class EarlyStopping(Callback):
    """早停回调"""
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 mode: str = 'min',
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始时重置状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """检查是否需要早停"""
        if logs is None:
            return
            
        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(f"监控指标 {self.monitor} 不存在于日志中")
            return
            
        # 检查是否改善
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights and self.trainer is not None:
                self.best_weights = self.trainer.model.state_dict().copy()
        else:
            self.wait += 1
            
        # 检查是否需要停止
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.trainer is not None:
                self.trainer.stop_training = True
                
            if self.restore_best_weights and self.best_weights is not None:
                logger.info("恢复最佳权重")
                self.trainer.model.load_state_dict(self.best_weights)
                
    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束时输出早停信息"""
        if self.stopped_epoch > 0:
            logger.info(f"在第 {self.stopped_epoch + 1} 个epoch早停")

class ReduceLROnPlateau(Callback):
    """学习率衰减回调"""
    
    def __init__(self,
                 monitor: str = 'val_loss',
                 factor: float = 0.1,
                 patience: int = 10,
                 mode: str = 'min',
                 min_delta: float = 1e-4,
                 cooldown: int = 0,
                 min_lr: float = 0):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.cooldown_counter = 0
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """检查是否需要降低学习率"""
        if logs is None or self.trainer is None:
            return
            
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
            
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
            
        # 检查是否改善
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        # 检查是否需要降低学习率
        if self.wait >= self.patience:
            old_lr = self.trainer.optimizer.param_groups[0]['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
                logger.info(f"降低学习率: {old_lr:.6f} -> {new_lr:.6f}")
                self.cooldown_counter = self.cooldown
                self.wait = 0

class ProgressLogger(Callback):
    """进度日志回调"""
    
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """记录epoch开始时间"""
        self.epoch_start_time = time.time()
        logger.info(f"开始第 {epoch + 1} 个epoch")
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """记录epoch结束信息"""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            
            log_msg = f"Epoch {epoch + 1} 完成 (耗时: {epoch_time:.2f}s)"
            if logs:
                metrics = [f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))]
                if metrics:
                    log_msg += f" - {' - '.join(metrics)}"
                    
            logger.info(log_msg)
            
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """记录batch信息"""
        if batch % self.log_freq == 0 and logs:
            metrics = [f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))]
            if metrics:
                logger.info(f"Batch {batch}: {' - '.join(metrics)}")

class MetricsLogger(Callback):
    """指标记录回调"""

    def __init__(self, log_file: Optional[str] = None):
        super().__init__()
        self.log_file = log_file
        self.history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """记录epoch指标"""
        if logs:
            epoch_logs = {'epoch': epoch, **logs}
            self.history.append(epoch_logs)

            # 保存到文件
            if self.log_file:
                with open(self.log_file, 'w') as f:
                    json.dump(self.history, f, indent=2)

    def get_history(self) -> List[Dict]:
        """获取训练历史"""
        return self.history

class LearningRateScheduler(Callback):
    """学习率调度器回调"""

    def __init__(self, scheduler):
        """
        初始化学习率调度器

        Args:
            scheduler: PyTorch学习率调度器
        """
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """在epoch结束时更新学习率"""
        if hasattr(self.scheduler, 'step'):
            if logs and hasattr(self.scheduler, 'step') and 'val_loss' in logs:
                # 对于ReduceLROnPlateau调度器
                if hasattr(self.scheduler, 'mode'):
                    self.scheduler.step(logs['val_loss'])
                else:
                    self.scheduler.step()
            else:
                self.scheduler.step()

        # 记录当前学习率
        if self.trainer and hasattr(self.trainer, 'optimizer'):
            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            logger.info(f"当前学习率: {current_lr:.6f}")
