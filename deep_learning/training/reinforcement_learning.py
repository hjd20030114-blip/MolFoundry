#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习模块
包含PPO训练器和相关配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO配置"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 2048
    normalize_advantages: bool = True

class PPOBuffer:
    """PPO经验回放缓冲区"""
    
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # 缓冲区
        self.observations = torch.zeros((buffer_size, obs_dim))
        self.actions = torch.zeros((buffer_size, act_dim))
        self.rewards = torch.zeros(buffer_size)
        self.values = torch.zeros(buffer_size)
        self.log_probs = torch.zeros(buffer_size)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.ptr = 0
        self.size = 0
        
    def store(self, obs: torch.Tensor, act: torch.Tensor, rew: float, 
              val: float, log_prob: float, done: bool):
        """存储一步经验"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def get(self, config: PPOConfig) -> Dict[str, torch.Tensor]:
        """获取所有经验并计算优势"""
        assert self.size == self.buffer_size, "缓冲区未满"
        
        # 计算GAE优势
        advantages = self._compute_gae(config.gamma, config.gae_lambda)
        
        # 计算回报
        returns = advantages + self.values
        
        # 标准化优势
        if config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': self.values
        }
    
    def _compute_gae(self, gamma: float, gae_lambda: float) -> torch.Tensor:
        """计算GAE优势"""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = 0  # 假设最后一步的下一个值为0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
        
        return advantages
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        
        # Critic头（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value.squeeze(-1)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作"""
        action_logits, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            log_prob = F.log_softmax(action_logits, dim=-1)[range(len(action)), action]
        else:
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        action_logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, 
                 actor_critic: ActorCritic,
                 config: PPOConfig,
                 device: torch.device = torch.device('cpu')):
        self.actor_critic = actor_critic.to(device)
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=config.learning_rate
        )
        
        # 缓冲区
        obs_dim = actor_critic.feature_extractor[0].in_features
        act_dim = actor_critic.actor[-1].out_features
        self.buffer = PPOBuffer(config.buffer_size, obs_dim, act_dim)
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def collect_experience(self, env, num_steps: int):
        """收集经验"""
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action(obs_tensor)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action.cpu().numpy()[0])
            
            # 存储经验
            self.buffer.store(
                obs_tensor.squeeze(0).cpu(),
                action.cpu(),
                reward,
                value.cpu().item(),
                log_prob.cpu().item(),
                done
            )
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
    
    def update_policy(self) -> Dict[str, float]:
        """更新策略"""
        # 获取经验
        data = self.buffer.get(self.config)
        
        # 移动到设备
        for key in data:
            data[key] = data[key].to(self.device)
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(self.config.buffer_size)
            
            for start in range(0, self.config.buffer_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_obs = data['observations'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_log_probs = data['log_probs'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                
                # 评估当前策略
                new_log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_obs, batch_actions.long()
                )
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values, batch_returns)
                
                # 总损失
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss - 
                       self.config.entropy_coef * entropy.mean())
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 返回训练统计
        num_updates = self.config.ppo_epochs * (self.config.buffer_size // self.config.batch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }
    
    def train_step(self, env) -> Dict[str, float]:
        """执行一步训练"""
        # 收集经验
        self.collect_experience(env, self.config.buffer_size)
        
        # 更新策略
        return self.update_policy()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"模型已从 {path} 加载")

class MolecularEnvironment:
    """分子生成环境（示例）"""
    
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.current_molecule = []
        self.step_count = 0
        
        # 简化的原子词汇表
        self.vocab = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', '<END>']
        self.vocab_size = len(self.vocab)
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_molecule = []
        self.step_count = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        self.step_count += 1
        
        # 添加原子到分子
        if action < len(self.vocab) - 1:  # 不是结束符
            self.current_molecule.append(self.vocab[action])
        
        # 计算奖励
        reward = self._compute_reward(action)
        
        # 检查是否结束
        done = (action == len(self.vocab) - 1 or  # 选择了结束符
                self.step_count >= self.max_length or  # 达到最大长度
                len(self.current_molecule) >= self.max_length)
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 简化的状态表示：当前分子长度和最后几个原子
        state = np.zeros(10)  # 固定长度状态向量
        state[0] = len(self.current_molecule) / self.max_length  # 标准化长度
        
        # 最后几个原子的编码
        for i, atom in enumerate(self.current_molecule[-5:]):
            if i < 5:
                state[i + 1] = self.vocab.index(atom) / len(self.vocab)
        
        return state
    
    def _compute_reward(self, action: int) -> float:
        """计算奖励"""
        # 简化的奖励函数
        reward = 0.0
        
        # 鼓励生成有效的分子
        if action < len(self.vocab) - 1:
            reward += 0.1
        
        # 如果选择结束符且分子长度合理，给予奖励
        if action == len(self.vocab) - 1 and 5 <= len(self.current_molecule) <= 50:
            reward += 1.0
        
        # 惩罚过长的分子
        if len(self.current_molecule) > 50:
            reward -= 0.5
        
        return reward
