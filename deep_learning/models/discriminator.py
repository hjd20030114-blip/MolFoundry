"""
多任务判别器
并行预测结合亲和力、相互作用指纹、ADMET性质与合成可行性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base_model import BaseModel, ModelConfig, ModelRegistry

class BindingAffinityHead(nn.Module):
    """结合亲和力预测头"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class InteractionFingerprintHead(nn.Module):
    """相互作用指纹预测头"""
    
    def __init__(self, input_dim: int, num_interactions: int = 1024):
        super().__init__()
        self.num_interactions = num_interactions
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_interactions),
            nn.Sigmoid()  # 二进制指纹
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ADMETHead(nn.Module):
    """ADMET性质预测头"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # 吸收性质
        self.absorption = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 口服生物利用度 [0, 1]
        )
        
        # 分布性质
        self.distribution = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # LogP
        )
        
        # 代谢性质
        self.metabolism = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()  # CYP抑制概率
        )
        
        # 排泄性质
        self.excretion = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 肾清除率
        )
        
        # 毒性预测
        self.toxicity = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # hERG, 肝毒性, 致突变性
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'absorption': self.absorption(x),
            'distribution': self.distribution(x),
            'metabolism': self.metabolism(x),
            'excretion': self.excretion(x),
            'toxicity': self.toxicity(x)
        }

class SynthesizabilityHead(nn.Module):
    """合成可行性预测头"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # SA Score预测
        self.sa_score = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] 越低越容易合成
        )
        
        # 反应可行性
        self.reaction_feasibility = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 起始原料可获得性
        self.starting_material_availability = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'sa_score': self.sa_score(x),
            'reaction_feasibility': self.reaction_feasibility(x),
            'starting_material_availability': self.starting_material_availability(x)
        }

class DrugLikenessHead(nn.Module):
    """成药性预测头"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Lipinski规则
        self.lipinski = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # MW, LogP, HBD, HBA
        )
        
        # QED分数
        self.qed = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 药物相似性
        self.drug_similarity = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'lipinski_properties': self.lipinski(x),
            'qed_score': self.qed(x),
            'drug_similarity': self.drug_similarity(x)
        }

@ModelRegistry.register("multitask_discriminator")
class MultiTaskDiscriminator(BaseModel):
    """多任务判别器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.input_dim = config.hidden_dim
        self.hidden_dim = config.hidden_dim
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 任务特定的预测头
        self.binding_affinity_head = BindingAffinityHead(self.hidden_dim)
        self.interaction_fingerprint_head = InteractionFingerprintHead(self.hidden_dim)
        self.admet_head = ADMETHead(self.hidden_dim)
        self.synthesizability_head = SynthesizabilityHead(self.hidden_dim)
        self.drug_likeness_head = DrugLikenessHead(self.hidden_dim)
        
        # 任务权重（可学习）
        self.task_weights = nn.Parameter(torch.ones(5))
        
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
        molecular_features: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            molecular_features: [batch_size, input_dim] 分子特征
            return_features: 是否返回中间特征
        """
        # 共享特征提取
        shared_features = self.feature_extractor(molecular_features)
        
        # 各任务预测
        outputs = {}
        
        # 结合亲和力
        outputs['binding_affinity'] = self.binding_affinity_head(shared_features)
        
        # 相互作用指纹
        outputs['interaction_fingerprint'] = self.interaction_fingerprint_head(shared_features)
        
        # ADMET性质
        admet_outputs = self.admet_head(shared_features)
        outputs.update({f'admet_{k}': v for k, v in admet_outputs.items()})
        
        # 合成可行性
        synth_outputs = self.synthesizability_head(shared_features)
        outputs.update({f'synthesizability_{k}': v for k, v in synth_outputs.items()})
        
        # 成药性
        drug_outputs = self.drug_likeness_head(shared_features)
        outputs.update({f'drug_likeness_{k}': v for k, v in drug_outputs.items()})
        
        if return_features:
            outputs['shared_features'] = shared_features
        
        return outputs
    
    def compute_multitask_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """计算多任务损失"""
        losses = {}
        total_loss = 0.0
        
        # 默认任务权重
        if task_weights is None:
            task_weights = {
                'binding_affinity': 1.0,
                'interaction_fingerprint': 0.5,
                'admet': 0.3,
                'synthesizability': 0.2,
                'drug_likeness': 0.2
            }
        
        # 结合亲和力损失
        if 'binding_affinity' in targets:
            ba_loss = F.mse_loss(
                predictions['binding_affinity'],
                targets['binding_affinity']
            )
            losses['binding_affinity_loss'] = ba_loss
            total_loss += task_weights['binding_affinity'] * ba_loss
        
        # 相互作用指纹损失
        if 'interaction_fingerprint' in targets:
            ifp_loss = F.binary_cross_entropy(
                predictions['interaction_fingerprint'],
                targets['interaction_fingerprint']
            )
            losses['interaction_fingerprint_loss'] = ifp_loss
            total_loss += task_weights['interaction_fingerprint'] * ifp_loss
        
        # ADMET损失
        admet_loss = 0.0
        admet_count = 0
        for key in ['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity']:
            target_key = f'admet_{key}'
            if target_key in targets:
                if key in ['absorption', 'metabolism', 'excretion', 'toxicity']:
                    loss = F.binary_cross_entropy(
                        predictions[target_key],
                        targets[target_key]
                    )
                else:  # distribution (LogP)
                    loss = F.mse_loss(
                        predictions[target_key],
                        targets[target_key]
                    )
                admet_loss += loss
                admet_count += 1
        
        if admet_count > 0:
            admet_loss /= admet_count
            losses['admet_loss'] = admet_loss
            total_loss += task_weights['admet'] * admet_loss
        
        # 合成可行性损失
        synth_loss = 0.0
        synth_count = 0
        for key in ['sa_score', 'reaction_feasibility', 'starting_material_availability']:
            target_key = f'synthesizability_{key}'
            if target_key in targets:
                loss = F.binary_cross_entropy(
                    predictions[target_key],
                    targets[target_key]
                )
                synth_loss += loss
                synth_count += 1
        
        if synth_count > 0:
            synth_loss /= synth_count
            losses['synthesizability_loss'] = synth_loss
            total_loss += task_weights['synthesizability'] * synth_loss
        
        # 成药性损失
        drug_loss = 0.0
        drug_count = 0
        for key in ['lipinski_properties', 'qed_score', 'drug_similarity']:
            target_key = f'drug_likeness_{key}'
            if target_key in targets:
                if key == 'lipinski_properties':
                    loss = F.mse_loss(
                        predictions[target_key],
                        targets[target_key]
                    )
                else:
                    loss = F.binary_cross_entropy(
                        predictions[target_key],
                        targets[target_key]
                    )
                drug_loss += loss
                drug_count += 1
        
        if drug_count > 0:
            drug_loss /= drug_count
            losses['drug_likeness_loss'] = drug_loss
            total_loss += task_weights['drug_likeness'] * drug_loss
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def predict_all_properties(
        self,
        molecular_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """预测所有性质"""
        with torch.no_grad():
            outputs = self.forward(molecular_features)
        
        # 组织输出
        results = {
            'binding_affinity': outputs['binding_affinity'].cpu().numpy(),
            'interaction_fingerprint': outputs['interaction_fingerprint'].cpu().numpy(),
            'admet': {
                'absorption': outputs['admet_absorption'].cpu().numpy(),
                'distribution': outputs['admet_distribution'].cpu().numpy(),
                'metabolism': outputs['admet_metabolism'].cpu().numpy(),
                'excretion': outputs['admet_excretion'].cpu().numpy(),
                'toxicity': outputs['admet_toxicity'].cpu().numpy()
            },
            'synthesizability': {
                'sa_score': outputs['synthesizability_sa_score'].cpu().numpy(),
                'reaction_feasibility': outputs['synthesizability_reaction_feasibility'].cpu().numpy(),
                'starting_material_availability': outputs['synthesizability_starting_material_availability'].cpu().numpy()
            },
            'drug_likeness': {
                'lipinski_properties': outputs['drug_likeness_lipinski_properties'].cpu().numpy(),
                'qed_score': outputs['drug_likeness_qed_score'].cpu().numpy(),
                'drug_similarity': outputs['drug_likeness_drug_similarity'].cpu().numpy()
            }
        }
        
        return results
