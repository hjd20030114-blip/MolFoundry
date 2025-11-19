#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRRSV抑制剂设计平台 - 深度学习流水线
基于SE(3)-Equivariant GNN和Transformer的分子生成与筛选系统

实施路线：
- 30天：训练Equivariant GNN评分器，并在PRRSV蛋白数据上微调
- 60天：实现Pocket-conditioned Transformer生成候选分子，结合GNN筛选与MD精修
- 90天：基于规则与强化学习的优化与主动学习闭环
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime
import json
from torch.utils.data import Dataset, DataLoader, random_split

# 添加深度学习模块路径
sys.path.append('deep_learning')

from deep_learning.models import (
    EquivariantGNN, 
    PocketLigandTransformer, MultiTaskDiscriminator,
    ModelConfig, create_model
)
from deep_learning.data.featurizers import (
    MolecularFeaturizer, ProteinFeaturizer, InteractionFeaturizer,
    FeaturizationConfig
)
from deep_learning.training import Trainer, TrainingConfig
from deep_learning.evaluation import MolecularEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepLearningPipeline:
    """深度学习流水线主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化流水线"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        self.output_dir = Path(self.config.get('output_dir', 'deep_learning_results'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.featurization_config = FeaturizationConfig(**self.config.get('featurization', {}))
        self.molecular_featurizer = MolecularFeaturizer(self.featurization_config)
        self.protein_featurizer = ProteinFeaturizer(self.featurization_config)
        self.interaction_featurizer = InteractionFeaturizer(self.featurization_config)
        
        # 模型配置
        self.model_configs = {
            'equivariant_gnn': ModelConfig(
                model_type='equivariant_gnn',
                **self.config.get('equivariant_gnn', {})
            ),
            'pocket_ligand_transformer': ModelConfig(
                model_type='pocket_ligand_transformer',
                **self.config.get('pocket_ligand_transformer', {})
            ),
            'multitask_discriminator': ModelConfig(
                model_type='multitask_discriminator',
                **self.config.get('multitask_discriminator', {})
            )
        }
        
        # 训练配置
        self.training_config = TrainingConfig(**self.config.get('training', {}))
        # CPU/MPS 安全设置：禁用混合精度与固定内存，使用单进程数据加载，避免多进程pickle问题
        if self.device.type != "cuda":
            try:
                self.training_config.mixed_precision = False
                self.training_config.pin_memory = False
                self.training_config.num_workers = 0
            except Exception:
                # 容错，不影响后续流程
                pass
        
        # 外部推理集成选项（可由UI设置）
        self.external = self.config.get('external', {})
        self.external.setdefault('use_real_transformer', False)
        self.external.setdefault('transformer_script', 'scripts/external_transformer_infer.py')
        self.external.setdefault('transformer_checkpoint', None)
        # 可选：外部模块/函数与额外参数
        self.external.setdefault('tf_module', None)
        self.external.setdefault('tf_function', None)
        self.external.setdefault('tf_kwargs_json', None)
        
        logger.info(f"深度学习流水线初始化完成，使用设备: {self.device}")

    def set_external_options(self, external_opts: Dict):
        """设置外部推理集成选项"""
        if not isinstance(external_opts, dict):
            return
        self.external.update(external_opts)

    def check_model_availability(self) -> Dict[str, bool]:
        """
        检查模型可用性

        Returns:
            模型可用性字典
        """
        availability = {}

        try:
            # 检查PyTorch
            import torch
            availability['pytorch'] = True
        except ImportError:
            availability['pytorch'] = False

        try:
            # 检查PyTorch Geometric
            import torch_geometric
            availability['torch_geometric'] = True
        except ImportError:
            availability['torch_geometric'] = False

        try:
            # 检查e3nn
            import e3nn
            availability['e3nn'] = True
        except ImportError:
            availability['e3nn'] = False

        try:
            # 检查scikit-learn
            import sklearn
            availability['sklearn'] = True
        except ImportError:
            availability['sklearn'] = False

        # 检查配置
        availability['config_loaded'] = self.config is not None

        # 检查模型配置
        if self.config:
            availability['gnn_config'] = 'equivariant_gnn' in self.config.get('models', {})
            availability['transformer_config'] = 'transformer' in self.config.get('models', {})
            availability['discriminator_config'] = 'discriminator' in self.config.get('models', {})
        else:
            availability['gnn_config'] = False
            availability['transformer_config'] = False
            availability['discriminator_config'] = False

        return availability

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'output_dir': 'deep_learning_results',
            'featurization': {
                'use_morgan_fingerprints': True,
                'morgan_radius': 2,
                'morgan_nbits': 2048,
                'use_maccs_keys': True,
                'use_descriptors': True,
                'pocket_radius': 5.0,
                'interaction_cutoff': 4.0
            },
            'equivariant_gnn': {
                'hidden_dim': 256,
                'num_layers': 6,
                'dropout': 0.1,
                'max_radius': 5.0,
                'num_neighbors': 32
            },
            'pocket_ligand_transformer': {
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 6,
                'ff_dim': 1024,
                'dropout': 0.1
            },
            'multitask_discriminator': {
                'hidden_dim': 256,
                'dropout': 0.1
            },
            'training': {
                'num_epochs': 10000,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'early_stopping_patience': 20
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # 递归更新配置
            self._update_config(default_config, user_config)
        
        return default_config
    
    def _update_config(self, default: Dict, user: Dict):
        """递归更新配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._update_config(default[key], value)
            else:
                default[key] = value
    
    def phase_1_equivariant_gnn(self, protein_pdb: str, ligand_data: List[Dict]) -> str:
        """
        第一阶段（30天）：训练Equivariant GNN评分器
        
        Args:
            protein_pdb: PRRSV蛋白PDB文件路径
            ligand_data: 配体数据列表，每个元素包含SMILES和结合亲和力
        
        Returns:
            训练好的模型路径
        """
        logger.info("🚀 开始第一阶段：训练SE(3)-Equivariant GNN评分器")
        
        # 1. 数据预处理
        logger.info("📊 预处理数据...")
        processed_data = self._preprocess_data_phase1(protein_pdb, ligand_data)
        
        # 2. 创建数据加载器
        train_loader, val_loader = self._create_data_loaders_phase1(processed_data)
        
        # 3. 创建模型
        logger.info("🧠 创建SE(3)-Equivariant GNN模型...")
        model = create_model('equivariant_gnn', self.model_configs['equivariant_gnn'])
        
        # 4. 训练模型
        logger.info("🏋️ 开始训练...")
        trainer = Trainer(
            model=model,
            config=self.training_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        training_history = trainer.train()
        
        # 5. 保存模型
        model_path = self.output_dir / "phase1_equivariant_gnn.pth"
        trainer.save_checkpoint(str(model_path))
        
        # 6. 评估模型（若无验证集则跳过）
        logger.info("📈 评估模型性能...")
        evaluator = MolecularEvaluator()
        if val_loader is not None:
            evaluation_results = evaluator.evaluate_binding_affinity_model(
                model, val_loader, self.device
            )
        else:
            logger.warning("Val loader is None, skip evaluation. Returning default metrics.")
            evaluation_results = {
                'mse': None,
                'rmse': None,
                'mae': None,
                'correlation': None,
                'r2_score': None,
                'num_samples': 0
            }
        
        # 7. 保存结果
        results = {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'model_config': self.model_configs['equivariant_gnn'].to_dict(),
            'training_config': self.training_config.__dict__
        }
        
        results_path = self.output_dir / "phase1_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ 第一阶段完成！模型保存至: {model_path}")
        logger.info(f"📊 结果保存至: {results_path}")
        
        return str(model_path)
    
    def phase_2_transformer_generation(
        self,
        phase1_model_path: str,
        protein_pdb: str,
        generation_targets: List[Dict]
    ) -> str:
        """
        第二阶段（60天）：实现Pocket-conditioned Transformer生成
        
        Args:
            phase1_model_path: 第一阶段训练的GNN模型路径
            protein_pdb: PRRSV蛋白PDB文件路径
            generation_targets: 生成目标列表
        
        Returns:
            生成的分子数据路径
        """
        logger.info("🚀 开始第二阶段：Pocket-conditioned Transformer分子生成")
        
        # 1. 加载第一阶段模型
        logger.info("📥 加载第一阶段GNN模型...")
        gnn_model = create_model('equivariant_gnn', self.model_configs['equivariant_gnn'])
        # 兼容两种格式：1) 完整checkpoint含'model_state_dict'；2) 仅state_dict
        try:
            checkpoint = torch.load(phase1_model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(phase1_model_path, map_location=self.device)
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                state_dict = checkpoint['model_state_dict']
            else:
                # 假定直接是state_dict
                state_dict = checkpoint
        if state_dict is None:
            raise ValueError(f"无法从检查点解析state_dict: {phase1_model_path}")
        missing, unexpected = gnn_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"加载权重缺失键: {missing}")
        if unexpected:
            logger.warning(f"加载权重中存在未使用键: {unexpected}")
        gnn_model.eval()
        
        # 2. 创建Transformer生成模型
        logger.info("🧠 创建Pocket-Ligand Transformer模型...")
        transformer_model = create_model(
            'pocket_ligand_transformer', 
            self.model_configs['pocket_ligand_transformer']
        )
        
        # 3. 预处理蛋白口袋数据
        logger.info("🔬 分析蛋白口袋...")
        pocket_features = self._extract_pocket_features(protein_pdb)
        
        # 4. 生成候选分子
        logger.info("🧪 生成候选分子...")
        if self.external.get('use_real_transformer', False):
            logger.info("🔌 使用外部Transformer推理脚本进行生成")
            generated_molecules = self._external_transformer_generate(
                protein_pdb, generation_targets
            )
        else:
            generated_molecules = self._generate_molecules_with_transformer(
                transformer_model, pocket_features, generation_targets
            )
        
        # 5. 使用GNN筛选分子
        logger.info("🔍 使用GNN筛选生成的分子...")
        screened_molecules = self._screen_molecules_with_gnn(
            gnn_model, generated_molecules, pocket_features
        )
        
        # 6. MD精修（简化版）
        logger.info("⚗️ 分子动力学精修...")
        refined_molecules = self._md_refinement(screened_molecules, protein_pdb)
        
        # 7. 保存结果
        output_path = self.output_dir / "phase2_generated_molecules.csv"
        pd.DataFrame(refined_molecules).to_csv(output_path, index=False)
        
        logger.info(f"✅ 第二阶段完成！生成 {len(refined_molecules)} 个候选分子")
        logger.info(f"📊 结果保存至: {output_path}")
        
        return str(output_path)
    
    def phase_3_diffusion_and_rl(
        self,
        phase2_data_path: str,
        protein_pdb: str,
        optimization_targets: Dict
    ) -> str:
        """
        第三阶段（90天）：强化学习优化（已移除Diffusion模块）
        
        Args:
            phase2_data_path: 第二阶段生成的分子数据路径
            protein_pdb: PRRSV蛋白PDB文件路径
            optimization_targets: 优化目标
        
        Returns:
            最终优化的分子数据路径
        """
        logger.info("🚀 开始第三阶段：多目标优化（无Diffusion/无RL）")
        
        # 1. 加载第二阶段数据
        phase2_data = pd.read_csv(phase2_data_path)
        
        # 2. 多目标优化（无Diffusion/无RL）
        logger.info("🎯 执行多目标优化（Pareto/加权排序）...")
        optimized_molecules = self._multi_objective_optimization(
            phase2_data, protein_pdb, optimization_targets
        )
        
        # 6. 主动学习闭环
        logger.info("🔄 主动学习闭环...")
        final_molecules = self._active_learning_loop(
            optimized_molecules, protein_pdb, optimization_targets
        )
        
        # 7. 保存最终结果
        output_path = self.output_dir / "phase3_optimized_molecules.csv"
        pd.DataFrame(final_molecules).to_csv(output_path, index=False)
        
        # 8. 生成综合报告
        report_path = self._generate_final_report(final_molecules, protein_pdb)
        
        logger.info(f"✅ 第三阶段完成！优化得到 {len(final_molecules)} 个最终候选分子")
        logger.info(f"📊 结果保存至: {output_path}")
        logger.info(f"📋 综合报告: {report_path}")
        
        return str(output_path)
    
    def run_full_pipeline(
        self,
        protein_pdb: str,
        initial_ligand_data: List[Dict],
        generation_targets: List[Dict],
        optimization_targets: Dict
    ) -> Dict[str, str]:
        """运行完整的深度学习流水线"""
        logger.info("🚀 开始运行完整的深度学习流水线")
        
        results = {}
        
        try:
            # 第一阶段
            phase1_model = self.phase_1_equivariant_gnn(protein_pdb, initial_ligand_data)
            results['phase1_model'] = phase1_model
            
            # 第二阶段
            phase2_molecules = self.phase_2_transformer_generation(
                phase1_model, protein_pdb, generation_targets
            )
            results['phase2_molecules'] = phase2_molecules
            
            # 第三阶段
            phase3_molecules = self.phase_3_diffusion_and_rl(
                phase2_molecules, protein_pdb, optimization_targets
            )
            results['phase3_molecules'] = phase3_molecules
            
            logger.info("🎉 完整流水线执行成功！")
            
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {e}")
            raise
        
        return results
    
    # 辅助方法与第一阶段实现
    def _build_protein_graph(self, protein_pdb: str, max_atoms: int = 200, radius: float = 5.0) -> Dict[str, torch.Tensor]:
        """从PDB解析蛋白原子，构建简化图: 原子类型(int) + 坐标 + 半径邻接边
        - 不依赖MDAnalysis，直接解析PDB ATOM/HETATM 行
        - 为控制复杂度，仅取前 max_atoms 个原子
        - 边: 两点距离<=radius 即连边（双向）
        """
        elements = []
        coords = []
        try:
            with open(protein_pdb, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                        except Exception:
                            continue
                        elem = line[76:78].strip()
                        if not elem:
                            name = line[12:16].strip()
                            elem = name[0]
                        elem = elem.capitalize()
                        elements.append(elem)
                        coords.append([x, y, z])
                        if len(coords) >= max_atoms:
                            break
        except Exception as e:
            logger.warning(f"解析PDB失败，使用空蛋白图: {e}")
            elements, coords = [], []

        if not coords:
            return {
                'atom_types': torch.zeros(0, dtype=torch.long, device=self.device),
                'positions': torch.zeros(0, 3, dtype=torch.float32, device=self.device),
                'edge_index': torch.zeros(2, 0, dtype=torch.long, device=self.device)
            }

        # 元素到类型ID映射（确保 < 100 以匹配 EquivariantGNN 的 nn.Embedding(100, ...)）
        element_table = [
            'H','C','N','O','S','P','F','Cl','Br','I',
            'Na','K','Ca','Mg','Zn','Fe','Cu','Mn','Co','Ni'
        ]
        elem2id = {e: i+1 for i, e in enumerate(element_table)}  # 0保留为未知

        atom_types = torch.tensor([elem2id.get(e, 0) for e in elements], dtype=torch.long)
        positions = torch.tensor(np.array(coords, dtype=np.float32))

        # 构建半径邻接边
        with torch.no_grad():
            dmat = torch.cdist(positions, positions)
            mask = (dmat <= radius) & (~torch.eye(positions.size(0), dtype=torch.bool))
            src, dst = torch.where(mask)
            edge_index = torch.stack([src, dst], dim=0).to(torch.long)

        return {
            'atom_types': atom_types.to(self.device),
            'positions': positions.to(self.device),
            'edge_index': edge_index.to(self.device)
        }

    def _build_ligand_graph(self, smiles: str, radius: float = 5.0) -> Dict[str, torch.Tensor]:
        """从SMILES生成配体图（优先RDKit 3D; 失败则回退为随机小图）"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                raise ValueError("无效SMILES")
            m = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            ok = AllChem.EmbedMolecule(m, params)
            if ok != 0:
                # 回退随机坐标
                ok = AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42)
            try:
                # 优化
                AllChem.UFFOptimizeMolecule(m)
            except Exception:
                pass
            conf = m.GetConformer()
            atom_types = []
            coords = []
            for i, atom in enumerate(m.GetAtoms()):
                atom_types.append(int(atom.GetAtomicNum()) % 100)  # 映射到 <100
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            positions = torch.tensor(np.array(coords, dtype=np.float32))
            atom_types = torch.tensor(atom_types, dtype=torch.long)
            # 键为边（双向）
            edges = []
            for b in m.GetBonds():
                i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
                edges.append([i, j]); edges.append([j, i])
            if edges:
                edge_index = torch.tensor(np.array(edges, dtype=np.int64)).t().contiguous()
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
            return {
                'atom_types': atom_types.to(self.device),
                'positions': positions.to(self.device),
                'edge_index': edge_index.to(self.device)
            }
        except Exception:
            # 简易回退：10个碳的随机链
            n = 10
            rng = np.random.default_rng(42)
            positions = torch.tensor(
                rng.normal(0, 1.0, size=(n, 3)).astype(np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            atom_types = torch.full((n,), 6, dtype=torch.long, device=self.device)  # 6近似为C
            edges = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]
            edge_index = torch.tensor(
                np.array(edges, dtype=np.int64), dtype=torch.long, device=self.device
            ).t().contiguous()
            return {
                'atom_types': atom_types,
                'positions': positions,
                'edge_index': edge_index
            }

    def _merge_protein_ligand(self, prot: Dict[str, torch.Tensor], lig: Dict[str, torch.Tensor], cross_radius: float = 5.0) -> Dict[str, torch.Tensor]:
        """合并蛋白与配体图并添加跨分子边（<= cross_radius）"""
        n_prot = prot['atom_types'].size(0)
        n_lig = lig['atom_types'].size(0)
        atom_types = torch.cat([prot['atom_types'], lig['atom_types']], dim=0)
        positions = torch.cat([prot['positions'], lig['positions']], dim=0)
        # 偏移配体边
        lig_edge = lig['edge_index'] + n_prot
        edge_index = torch.cat([prot['edge_index'], lig_edge], dim=1) if prot['edge_index'].numel() > 0 else lig_edge
        # 跨边
        if n_prot > 0 and n_lig > 0:
            d = torch.cdist(lig['positions'], prot['positions'])  # [N_lig,N_prot]
            rows, cols = torch.where(d <= cross_radius)
            if rows.numel() > 0:
                # lig i -> prot j 以及 prot j -> lig i
                cross1 = torch.stack([rows + n_prot, cols], dim=0)
                cross2 = torch.stack([cols, rows + n_prot], dim=0)
                edge_index = torch.cat([edge_index, cross1, cross2], dim=1)
        return {
            'atom_types': atom_types,
            'positions': positions,
            'edge_index': edge_index
        }

    def _preprocess_data_phase1(self, protein_pdb: str, ligand_data: List[Dict]) -> Dict:
        """预处理第一阶段数据：构建蛋白图并缓存配体条目（SMILES与标签）"""
        radius = float(self.model_configs['equivariant_gnn'].max_radius)
        prot_graph = self._build_protein_graph(protein_pdb, max_atoms=200, radius=radius)
        # 仅保留含有SMILES与binding_affinity的条目
        lig_items = []
        for item in ligand_data:
            smi = str(item.get('smiles', '')).strip()
            if not smi:
                continue
            try:
                y = float(item.get('binding_affinity', np.nan))
            except Exception:
                y = np.nan
            if np.isnan(y):
                continue
            lig_items.append({'smiles': smi, 'binding_affinity': y})
        if not lig_items:
            logger.warning("Phase1 无有效配体数据，后续训练将被跳过")
        return {'protein_graph': prot_graph, 'ligand_items': lig_items}
    
    def _create_data_loaders_phase1(self, data: Dict) -> Tuple:
        """创建第一阶段数据加载器（图批处理，提供 batch 向量）"""
        prot = data.get('protein_graph', None)
        lig_items = data.get('ligand_items', [])
        if prot is None or not lig_items:
            logger.warning("Phase1 数据不足，返回空数据加载器")
            return None, None

        class GraphBindingDataset(Dataset):
            def __init__(self, pipeline: 'DeepLearningPipeline', prot_graph: Dict[str, torch.Tensor], items: List[Dict], radius: float):
                self.pipeline = pipeline
                self.prot = prot_graph
                self.items = items
                self.radius = radius
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                item = self.items[idx]
                lig = self.pipeline._build_ligand_graph(item['smiles'], radius=self.radius)
                merged = self.pipeline._merge_protein_ligand(self.prot, lig, cross_radius=self.radius)
                y = torch.tensor([[float(item['binding_affinity'])]], dtype=torch.float32, device=self.pipeline.device)
                return {
                    'atom_types': merged['atom_types'],
                    'positions': merged['positions'],
                    'edge_index': merged['edge_index'],
                    'binding_affinity': y
                }

        def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # 合并多图到单个批：拼接节点，边索引偏移，生成batch向量
            node_offset = 0
            atom_types_list = []
            positions_list = []
            edge_indices = []
            batch_vec = []
            targets = []
            for i, sample in enumerate(batch):
                n = sample['atom_types'].size(0)
                atom_types_list.append(sample['atom_types'])
                positions_list.append(sample['positions'])
                edge_indices.append(sample['edge_index'] + node_offset)
                batch_vec.append(torch.full((n,), i, dtype=torch.long, device=sample['atom_types'].device))
                targets.append(sample['binding_affinity'])
                node_offset += n
            atom_types = torch.cat(atom_types_list, dim=0)
            positions = torch.cat(positions_list, dim=0)
            edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2,0,dtype=torch.long, device=atom_types.device)
            batch_tensor = torch.cat(batch_vec, dim=0)
            y = torch.cat(targets, dim=0)  # [B,1]
            return {
                'atom_types': atom_types,
                'positions': positions,
                'edge_index': edge_index,
                'batch': batch_tensor,
                'binding_affinity': y
            }

        radius = float(self.model_configs['equivariant_gnn'].max_radius)
        dataset = GraphBindingDataset(self, prot, lig_items, radius)

        # 8:2 划分
        total = len(dataset)
        if total == 0:
            return None, None
        val_size = max(1, int(0.2 * total))
        train_size = total - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.training_config.batch_size, shuffle=True, num_workers=self.training_config.num_workers, pin_memory=self.training_config.pin_memory, collate_fn=collate_graph_batch)
        val_loader = DataLoader(val_ds, batch_size=self.training_config.batch_size, shuffle=False, num_workers=self.training_config.num_workers, pin_memory=self.training_config.pin_memory, collate_fn=collate_graph_batch)
        return train_loader, val_loader
    
    def _extract_pocket_features(self, protein_pdb: str) -> Dict:
        """提取口袋特征（用于后续生成/筛选阶段的上下文）。此处返回蛋白图以复用。"""
        radius = float(self.model_configs['equivariant_gnn'].max_radius)
        prot_graph = self._build_protein_graph(protein_pdb, max_atoms=200, radius=radius)
        return prot_graph
    
    def _generate_molecules_with_transformer(self, model, pocket_features, targets) -> List[Dict]:
        """使用Transformer生成分子"""
        # 占位实现：基于一个基础SMILES库与简单启发式，生成目标数量的候选分子
        try:
            # 目标数量与目标亲和力
            total_num = 0
            target_aff = -6.0
            if isinstance(targets, list) and targets:
                for t in targets:
                    total_num += int(t.get('num_molecules', 0))
                target_aff = float(targets[0].get('target_affinity', target_aff))
            if total_num <= 0:
                total_num = 50

            # 基础SMILES库（内置+动态组合，目标：数百到上千唯一分子）
            base_library = [
                # 小分子与溶剂样分子（简单、鲁棒）
                "CCO", "CCN", "CCC", "CC(C)O", "CC(C)N", "C1CCCCC1", "c1ccccc1",
                # 已验证的常见取代芳香片段
                "Cc1ccccc1", "Clc1ccccc1", "Fc1ccccc1", "Brc1ccccc1",
                "COc1ccccc1", "Oc1ccccc1", "NCc1ccccc1", "CC(=O)Nc1ccccc1",
                # 羧酸/酯/酰胺/脲/磺酰胺
                "CC(=O)O", "CCOC(=O)C1=CC=CC=C1", "CCOC(=O)NCC", "CNC(=O)C1=CC=CC=C1",
                "C1=CC(=O)NC(=O)N1", "CCS(=O)(=O)N1CCOCC1", "O=S(=O)(Nc1ccccc1)C",
                # 杂环与药效团
                "N1CCOCC1", "N1CCCCC1", "O1CCNCC1", "O1CCOCC1", "C1CCN(CC1)C",
                # 多环与稠环
                "c1ccc2ccccc2c1", "COc1ccc2ccccc2c1",
                # 杂取代苯环（已用过的安全模板）
                "c1ccc(OC)cc1", "c1ccc(CN)cc1", "c1ccc(CF)cc1",
                # 其它常见片段
                "CC(C)OC(=O)N", "CC(C)NCCO", "CCN(CC)CC", "NCCO", "CCOC",
                "CC(C)C(=O)N", "COCCN", "OCCN1CCCC1",
                # 含氟/氯修饰的芳香酰胺
                "O=C(Nc1ccc(F)cc1)C2CC2", "O=C(Nc1ccc(Cl)cc1)C2CC2",
            ]

            # 动态扩充：基于苯环骨架的单/双取代，以及常见杂芳环/稠环/环胺
            substituents = [
                # 卤素/强吸电子
                "F", "Cl", "Br", "C(F)(F)F", "[N+](=O)[O-]", "C#N",
                # 含氧取代
                "OC", "OCC", "OCCC", "CO", "COC", "C(=O)O", "OC(F)(F)F",
                # 胺/酰胺/脲/磺酰胺
                "N", "NC", "CN", "C(=O)N", "NC(=O)C", "N(C)C",
                "S(=O)(=O)N", "S(=O)(=O)NC", "S(=O)(=O)C",
                # 疏水体积
                "C(C)(C)C",  # 叔丁基
            ]

            generated = set()
            # 单取代苯
            for x in substituents:
                generated.add(f"c1ccc({x})cc1")

            # 双取代苯（控制组合数量，避免爆炸）
            subs_for_di = substituents[:12]  # 取前12个常见取代基进行双取代组合
            for i, x in enumerate(subs_for_di):
                for j, y in enumerate(subs_for_di):
                    if j < i:
                        continue
                    generated.add(f"c1ccc({x})c({y})c1")

            # 常见杂芳环（不带取代）
            hetero_scaffolds = [
                "n1ccccc1",     # 吡啶
                "c1ccncc1",     # 吡啶异构
                "c1ncncc1",     # 嘧啶
                "c1ccoc1",      # 呋喃
                "c1ccsc1",      # 噻吩
            ]
            generated.update(hetero_scaffolds)

            # 杂芳环单取代（简单位置模板）
            for x in substituents:
                generated.add(f"c1cc({x})ncc1")   # 吡啶-取代
                generated.add(f"c1cc({x})oc1")    # 呋喃-取代
                generated.add(f"c1cc({x})sc1")    # 噻吩-取代

            # 稠环单取代（萘）
            fused_templates = ["c1ccc2ccccc2c1"]
            fused_subs = ["F", "Cl", "Br", "C(F)(F)F", "OC", "CO", "C(=O)N", "S(=O)(=O)N"]
            for tpl in fused_templates:
                for x in fused_subs:
                    generated.add(f"c1ccc2cc({x})ccc2c1")

            # 环胺类（已经在基础库，但再加入少量取代变体）
            ring_amines = ["N1CCCCC1", "N1CCOCC1", "O1CCNCC1", "O1CCOCC1", "N1CCNCC1"]
            generated.update(ring_amines)

            # o/m/p 位点控制的二取代苯（选择有限取代基对，系统性生成位置异构体）
            pos_subs = ["F", "Cl", "Br", "C(F)(F)F", "OC", "CO", "N", "C#N", "[N+](=O)[O-]", "C(=O)N", "S(=O)(=O)N"]
            for i, x in enumerate(pos_subs):
                for j, y in enumerate(pos_subs):
                    if j < i:
                        continue
                    # ortho (1,2-二取代)
                    generated.add(f"c1c({x})cccc1{y}")
                    # meta (1,3-二取代)
                    generated.add(f"c1cc({x})ccc1{y}")
                    # para (1,4-二取代)
                    generated.add(f"c1cc({x})cc({y})c1")

            # 更多骨架：联苯/三联苯、亚当烷、双环（简版）
            poly_aromatics = [
                "c1ccc(cc1)c2ccccc2",  # 联苯 biphenyl
                "c1ccc(cc1)c2ccc(cc2)c3ccccc3"  # 三联苯 terphenyl（线型）
            ]
            generated.update(poly_aromatics)

            # 在联苯上做简单单取代
            biphenyl = "c1ccc(cc1)c2ccccc2"
            for x in ["F", "Cl", "Br", "C(F)(F)F", "OC", "N", "C#N", "[N+](=O)[O-]"]:
                generated.add(f"c1ccc({x})cc1c2ccccc2")

            # 亚当烷 / 双环（诺尔波烷）
            alicyclics = [
                "C1C2CC3CC(C2)CC1C3",   # 亚当烷 adamantane（常见SMILES之一）
                "C1CC2CCC1C2"           # 诺尔波烷 norbornane（简化）
            ]
            generated.update(alicyclics)

            # 三取代苯（进一步扩充，限制取代基集合以控制组合规模）
            tri_subs = substituents[:8]
            for i, x in enumerate(tri_subs):
                for j, y in enumerate(tri_subs):
                    for k, z in enumerate(tri_subs):
                        if j < i or k < j:
                            continue
                        generated.add(f"c1c({x})cc({y})c({z})c1")

            # 更多稠环骨架：喹啉 / 异喹啉 / 吲哚（单取代）
            fused_scaffolds = [
                "c1ccc2ncccc2c1",   # 喹啉 quinoline
                "c1ccc2cccnc2c1",   # 异喹啉 isoquinoline
                "c1cc2ccccc2[nH]1"  # 吲哚 indole
            ]
            for tpl in fused_scaffolds:
                for x in fused_subs:
                    # 简单在稠环上挂一个取代（模板化，具体位点不完全精准，但能产生有效多样性）
                    # 这里采用在第二个环上某位点替换的简单形式
                    generated.add(tpl.replace("ccccc2", f"c{ 'c' if x else '' }c({x})cc2"))

            # 苯并唑类：吲唑/苯并噁唑/苯并噻唑/苯并咪唑（基础与少量单取代）
            benzo_azoles = [
                "c1ccc2[nH]ncc2c1",   # 吲唑 indazole（简化）
                "c1ccc2ocnc2c1",     # 苯并噁唑 benzoxazole
                "c1ccc2scnc2c1",     # 苯并噻唑 benzothiazole
                "c1ccc2[nH]c(nc2)c1" # 苯并咪唑 benzimidazole
            ]
            generated.update(benzo_azoles)
            for core in benzo_azoles:
                for x in ["F", "Cl", "OC", "N", "C(=O)N"]:
                    generated.add(core.replace("c1", f"c1{''}", 1) + f"{x}")  # 简易单取代（字符串拼接保守处理）

            # 五元唑环：咪唑/吡唑/噁唑/噻唑/1,2,4-三唑（基础与少量单取代）
            five_membered_azoles = [
                "n1c[nH]cc1",   # 咪唑 imidazole（简化）
                "n1nccc1",      # 吡唑 pyrazole（简化）
                "c1ocnc1",      # 噁唑 oxazole
                "c1scnc1",      # 噻唑 thiazole
                "n1nc[nH]1"     # 1,2,4-三唑 1,2,4-triazole（简化）
            ]
            generated.update(five_membered_azoles)
            for core in five_membered_azoles:
                for x in ["F", "Cl", "CO", "C(=O)N"]:
                    generated.add(core + f"{x}")

            # 三嗪/噻二唑（基础与少量单取代）
            other_hetero = [
                "n1cncnc1",   # 1,3,5-三嗪 triazine（简化芳香式）
                "c1nsnc1"     # 噻二唑 thiadiazole（简化）
            ]
            generated.update(other_hetero)
            for core in other_hetero:
                for x in ["F", "Cl", "N", "C#N", "[N+](=O)[O-]"]:
                    generated.add(core + f"{x}")

            # 联苯/三联苯：补充一些位点单取代
            for x in ["F", "Cl", "Br", "C(F)(F)F", "OC", "CO", "N", "C#N", "[N+](=O)[O-]"]:
                generated.add(f"c1c({x})ccc(c1)c2ccccc2")  # 在第一环上位点取代
                generated.add(f"c1ccc(cc1)c2c({x})cccc2")  # 在第二环上位点取代

            # 合并去重
            base_set = set(base_library)
            base_set.update(generated)
            # 控制最大库规模，避免内存过大
            max_dynamic = 8000
            if len(base_set) > max_dynamic:
                _rng_cap = np.random.default_rng()
                base_list = list(base_set)
                idx = _rng_cap.choice(len(base_list), size=max_dynamic, replace=False)
                base_library = [base_list[i] for i in idx]
            else:
                base_library = sorted(base_set)

            # 生成所需数量（允许重复抽样）
            rng = np.random.default_rng()
            chosen = list(rng.choice(base_library, size=total_num, replace=True))

            molecules: List[Dict] = []

            # 尝试使用RDKit计算性质
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, Crippen
                has_rdkit = True
            except Exception:
                has_rdkit = False

            for i, smi in enumerate(chosen, 1):
                mw = None
                logp = None
                if has_rdkit:
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            mw = float(Descriptors.MolWt(mol))
                            logp = float(Crippen.MolLogP(mol))
                    except Exception:
                        pass

                # 简单的亲和力估计：围绕目标值添加噪声，并对过大/过小的性质做轻微校正
                affinity = float(target_aff + rng.normal(0.0, 0.6))
                if mw is not None:
                    if mw > 500:
                        affinity += 0.5  # 过大分子，略差
                    elif mw < 150:
                        affinity += 0.3  # 过小分子，略差
                if logp is not None:
                    if logp > 5.0:
                        affinity += 0.4
                    elif logp < -1.0:
                        affinity += 0.2

                # 合理范围裁剪
                affinity = float(np.clip(affinity, -12.0, -3.0))

                molecules.append({
                    'compound_id': f'gen_{i:04d}',
                    'smiles': smi,
                    'binding_affinity': affinity,
                    'molecular_weight': mw if mw is not None else 'N/A',
                    'logp': logp if logp is not None else 'N/A',
                    'source': 'transformer_placeholder'
                })

            return molecules

        except Exception as e:
            logger.error(f"占位分子生成失败: {e}")
            return []

    def _external_transformer_generate(self, protein_pdb: str, targets: List[Dict]) -> List[Dict]:
        """调用外部Transformer推理脚本生成分子。脚本需输出包含smiles与可选属性的CSV。"""
        try:
            total_num = 0
            target_aff = -6.0
            if isinstance(targets, list) and targets:
                for t in targets:
                    total_num += int(t.get('num_molecules', 0))
                target_aff = float(targets[0].get('target_affinity', target_aff))
            if total_num <= 0:
                total_num = 50

            script_path = self.external.get('transformer_script')
            ckpt_path = self.external.get('transformer_checkpoint')
            if not script_path or not os.path.exists(script_path):
                logger.warning("未找到外部Transformer脚本，回退到占位生成")
                return self._generate_molecules_with_transformer(None, {}, targets)

            tmp_out = self.output_dir / f"external_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            cmd = [
                sys.executable, script_path,
                "--protein", protein_pdb,
                "--num", str(total_num),
                "--target_aff", str(target_aff),
                "--out", str(tmp_out)
            ]
            if ckpt_path:
                cmd += ["--checkpoint", ckpt_path]
            # 动态模块/函数/参数
            if self.external.get('tf_module'):
                cmd += ["--module", str(self.external.get('tf_module'))]
            if self.external.get('tf_function'):
                cmd += ["--function", str(self.external.get('tf_function'))]
            if self.external.get('tf_kwargs_json'):
                cmd += ["--kwargs_json", str(self.external.get('tf_kwargs_json'))]

            logger.info(f"运行外部Transformer脚本: {' '.join(cmd)}")
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"外部Transformer脚本失败: {result.stderr}")
                return self._generate_molecules_with_transformer(None, {}, targets)

            if tmp_out.exists():
                df = pd.read_csv(tmp_out)
                if 'smiles' in df.columns:
                    # 确保必要字段
                    if 'binding_affinity' not in df.columns:
                        df['binding_affinity'] = target_aff
                    df['compound_id'] = [f'gen_ext_{i+1:04d}' for i in range(len(df))]
                    keep = [c for c in ['compound_id','smiles','binding_affinity','molecular_weight','logp'] if c in df.columns]
                    return df[keep].to_dict(orient='records')
            logger.warning("外部Transformer脚本未产生有效输出，回退到占位生成")
            return self._generate_molecules_with_transformer(None, {}, targets)
        except Exception as e:
            logger.error(f"外部Transformer生成失败: {e}")
            return self._generate_molecules_with_transformer(None, {}, targets)
    
    def _screen_molecules_with_gnn(self, model, molecules, pocket_features) -> List[Dict]:
        """使用已训练的 EquivariantGNN 对候选分子打分并排序（binding_affinity 越负越好）"""
        if not molecules:
            return []
        try:
            self.model_configs['equivariant_gnn']  # 确保存在
        except Exception:
            return molecules

        # pocket_features 在此实现为蛋白图
        prot = pocket_features if isinstance(pocket_features, dict) else None
        if prot is None:
            logger.warning("筛选缺少蛋白图，将跳过GNN筛选")
            return molecules

        radius = float(self.model_configs['equivariant_gnn'].max_radius)
        # 构建小批量以提速推理
        batch_size = 32
        scored: List[Dict] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(molecules), batch_size):
                batch_mols = molecules[start:start+batch_size]
                samples = []
                for m in batch_mols:
                    smi = str(m.get('smiles', ''))
                    if not smi:
                        continue
                    lig = self._build_ligand_graph(smi, radius=radius)
                    merged = self._merge_protein_ligand(prot, lig, cross_radius=radius)
                    samples.append(merged)
                if not samples:
                    continue
                # 拼批
                node_offset = 0
                atom_types_list = []
                positions_list = []
                edge_indices = []
                batch_vec = []
                for i, s in enumerate(samples):
                    n = s['atom_types'].size(0)
                    atom_types_list.append(s['atom_types'])
                    positions_list.append(s['positions'])
                    edge_indices.append(s['edge_index'] + node_offset)
                    batch_vec.append(torch.full((n,), i, dtype=torch.long, device=self.device))
                    node_offset += n
                atom_types = torch.cat(atom_types_list, dim=0)
                positions = torch.cat(positions_list, dim=0)
                edge_index = torch.cat(edge_indices, dim=1)
                batch_tensor = torch.cat(batch_vec, dim=0)

                outputs = model(atom_types=atom_types, positions=positions, edge_index=edge_index, batch=batch_tensor)
                scores = outputs['binding_affinity'].detach().cpu().view(-1).numpy().tolist()
                for m, s in zip(batch_mols, scores):
                    mm = dict(m)
                    mm['pred_binding_affinity'] = float(s)
                    scored.append(mm)

        # 排序（更负更好）
        scored = sorted(scored, key=lambda x: x.get('pred_binding_affinity', 0.0))
        return scored
    
    def _md_refinement(self, molecules: List[Dict], protein_pdb: str) -> List[Dict]:
        """分子动力学精修"""
        # 简化实现
        return molecules
    
    def _multi_objective_optimization(self, base_molecules: pd.DataFrame, protein_pdb: str, targets: Dict) -> List[Dict]:
        """多目标优化：基于 RDKit 性质 + 亲和力的帕累托筛选与加权排序（无RL/无Diffusion）。
        目标（默认权重，可由 targets 覆盖）：
          - 亲和力（binding_affinity 或 pred_binding_affinity）：越负越好
          - QED：越高越好
          - SA（合成难度）：越低越好
          - MW 距离窗口 [200,500] 越近越好
          - LogP 距离窗口 [-1,4.5] 越近越好
        返回：Top-K（默认50）优化候选。
        """
        try:
            # 读取/就地 DataFrame
            if isinstance(base_molecules, pd.DataFrame):
                df = base_molecules.copy()
            else:
                df = pd.read_csv(base_molecules)

            # RDKit 可选
            has_rdkit = False
            try:
                from rdkit import Chem
                from rdkit.Chem import Crippen, Descriptors, QED
                from rdkit.Contrib.SA_Score import sascorer
                has_rdkit = True
            except Exception:
                has_rdkit = False

            # 计算性质
            if has_rdkit:
                props = {
                    'molecular_weight': [], 'logp': [], 'qed': [], 'sa_score': []
                }
                for smi in df['smiles'].astype(str).tolist():
                    mw = logp = qed = sa = None
                    try:
                        m = Chem.MolFromSmiles(smi)
                        if m is not None:
                            mw = float(Descriptors.MolWt(m))
                            logp = float(Crippen.MolLogP(m))
                            qed = float(QED.qed(m))
                            sa = float(sascorer.calculateScore(m))
                    except Exception:
                        pass
                    props['molecular_weight'].append(mw)
                    props['logp'].append(logp)
                    props['qed'].append(qed)
                    props['sa_score'].append(sa)
                for k, v in props.items():
                    df[k] = v

            # 归一化/目标函数
            aff = df.get('binding_affinity', df.get('pred_binding_affinity', pd.Series([-6.0]*len(df))))
            # 亲和力越负越好 → aff_score = -aff
            df['_obj_aff'] = -pd.to_numeric(aff, errors='coerce').fillna(0.0)

            # QED 越高越好
            df['_obj_qed'] = pd.to_numeric(df.get('qed', pd.Series([0.0]*len(df))), errors='coerce').fillna(0.0)

            # SA 越低越好 → 目标取 (max_sa - sa)，归一化
            sa = pd.to_numeric(df.get('sa_score', pd.Series([None]*len(df))), errors='coerce')
            max_sa = float(sa.max()) if sa.notna().any() else 10.0
            df['_obj_sa'] = sa.apply(lambda x: (max_sa - x) if pd.notna(x) else 0.0)

            # MW 距离窗口 [200,500] 越近越好 → 目标 = -距离
            mw = pd.to_numeric(df.get('molecular_weight', pd.Series([None]*len(df))), errors='coerce')
            def dist_window(x, lo, hi):
                if pd.isna(x):
                    return None
                if lo <= x <= hi:
                    return 0.0
                return min(abs(x-lo), abs(x-hi))
            df['_obj_mw'] = mw.apply(lambda x: -dist_window(x, 200.0, 500.0) if x is not None else None).fillna(-1.0)

            # LogP 距离窗口 [-1,4.5] 越近越好 → 目标 = -距离
            logp = pd.to_numeric(df.get('logp', pd.Series([None]*len(df))), errors='coerce')
            df['_obj_logp'] = logp.apply(lambda x: -dist_window(x, -1.0, 4.5) if x is not None else None).fillna(-1.0)

            # 非支配排序（简单实现）
            objectives = ['_obj_aff','_obj_qed','_obj_sa','_obj_mw','_obj_logp']
            vals = df[objectives].values
            n = len(df)
            ranks = [None]*n
            dominated_count = [0]*n
            dominates = [set() for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    vi, vj = vals[i], vals[j]
                    if np.all(vi >= vj) and np.any(vi > vj):
                        dominates[i].add(j)
                    elif np.all(vj >= vi) and np.any(vj > vi):
                        dominated_count[i] += 1
            fronts = []
            current = [i for i,c in enumerate(dominated_count) if c==0]
            r = 0
            while current:
                fronts.append(current)
                next_front = []
                for p in current:
                    ranks[p] = r
                    for q in dominates[p]:
                        dominated_count[q] -= 1
                        if dominated_count[q] == 0:
                            next_front.append(q)
                current = next_front
                r += 1
            df['_pareto_rank'] = ranks

            # 前沿内加权排序（可由targets覆盖权重）
            w = targets.get('weights', {}) if isinstance(targets, dict) else {}
            w_aff = float(w.get('affinity', 1.0))
            w_qed = float(w.get('qed', 1.0))
            w_sa  = float(w.get('sa', 1.0))
            w_mw  = float(w.get('mw', 0.5))
            w_logp= float(w.get('logp', 0.5))
            df['_score'] = (
                w_aff*df['_obj_aff'] + w_qed*df['_obj_qed'] + w_sa*df['_obj_sa'] +
                w_mw*df['_obj_mw'] + w_logp*df['_obj_logp']
            )

            # 去重、排序与Top-K
            if has_rdkit:
                try:
                    from rdkit import Chem
                    df['can_smiles'] = df['smiles'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if s else s)
                    df = df.drop_duplicates(subset=['can_smiles']).drop(columns=['can_smiles'])
                except Exception:
                    pass

            df = df.sort_values(by=['_pareto_rank','_score'], ascending=[True, False])
            top_k = int(targets.get('top_k', 50)) if isinstance(targets, dict) else 50
            df = df.head(top_k)

            # 选择输出列
            keep_cols = [c for c in ['compound_id','smiles','binding_affinity','pred_binding_affinity','molecular_weight','logp','qed','sa_score'] if c in df.columns]
            df['_source'] = 'moo_selection'
            res = df[keep_cols + ['_pareto_rank','_score','_source']].to_dict(orient='records')
            return res
        except Exception as e:
            logger.error(f"多目标优化失败: {e}")
            return []

    
    def _active_learning_loop(self, molecules, protein_pdb, targets):
        """主动学习闭环（占位实现：去重、排序并保留Top-N）"""
        if not molecules:
            return []
        try:
            df = pd.DataFrame(molecules)
            # 去重
            df = df.drop_duplicates(subset=['smiles'])
            # 排序（binding_affinity 越负越好）
            df = df.sort_values(by='binding_affinity', ascending=True)
            top_n = int(targets.get('final_top_n', 50)) if isinstance(targets, dict) else 50
            df = df.head(top_n)
            # 重新编号
            df = df.reset_index(drop=True)
            df['compound_id'] = [f'final_{i+1:04d}' for i in range(len(df))]
            return df.to_dict(orient='records')
        except Exception as e:
            logger.error(f"主动学习闭环失败: {e}")
            return molecules
    
    def _generate_final_report(self, molecules, protein_pdb) -> str:
        """生成最终报告"""
        report_path = self.output_dir / "final_report.html"
        # 简化实现
        with open(report_path, 'w') as f:
            f.write("<html><body><h1>深度学习流水线最终报告</h1></body></html>")
        return str(report_path)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PRRSV抑制剂设计深度学习流水线")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--protein", type=str, default="data/1p65.pdb", help="蛋白质PDB文件")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="运行特定阶段")
    parser.add_argument("--output", type=str, default="deep_learning_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = DeepLearningPipeline(args.config)
    
    # 示例数据
    initial_ligand_data = [
        {"smiles": "CCO", "binding_affinity": -5.2},
        {"smiles": "CCN", "binding_affinity": -4.8}
    ]
    
    generation_targets = [
        {"target_affinity": -6.0, "num_molecules": 100}
    ]
    
    optimization_targets = {
        "binding_affinity": -7.0,
        "drug_likeness": 0.8,
        "synthesizability": 0.7
    }
    
    if args.phase == 1:
        pipeline.phase_1_equivariant_gnn(args.protein, initial_ligand_data)
    elif args.phase == 2:
        phase1_model = "deep_learning_results/phase1_equivariant_gnn.pth"
        pipeline.phase_2_transformer_generation(phase1_model, args.protein, generation_targets)
    elif args.phase == 3:
        phase2_data = "deep_learning_results/phase2_generated_molecules.csv"
        pipeline.phase_3_diffusion_and_rl(phase2_data, args.protein, optimization_targets)
    else:
        # 运行完整流水线
        results = pipeline.run_full_pipeline(
            args.protein, initial_ligand_data, generation_targets, optimization_targets
        )
        print("流水线执行完成！")
        print(f"结果: {results}")

if __name__ == "__main__":
    main()
