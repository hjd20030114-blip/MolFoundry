#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于目标口袋的配体生成器
根据蛋白质口袋特征生成或筛选小分子抑制剂
"""

import os
import random
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from .config import PROJECT_ROOT
from .structure_library import StructureLibrary

# 设置日志
logger = logging.getLogger(__name__)

# RDKit导入处理
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
    from rdkit.Chem import rdMolDescriptors, rdDepictor
    RDKIT_AVAILABLE = True
    logger.info("RDKit已成功导入，将使用完整配体生成功能")
except ImportError as e:
    logger.warning(f"RDKit不可用，将使用简化模式: {e}")

class PocketBasedLigandGenerator:
    """基于目标口袋的配体生成器"""
    
    def __init__(self, pocket_info: Dict = None):
        """初始化配体生成器"""
        self.pocket_info = pocket_info
        self.generated_ligands = []

        # 基于口袋特征的分子模板
        self.pocket_templates = self._load_pocket_specific_templates()

        # 预生成的结构库
        self.structure_library = StructureLibrary()

        logger.info("基于口袋的配体生成器初始化完成")
    
    def _load_pocket_specific_templates(self) -> Dict[str, List[str]]:
        """加载针对不同口袋特征的分子模板"""
        templates = {
            # 静电相互作用主导的口袋
            'electrostatic': [
                "c1ccc(cc1)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # 天冬氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](CCCCN)C(=O)O",     # 赖氨酸类似物
                "c1ccc(cc1)S(=O)(=O)N[C@@H](CC(=O)O)C(=O)O", # 磺酰胺类
                "NC(=N)NCCC[C@H](N)C(=O)O",                # 精氨酸类似物
                "c1ccc2c(c1)c(C(=O)O)c(N)n2C",             # 色氨酸衍生物
            ],
            
            # 疏水相互作用主导的口袋
            'hydrophobic': [
                "CCc1ccc(cc1)C(C)C(=O)N[C@@H](Cc2ccccc2)C(=O)O",  # 疏水氨基酸类似物
                "c1ccc(cc1)CCc2ccc(cc2)C(=O)N[C@@H](CC(C)C)C(=O)O", # 双苯基化合物
                "CCCCc1ccc(cc1)C(=O)N[C@@H](CC(C)C)C(=O)O",        # 长链疏水化合物
                "c1ccc2c(c1)ccc3c2ccc(c3)C(=O)N[C@@H](CC(C)C)C(=O)O", # 萘类化合物
                "CCc1ccc(cc1)Oc2ccc(cc2)C(=O)N[C@@H](CC(C)C)C(=O)O",  # 醚键连接的疏水化合物
            ],
            
            # 氢键相互作用主导的口袋
            'hydrogen_bond': [
                "c1ccc(cc1)C(=O)N[C@@H](CO)C(=O)O",        # 丝氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](C(C)O)C(=O)O",     # 苏氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](CC(=O)N)C(=O)O",   # 天冬酰胺类似物
                "c1ccc(cc1)C(=O)N[C@@H](CCC(=O)N)C(=O)O",  # 谷氨酰胺类似物
                "c1ccc(cc1)C(=O)N[C@@H](Cc2c[nH]c3ccccc23)C(=O)O", # 色氨酸类似物
            ],
            
            # 范德华力主导的口袋
            'van_der_waals': [
                "c1ccc(cc1)C(=O)N[C@@H](C)C(=O)O",         # 丙氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](CC)C(=O)O",        # 缬氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](CCC)C(=O)O",       # 亮氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](C(C)CC)C(=O)O",    # 异亮氨酸类似物
                "c1ccc(cc1)C(=O)N[C@@H](CCSC)C(=O)O",      # 蛋氨酸类似物
            ],
            
            # 小分子口袋（体积<500）- 使用更复杂的分子
            'small_pocket': [
                "COc1ccc(cc1)C(=O)N[C@@H](C)C(=O)O",       # 甲氧基苯甲酰丙氨酸
                "Cc1ccc(cc1)C(=O)N[C@@H](CC)C(=O)O",       # 甲苯酰缬氨酸
                "c1ccc(cc1)C(=O)N[C@@H](CO)C(=O)O",        # 苯甲酰丝氨酸
                "COc1ccc(cc1)CCN[C@@H](C)C(=O)O",          # 甲氧基苯乙胺丙氨酸
                "Cc1ccc(cc1)S(=O)(=O)N[C@@H](C)C(=O)O",    # 甲苯磺酰丙氨酸
                "c1ccc(cc1)OC[C@@H](N)C(=O)O",             # 苯氧基丙氨酸
                "COc1ccc(cc1)C(=O)OC[C@@H](N)C(=O)O",      # 甲氧基苯甲酸酯丙氨酸
            ],
            
            # 大分子口袋（体积>1500）
            'large_pocket': [
                "c1ccc(cc1)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CC(=O)O)C(=O)O", # 二肽
                "c1ccc2c(c1)ccc3c2ccc4c3ccc5c4cccc5C(=O)O",                      # 大环芳香化合物
                "CCCCCCCCc1ccc(cc1)C(=O)N[C@@H](Cc2ccccc2)C(=O)O",              # 长链化合物
                "c1ccc(cc1)COc2ccc(cc2)COc3ccc(cc3)C(=O)O",                     # 多环醚化合物
            ]
        }
        
        return templates
    
    def generate_pocket_specific_ligands(self, num_ligands: int = 50) -> List[Dict]:
        """基于口袋特征生成特异性配体"""
        try:
            if not self.pocket_info:
                logger.warning("未提供口袋信息，使用通用配体生成")
                return self._generate_generic_ligands(num_ligands)

            logger.info(f"基于口袋特征生成 {num_ligands} 个配体")

            ligands = []

            # 优先使用结构库中的分子（占70%）
            library_count = int(num_ligands * 0.7)
            library_ligands = self._generate_from_structure_library(library_count)
            ligands.extend(library_ligands)

            # 剩余部分使用模板生成（占30%）
            template_count = num_ligands - len(ligands)
            if template_count > 0:
                template_ligands = self._generate_from_templates(template_count)
                ligands.extend(template_ligands)

            self.generated_ligands = ligands
            logger.info(f"成功生成 {len(ligands)} 个口袋特异性配体")

            return ligands

        except Exception as e:
            logger.error(f"生成口袋特异性配体时出错: {e}")
            return []

    def _generate_from_structure_library(self, count: int) -> List[Dict]:
        """从结构库生成配体"""
        try:
            # 获取主要相互作用类型
            interaction_types = self.pocket_info.get('interaction_types', {})
            dominant_interaction = max(interaction_types, key=interaction_types.get) if interaction_types else 'van_der_waals'

            # 根据口袋体积确定分子量范围
            volume = self.pocket_info.get('volume', 1000)
            if volume < 500:
                min_mw, max_mw = 150, 250
            elif volume > 1500:
                min_mw, max_mw = 250, 400
            else:
                min_mw, max_mw = 180, 300

            # 从结构库筛选合适的分子
            filtered_molecules = self.structure_library.get_filtered_molecules(
                interaction_type=dominant_interaction,
                min_mw=min_mw,
                max_mw=max_mw
            )

            if not filtered_molecules:
                # 如果没有匹配的分子，使用所有分子
                filtered_molecules = self.structure_library.get_molecules_by_size(min_mw, max_mw)

            if not filtered_molecules:
                # 如果仍然没有，使用随机分子
                filtered_molecules = self.structure_library.get_random_molecules(count)

            # 随机选择并转换为配体格式
            ligands = []
            selected_molecules = random.sample(filtered_molecules, min(count, len(filtered_molecules)))

            for mol_data in selected_molecules:
                ligand = {
                    'smiles': mol_data['smiles'],
                    'template': mol_data['name'],
                    'molecular_weight': mol_data['mw'],
                    'logp': mol_data['logp'],
                    'hbd': self._estimate_hbd(mol_data['smiles']),
                    'hba': self._estimate_hba(mol_data['smiles']),
                    'rotatable_bonds': self._estimate_rotatable_bonds(mol_data['smiles']),
                    'tpsa': self._estimate_tpsa(mol_data['mw']),
                    'pocket_compatibility': self._assess_pocket_compatibility_simple(mol_data)
                }
                ligands.append(ligand)

            # 如果数量不够，重复选择
            while len(ligands) < count and filtered_molecules:
                additional = random.choice(filtered_molecules)
                ligand = {
                    'smiles': additional['smiles'],
                    'template': additional['name'],
                    'molecular_weight': additional['mw'],
                    'logp': additional['logp'],
                    'hbd': self._estimate_hbd(additional['smiles']),
                    'hba': self._estimate_hba(additional['smiles']),
                    'rotatable_bonds': self._estimate_rotatable_bonds(additional['smiles']),
                    'tpsa': self._estimate_tpsa(additional['mw']),
                    'pocket_compatibility': self._assess_pocket_compatibility_simple(additional)
                }
                ligands.append(ligand)

            logger.info(f"从结构库生成了 {len(ligands)} 个配体")
            return ligands

        except Exception as e:
            logger.error(f"从结构库生成配体时出错: {e}")
            return []

    def _generate_from_templates(self, count: int) -> List[Dict]:
        """从模板生成配体"""
        ligands = []
        attempts = 0
        max_attempts = count * 3

        while len(ligands) < count and attempts < max_attempts:
            attempts += 1

            # 根据口袋特征选择模板
            template = self._select_template_for_pocket()

            if not template:
                continue

            # 生成配体
            ligand = self._generate_ligand_from_template(template)

            if ligand and self._validate_ligand_for_pocket(ligand):
                ligands.append(ligand)

        logger.info(f"从模板生成了 {len(ligands)} 个配体")
        return ligands

    def _estimate_hbd(self, smiles: str) -> int:
        """估算氢键供体数量"""
        return smiles.count('OH') + smiles.count('NH')

    def _estimate_hba(self, smiles: str) -> int:
        """估算氢键受体数量"""
        return smiles.count('O') + smiles.count('N')

    def _estimate_rotatable_bonds(self, smiles: str) -> int:
        """估算可旋转键数量"""
        return smiles.count('C-C') + smiles.count('C-O') + smiles.count('C-N')

    def _estimate_tpsa(self, mw: float) -> float:
        """估算极性表面积"""
        return mw * 0.3  # 简化估算

    def _assess_pocket_compatibility_simple(self, mol_data: Dict) -> float:
        """简化的口袋兼容性评估"""
        if not self.pocket_info:
            return 0.5

        compatibility_score = 0.0

        # 基于口袋体积的分子大小兼容性
        volume = self.pocket_info.get('volume', 1000)
        mw = mol_data['mw']

        if volume < 500 and mw < 250:
            compatibility_score += 0.4
        elif 500 <= volume <= 1500 and 180 <= mw <= 300:
            compatibility_score += 0.4
        elif volume > 1500 and mw > 250:
            compatibility_score += 0.4

        # 基于相互作用类型的兼容性
        interaction_types = self.pocket_info.get('interaction_types', {})
        mol_interaction = mol_data.get('interaction_type', 'van_der_waals')

        if mol_interaction in interaction_types and interaction_types[mol_interaction] > 0:
            compatibility_score += 0.3

        # LogP兼容性
        logp = mol_data['logp']
        if 0 <= logp <= 4:
            compatibility_score += 0.3

        return min(1.0, compatibility_score)

    def _select_template_for_pocket(self) -> Optional[str]:
        """根据口袋特征选择合适的模板"""
        if not self.pocket_info:
            return None
        
        # 获取主要相互作用类型
        interaction_types = self.pocket_info.get('interaction_types', {})
        dominant_interaction = max(interaction_types, key=interaction_types.get) if interaction_types else 'van_der_waals'
        
        # 根据口袋体积调整模板选择
        volume = self.pocket_info.get('volume', 1000)
        
        if volume < 500:
            template_key = 'small_pocket'
        elif volume > 1500:
            template_key = 'large_pocket'
        else:
            template_key = dominant_interaction
        
        # 选择模板
        templates = self.pocket_templates.get(template_key, self.pocket_templates['van_der_waals'])
        return random.choice(templates)
    
    def _generate_ligand_from_template(self, template: str) -> Optional[Dict]:
        """从模板生成配体"""
        try:
            if not RDKIT_AVAILABLE:
                return self._generate_simple_ligand(template)
            
            # 使用RDKit处理模板
            mol = Chem.MolFromSmiles(template)
            if mol is None:
                return None
            
            # 随机修饰分子
            modified_mol = self._modify_molecule(mol)
            if modified_mol is None:
                return None
            
            # 计算分子性质
            smiles = Chem.MolToSmiles(modified_mol)
            properties = self._calculate_properties(modified_mol)
            
            ligand = {
                'smiles': smiles,
                'template': template,
                'molecular_weight': properties['molecular_weight'],
                'logp': properties['logp'],
                'hbd': properties['hbd'],
                'hba': properties['hba'],
                'rotatable_bonds': properties['rotatable_bonds'],
                'tpsa': properties['tpsa'],
                'pocket_compatibility': self._assess_pocket_compatibility(properties)
            }
            
            return ligand
            
        except Exception as e:
            logger.debug(f"从模板生成配体时出错: {e}")
            return None
    
    def _modify_molecule(self, mol):
        """随机修饰分子"""
        try:
            # 简单的分子修饰策略
            modifications = [
                self._add_methyl_group,
                self._add_hydroxyl_group,
                self._add_fluorine,
                self._add_methoxy_group
            ]
            
            # 随机选择修饰方法
            if random.random() < 0.3:  # 30%概率进行修饰
                modification = random.choice(modifications)
                return modification(mol)
            
            return mol
            
        except Exception as e:
            logger.debug(f"修饰分子时出错: {e}")
            return mol
    
    def _add_methyl_group(self, mol):
        """添加甲基"""
        # 简化实现：返回原分子
        return mol
    
    def _add_hydroxyl_group(self, mol):
        """添加羟基"""
        # 简化实现：返回原分子
        return mol
    
    def _add_fluorine(self, mol):
        """添加氟原子"""
        # 简化实现：返回原分子
        return mol
    
    def _add_methoxy_group(self, mol):
        """添加甲氧基"""
        # 简化实现：返回原分子
        return mol
    
    def _calculate_properties(self, mol) -> Dict:
        """计算分子性质"""
        try:
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms()
            }
            return properties
        except Exception as e:
            logger.debug(f"计算分子性质时出错: {e}")
            return {
                'molecular_weight': 0,
                'logp': 0,
                'hbd': 0,
                'hba': 0,
                'rotatable_bonds': 0,
                'tpsa': 0,
                'aromatic_rings': 0,
                'heavy_atoms': 0
            }
    
    def _assess_pocket_compatibility(self, properties: Dict) -> float:
        """评估与口袋的兼容性"""
        if not self.pocket_info:
            return 0.5
        
        compatibility_score = 0.0
        
        # 基于口袋体积的分子大小兼容性
        volume = self.pocket_info.get('volume', 1000)
        mw = properties['molecular_weight']
        
        if volume < 500 and mw < 300:
            compatibility_score += 0.3
        elif 500 <= volume <= 1500 and 300 <= mw <= 500:
            compatibility_score += 0.3
        elif volume > 1500 and mw > 400:
            compatibility_score += 0.3
        
        # 基于相互作用类型的兼容性
        interaction_types = self.pocket_info.get('interaction_types', {})
        
        if interaction_types.get('electrostatic', 0) > 0:
            if properties['hbd'] > 0 or properties['hba'] > 0:
                compatibility_score += 0.2
        
        if interaction_types.get('hydrophobic', 0) > 0:
            if properties['logp'] > 2:
                compatibility_score += 0.2
        
        if interaction_types.get('hydrogen_bond', 0) > 0:
            if properties['hbd'] + properties['hba'] > 2:
                compatibility_score += 0.2
        
        # Lipinski规则兼容性
        if (mw <= 500 and properties['logp'] <= 5 and 
            properties['hbd'] <= 5 and properties['hba'] <= 10):
            compatibility_score += 0.1
        
        return min(1.0, compatibility_score)
    
    def _validate_ligand_for_pocket(self, ligand: Dict) -> bool:
        """验证配体是否适合目标口袋"""
        if not ligand:
            return False
        
        # 基本验证
        if ligand['molecular_weight'] < 100 or ligand['molecular_weight'] > 800:
            return False
        
        if ligand['pocket_compatibility'] < 0.3:
            return False
        
        return True
    
    def _generate_simple_ligand(self, template: str) -> Optional[Dict]:
        """简化模式下生成配体"""
        return {
            'smiles': template,
            'template': template,
            'molecular_weight': 250.0,
            'logp': 2.5,
            'hbd': 2,
            'hba': 3,
            'rotatable_bonds': 5,
            'tpsa': 60.0,
            'pocket_compatibility': 0.7
        }
    
    def _generate_generic_ligands(self, num_ligands: int) -> List[Dict]:
        """生成通用配体"""
        logger.info("生成通用配体")
        
        # 使用所有模板
        all_templates = []
        for templates in self.pocket_templates.values():
            all_templates.extend(templates)
        
        ligands = []
        for i in range(num_ligands):
            template = random.choice(all_templates)
            ligand = self._generate_ligand_from_template(template)
            if ligand:
                ligands.append(ligand)
        
        return ligands
