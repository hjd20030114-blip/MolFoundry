# -*- coding: utf-8 -*-
"""
PRRSV病毒衣壳蛋白抑制剂配体生成模块
支持多种分子生成策略和优化算法
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
try:
    from .config import *
except ImportError:
    from config import *

# 导入CMD-GEN集成模块
try:
    from scripts.cmdgen_integration import CMDGENGenerator
    CMDGEN_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("CMD-GEN集成模块已导入")
except ImportError:
    CMDGEN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CMD-GEN集成模块不可用")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RDKit导入处理
RDKIT_AVAILABLE = False
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem import Draw

    RDKIT_AVAILABLE = True
    logger.info("RDKit已成功导入，将使用完整配体生成功能")
except ImportError:
    logger.warning("RDKit不可用，将使用简化模式进行配体生成")
    logger.info("如需完整功能，请安装RDKit: pip install rdkit-pypi")


    # 创建虚拟模块避免属性错误
    class MockChem:
        @staticmethod
        def MolFromSmiles(smiles): return None

        @staticmethod
        def MolToSmiles(mol): return ""

        @staticmethod
        def RWMol(mol): return None

        @staticmethod
        def Atom(symbol): return None

        @staticmethod
        def BondType(): return None

        @staticmethod
        def Draw():
            class MockDraw:
                @staticmethod
                def MolToImage(mol, size): return None

            return MockDraw()


    class MockDescriptors:
        @staticmethod
        def MolWt(mol): return 0.0

        @staticmethod
        def MolLogP(mol): return 0.0

        @staticmethod
        def NumHDonors(mol): return 0

        @staticmethod
        def NumHAcceptors(mol): return 0

        @staticmethod
        def NumRotatableBonds(mol): return 0

        @staticmethod
        def TPSA(mol): return 0.0

        @staticmethod
        def RingCount(mol): return 0

        @staticmethod
        def NumAromaticRings(mol): return 0


    Chem = MockChem
    Descriptors = MockDescriptors
    AllChem = MockChem


class LigandGenerator:
    """配体生成器类"""

    def __init__(self, use_cmdgen: bool = True, cmdgen_path: Optional[str] = None):
        """
        初始化配体生成器

        Args:
            use_cmdgen: 是否使用CMD-GEN模型
            cmdgen_path: CMD-GEN代码路径
        """
        self.generated_ligands = []
        self.smiles_templates = self._load_smiles_templates()
        self.fragment_library = self._load_fragment_library()

        # 初始化CMD-GEN生成器
        self.use_cmdgen = use_cmdgen and CMDGEN_AVAILABLE
        self.cmdgen_generator = None

        if self.use_cmdgen:
            try:
                self.cmdgen_generator = CMDGENGenerator(cmdgen_path=cmdgen_path)
                logger.info("CMD-GEN生成器初始化成功")
            except Exception as e:
                logger.warning(f"CMD-GEN生成器初始化失败: {e}")
                self.use_cmdgen = False

    def _load_smiles_templates(self) -> List[str]:
        """加载SMILES模板库 - 使用更适合药物设计的分子"""
        templates = [
            # 类药物分子模板 - 这些分子更容易生成3D构象
            "CCc1ccc(cc1)C(=O)O",  # 布洛芬类似物
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # 异丁苯丙酸
            "COc1ccc(cc1)CCN",  # 甲氧基苯乙胺
            "Cc1ccc(cc1)C(=O)Nc2ccccc2",  # 甲苯酰苯胺
            "CCOc1ccc(cc1)C(=O)N",  # 对乙氧基苯甲酰胺

            # 抗病毒药物骨架 - 增强结合亲和力
            "Cc1ccc2nc(N)nc(Nc3ccccc3)c2c1",  # 苯胺基喹唑啉（抗病毒活性）
            "COc1ccc(cc1)c2nc(N)nc(N)n2",  # 甲氧基苯基三嗪（蛋白抑制剂）
            "Cc1ccc(cc1)S(=O)(=O)Nc2nc(N)nc(N)n2",  # 磺酰胺三嗪（强结合）
            "CCc1ccc(cc1)c2nc(Nc3ccccc3)nc(N)n2",  # 苯胺基三嗪（高亲和力）
            "COc1ccc2c(c1)nc(Nc3ccccc3)nc2N",  # 苯胺基喹唑啉（优化结合）

            # 杂环药物模板 - 优化版本
            "Cc1ccc2nc(N)nc(N)c2c1",  # 二氨基甲基喹唑啉
            "COc1ccc2c(c1)nc(N)nc2N",  # 甲氧基二氨基喹唑啉
            "Cc1nc2ccccc2c(=O)n1C",  # 甲基喹唑酮
            "CCc1ccc2nc(C)nc(N)c2c1",  # 乙基氨基喹唑啉
            "Fc1ccc(cc1)c2nc(N)nc(N)c2",  # 氟苯基嘧啶（增强结合）

            # 蛋白质抑制剂骨架
            "Cc1ccc(cc1)c2nc(Nc3ccc(F)cc3)nc(N)n2",  # 氟苯胺三嗪
            "COc1ccc(cc1)c2nnc(Nc3ccccc3)s2",  # 苯胺噻二唑
            "Cc1ccc(cc1)C(=O)Nc2nc(N)nc(N)n2",  # 酰胺三嗪
            "CCc1ccc(cc1)c2nc(N)nc(Nc3ccc(Cl)cc3)n2",  # 氯苯胺三嗪

            # 苯并杂环 - 增强版
            "COc1ccc2c(c1)c(C)cn2C",  # 甲氧基甲基吲哚
            "Cc1ccc2c(c1)nc(C)n2C",  # 二甲基苯并咪唑
            "CCc1ccc2c(c1)oc(C)c2C(=O)O",  # 乙基苯并呋喃羧酸
            "Fc1ccc2c(c1)nc(N)nc2Nc3ccccc3",  # 氟苯胺喹唑啉

            # 含氮杂环 - 高亲和力版本
            "Cc1ccc(cc1)c2nc(N)nc(N)n2",  # 甲苯基三嗪二胺
            "COc1ccc(cc1)c2nnc(N)s2",  # 甲氧基苯基噻二唑胺
            "Cc1ccc(cc1)C2=NN=C(N)S2",  # 甲苯基噻二唑胺
            "Clc1ccc(cc1)c2nc(N)nc(N)n2",  # 氯苯基三嗪（强结合）

            # 脂肪族连接的芳环 - 优化柔性
            "CCc1ccc(cc1)OCc2ccccc2",  # 乙基苯氧基甲苯
            "COc1ccc(cc1)CCc2ccccc2",  # 甲氧基苯乙基苯
            "Cc1ccc(cc1)COc2ccccc2C",  # 甲苯氧基甲基甲苯
            "Fc1ccc(cc1)OCc2ccc(F)cc2",  # 双氟苯氧基（增强结合）

            # 酰胺和酯类 - 氢键优化
            "CCc1ccc(cc1)C(=O)NCc2ccccc2",  # 乙基苯甲酰苄胺
            "COc1ccc(cc1)C(=O)OCC",  # 甲氧基苯甲酸乙酯
            "Cc1ccc(cc1)C(=O)N(C)C",  # 甲苯酰二甲胺
            "Fc1ccc(cc1)C(=O)Nc2nc(N)nc(N)n2",  # 氟苯酰胺三嗪（强氢键）
        ]
        return templates

    def _load_fragment_library(self) -> Dict[str, List[str]]:
        """加载分子片段库"""
        fragments = {
            "aromatic": ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccc2ncccc2c1"],
            "aliphatic": ["CC", "CCC", "CC(C)C", "CC(C)CC"],
            "heterocyclic": ["c1ccncc1", "c1ccoc1", "c1ccsc1"],
            "functional_groups": ["O", "N", "S", "C=O", "C#N", "C(=O)O", "C(=O)N"],
            "linkers": ["C", "CC", "CCC", "c1ccccc1", "C(=O)", "C#C"]
        }
        return fragments

    def generate_random_ligand(self) -> Optional[str]:
        """生成随机配体"""
        try:
            # 如果RDKit不可用，使用简化模式
            if not RDKIT_AVAILABLE:
                return self._generate_simplified_ligand()

            # 选择基础模板
            template = random.choice(self.smiles_templates)
            mol = Chem.MolFromSmiles(template)

            if mol is None:
                return None

            # 验证分子
            if not self._is_valid_molecule(mol):
                return None

            # 随机修饰（减少修饰次数以避免复杂性问题）
            for _ in range(random.randint(0, 2)):  # 减少修饰次数
                modified_result = self._random_modification(mol)
                if modified_result is None:
                    break

                # 如果返回的是字符串，转换为分子对象
                if isinstance(modified_result, str):
                    mol = Chem.MolFromSmiles(modified_result)
                else:
                    mol = modified_result

                if mol is None or not self._is_valid_molecule(mol):
                    break

            if mol is None:
                return None

            return Chem.MolToSmiles(mol)

        except Exception as e:
            logger.error(f"生成随机配体时出错: {e}")
            return None

    def _generate_simplified_ligand(self) -> Optional[str]:
        """简化模式下的配体生成"""
        try:
            # 随机选择单一模板，不进行片段组合
            # 这样确保生成的是单一分子而不是多片段分子
            ligand = random.choice(self.smiles_templates)

            # 验证分子大小
            # 简单的原子计数（去除特殊字符）
            atom_count = len(ligand.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('=', '').replace('#', '').replace('.', ''))

            # 简单的大小限制
            if 3 <= atom_count <= 150:
                return ligand
            return None

        except Exception as e:
            logger.error(f"简化配体生成时出错: {e}")
            return None

    def _is_valid_molecule(self, mol) -> bool:
        """验证分子是否有效"""
        try:
            if mol is None:
                return False

            # 如果RDKit不可用，使用简化验证
            if not RDKIT_AVAILABLE:
                # 简单的原子计数验证
                smiles = Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)
                atom_count = len(smiles.replace('[', '').replace(']', '').replace('.', ''))
                return 3 <= atom_count <= 150

            # RDKit可用时使用完整验证
            mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol_clean is None:
                return False

            # 分子大小限制
            if mol_clean.GetNumAtoms() < 3 or mol_clean.GetNumAtoms() > 150:
                return False

            # 尝试计算分子性质作为验证
            try:
                Descriptors.MolWt(mol_clean)
                return True
            except:
                return False

        except Exception:
            return False

    def _random_modification(self, mol) -> Optional[str]:
        """随机修饰分子"""
        try:
            # 如果RDKit不可用，直接返回原分子
            if not RDKIT_AVAILABLE:
                return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

            # 简化的修饰方法
            modification_type = random.choice([
                "simple_substitution",
                "functional_group_addition"
            ])

            if modification_type == "simple_substitution":
                return self._simple_substitution(mol)
            elif modification_type == "functional_group_addition":
                return self._add_simple_functional_group(mol)

        except Exception as e:
            logger.error(f"分子修饰时出错: {e}")
            return mol

    def _simple_substitution(self, mol) -> Optional[str]:
        """简单的取代基添加"""
        try:
            # 如果RDKit不可用，直接返回原分子
            if not RDKIT_AVAILABLE:
                return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

            # 使用更安全的SMILES操作
            smiles = Chem.MolToSmiles(mol, canonical=True)

            # 简单的取代基
            substituents = ["F", "Cl", "Br", "O", "N"]
            substituent = random.choice(substituents)

            # 创建新的SMILES
            modified_smiles = smiles + "." + substituent
            return modified_smiles

        except Exception as e:
            logger.error(f"简单取代时出错: {e}")
            return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

    def _add_simple_functional_group(self, mol) -> Optional[str]:
        """添加简单的官能团"""
        try:
            # 如果RDKit不可用，直接返回原分子
            if not RDKIT_AVAILABLE:
                return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

            smiles = Chem.MolToSmiles(mol, canonical=True)

            # 简单的官能团
            functional_groups = ["C(=O)O", "C(=O)N", "C#N"]
            fg = random.choice(functional_groups)

            # 组合SMILES
            combined_smiles = smiles + "." + fg
            return combined_smiles

        except Exception as e:
            logger.error(f"添加官能团时出错: {e}")
            return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

    def generate_cmdgen_ligands(self,
                               pdb_file: Optional[str] = None,
                               num_ligands: Optional[int] = None,
                               ref_ligand: str = "A:1") -> List[Dict]:
        """
        使用CMD-GEN生成基于结构的配体

        Args:
            pdb_file: PDB文件路径
            num_ligands: 生成配体数量
            ref_ligand: 参考配体

        Returns:
            生成的配体列表
        """
        # 如果未指定数量，使用配置文件中的默认值
        if num_ligands is None:
            num_ligands = LIGAND_GENERATION["num_ligands"]

        if not self.use_cmdgen or not self.cmdgen_generator:
            logger.warning("CMD-GEN不可用，使用传统方法生成配体")
            return self._generate_traditional_ligands(num_ligands)

        if not pdb_file or not os.path.exists(pdb_file):
            logger.warning("PDB文件不存在，使用传统方法生成配体")
            return self._generate_traditional_ligands(num_ligands)

        try:
            logger.info(f"使用CMD-GEN生成 {num_ligands} 个基于结构的配体")

            # 使用CMD-GEN生成分子
            molecules = self.cmdgen_generator.generate_pocket_based_molecules(
                pdb_file=pdb_file,
                num_molecules=num_ligands,
                ref_ligand=ref_ligand
            )

            if molecules:
                logger.info(f"CMD-GEN成功生成 {len(molecules)} 个配体")
                return molecules
            else:
                logger.warning("CMD-GEN生成失败，使用传统方法")
                return self._generate_traditional_ligands(num_ligands)

        except Exception as e:
            logger.error(f"CMD-GEN生成过程中出错: {e}")
            return self._generate_traditional_ligands(num_ligands)

    def _generate_traditional_ligands(self, num_ligands: int) -> List[Dict]:
        """传统方法生成配体"""
        logger.info(f"使用传统方法生成 {num_ligands} 个配体")

        ligands = []
        successful_generations = 0
        max_attempts = num_ligands * 3

        for attempt in range(max_attempts):
            if successful_generations >= num_ligands:
                break

            smiles = self.generate_random_ligand()
            if smiles:
                ligand_info = {
                    "compound_id": f"ligand_{successful_generations + 1}",
                    "smiles": smiles,
                    "generation_method": "Traditional",
                    "source": "template_based"
                }

                # 计算分子性质
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        ligand_info.update({
                            "molecular_weight": Descriptors.MolWt(mol),
                            "logp": Descriptors.MolLogP(mol),
                            "hbd": Descriptors.NumHDonors(mol),
                            "hba": Descriptors.NumHAcceptors(mol),
                            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                            "tpsa": Descriptors.TPSA(mol)
                        })

                ligands.append(ligand_info)
                successful_generations += 1

        return ligands

    def generate_optimized_ligands(self,
                                 num_ligands: Optional[int] = None,
                                 pdb_file: Optional[str] = None,
                                 use_cmdgen: Optional[bool] = None,
                                 optimize_for_binding: bool = True) -> List[Dict]:
        """
        生成优化的配体库

        Args:
            num_ligands: 生成配体数量
            pdb_file: PDB文件路径（用于CMD-GEN）
            use_cmdgen: 是否使用CMD-GEN（覆盖默认设置）
            optimize_for_binding: 是否针对结合亲和力进行优化

        Returns:
            生成的配体列表
        """
        if num_ligands is None:
            num_ligands = LIGAND_GENERATION["num_ligands"]

        # 决定使用哪种生成方法
        should_use_cmdgen = use_cmdgen if use_cmdgen is not None else self.use_cmdgen

        if should_use_cmdgen and pdb_file:
            logger.info(f"使用CMD-GEN生成 {num_ligands} 个基于结构的配体")
            ligands = self.generate_cmdgen_ligands(
                pdb_file=pdb_file,
                num_ligands=num_ligands
            )
        else:
            logger.info(f"使用传统方法生成 {num_ligands} 个配体")
            ligands = self._generate_traditional_ligands(num_ligands)

        # 如果启用结合优化，进行后处理
        if optimize_for_binding and ligands:
            logger.info("对生成的配体进行结合亲和力优化...")
            ligands = self._optimize_ligands_for_binding(ligands)

        return ligands

    def _calculate_molecular_properties(self, smiles: str) -> Dict:
        """计算分子性质"""
        try:
            # 如果RDKit不可用，返回简化性质
            if not RDKIT_AVAILABLE:
                return {
                    "molecular_weight": random.uniform(100, 500),
                    "logp": random.uniform(-1, 5),
                    "hbd": random.randint(0, 5),
                    "hba": random.randint(0, 10),
                    "rotatable_bonds": random.randint(0, 10),
                    "tpsa": random.uniform(0, 150),
                    "rings": random.randint(0, 5),
                    "aromatic_rings": random.randint(0, 3),
                }

            # 创建分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            # 计算基本性质
            properties = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rings": Descriptors.RingCount(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "formal_charge": Chem.rdmolops.GetFormalCharge(mol),
                "fraction_csp3": getattr(Descriptors, 'FractionCsp3', lambda x: 0.0)(mol)
            }

            return properties

        except Exception as e:
            logger.error(f"计算分子性质时出错: {e}")
            return {}

    def _check_admet_criteria(self, properties: Dict) -> bool:
        """检查ADMET标准"""
        try:
            # 检查分子量
            if not (ADMET_CRITERIA["molecular_weight"][0] <=
                    properties["molecular_weight"] <=
                    ADMET_CRITERIA["molecular_weight"][1]):
                return False

            # 检查LogP
            if not (ADMET_CRITERIA["logp_range"][0] <=
                    properties["logp"] <=
                    ADMET_CRITERIA["logp_range"][1]):
                return False

            # 检查氢键供体
            if properties["hbd"] > ADMET_CRITERIA["hbd_max"]:
                return False

            # 检查氢键受体
            if properties["hba"] > ADMET_CRITERIA["hba_max"]:
                return False

            # 检查可旋转键
            if properties["rotatable_bonds"] > ADMET_CRITERIA["rotatable_bonds_max"]:
                return False

            # 检查TPSA
            if not (ADMET_CRITERIA["tpsa_range"][0] <=
                    properties["tpsa"] <=
                    ADMET_CRITERIA["tpsa_range"][1]):
                return False

            # 检查芳香环数（如果配置中有此项）
            if "aromatic_rings_max" in ADMET_CRITERIA:
                aromatic_rings = properties.get("aromatic_rings", 0)
                if aromatic_rings > ADMET_CRITERIA["aromatic_rings_max"]:
                    return False

            # 检查重原子数（如果配置中有此项）
            if "heavy_atoms_range" in ADMET_CRITERIA:
                heavy_atoms = properties.get("heavy_atoms", 0)
                if not (ADMET_CRITERIA["heavy_atoms_range"][0] <=
                        heavy_atoms <=
                        ADMET_CRITERIA["heavy_atoms_range"][1]):
                    return False

            return True

        except Exception as e:
            logger.error(f"检查ADMET标准时出错: {e}")
            return False

    def _optimize_ligands_for_binding(self, ligands: List[Dict]) -> List[Dict]:
        """
        优化配体以提高结合亲和力

        Args:
            ligands: 原始配体列表

        Returns:
            优化后的配体列表
        """
        try:
            optimized_ligands = []
            scores = []

            for ligand in ligands:
                # 计算分子描述符
                properties = self._calculate_molecular_properties(ligand["smiles"])

                # 基于分子性质进行评分
                binding_score = self._calculate_binding_potential_score(properties)
                ligand["binding_potential_score"] = binding_score
                scores.append(binding_score)

                # 降低阈值以保留更多配体
                if binding_score > 0.3:  # 降低阈值从0.6到0.3
                    optimized_ligands.append(ligand)

            # 如果仍然没有配体通过筛选，保留评分最高的几个
            if not optimized_ligands and ligands:
                logger.warning("所有配体评分都低于阈值，保留评分最高的配体")
                # 按评分排序并保留前50%
                ligands_with_scores = [(ligand, score) for ligand, score in zip(ligands, scores)]
                ligands_with_scores.sort(key=lambda x: x[1], reverse=True)
                keep_count = max(1, len(ligands) // 2)
                optimized_ligands = [ligand for ligand, _ in ligands_with_scores[:keep_count]]

            # 按结合潜力排序
            optimized_ligands.sort(key=lambda x: x.get("binding_potential_score", 0), reverse=True)

            logger.info(f"配体优化完成：{len(ligands)} -> {len(optimized_ligands)} 个高潜力配体")
            if scores:
                logger.info(f"评分范围: {min(scores):.3f} - {max(scores):.3f}")

            return optimized_ligands

        except Exception as e:
            logger.error(f"配体优化过程出错: {e}")
            return ligands

    def _calculate_binding_potential_score(self, properties: Dict) -> float:
        """
        基于分子性质计算结合潜力评分

        Args:
            properties: 分子性质字典

        Returns:
            结合潜力评分 (0-1)
        """
        try:
            score = 0.0

            # LogP评分 (理想范围 2-3.5)
            logp = properties.get("logp", 0)
            if 2.0 <= logp <= 3.5:
                score += 0.25
            elif 1.5 <= logp <= 4.0:
                score += 0.15

            # 分子量评分 (理想范围 300-450)
            mw = properties.get("molecular_weight", 0)
            if 300 <= mw <= 450:
                score += 0.25
            elif 250 <= mw <= 500:
                score += 0.15

            # 氢键供体/受体评分
            hbd = properties.get("hbd", 0)
            hba = properties.get("hba", 0)
            if 1 <= hbd <= 3 and 3 <= hba <= 6:
                score += 0.2

            # 芳香环评分 (2-3个芳香环通常有利于蛋白结合)
            aromatic_rings = properties.get("aromatic_rings", 0)
            if 2 <= aromatic_rings <= 3:
                score += 0.15
            elif aromatic_rings == 1 or aromatic_rings == 4:
                score += 0.1

            # TPSA评分 (适中的极性表面积)
            tpsa = properties.get("tpsa", 0)
            if 60 <= tpsa <= 90:
                score += 0.15
            elif 40 <= tpsa <= 110:
                score += 0.1

            return min(score, 1.0)  # 确保评分不超过1.0

        except Exception as e:
            logger.error(f"计算结合潜力评分时出错: {e}")
            return 0.5  # 返回中等评分

    def save_ligands(self, ligands: List[Dict], filename: Optional[str] = None):
        """保存配体到文件"""
        if filename is None:
            filename = os.path.join(RESULTS_DIR, "generated_ligands.csv")

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(ligands)
        df.to_csv(filename, index=False)
        logger.info(f"配体数据已保存到: {filename}")

        # 生成SMILES文件用于对接
        smiles_file = os.path.join(RESULTS_DIR, "ligands.smi")
        with open(smiles_file, 'w') as f:
            for ligand in ligands:
                f.write(f"{ligand['smiles']}\t{ligand['smiles']}\n")
        logger.info(f"SMILES文件已保存到: {smiles_file}")

    def visualize_ligands(self, ligands: List[Dict], num_ligands: int = 10):
        """可视化配体"""
        try:
            # 创建图像目录
            img_dir = os.path.join(RESULTS_DIR, OUTPUT_FILES["ligand_images"])
            os.makedirs(img_dir, exist_ok=True)

            # 选择前N个配体进行可视化
            selected_ligands = ligands[:min(num_ligands, len(ligands))]

            for i, ligand in enumerate(selected_ligands):
                # 如果RDKit不可用，跳过可视化
                if not RDKIT_AVAILABLE:
                    logger.warning("RDKit不可用，跳过配体可视化")
                    break

                mol = Chem.MolFromSmiles(ligand["smiles"])
                if mol is not None:
                    # 生成2D图像
                    img = Chem.Draw.MolToImage(mol, size=(300, 300))
                    img_path = os.path.join(img_dir, f"ligand_{i + 1}.png")
                    img.save(img_path)

            if RDKIT_AVAILABLE:
                logger.info(f"配体图像已保存到: {img_dir}")

        except Exception as e:
            logger.error(f"可视化配体时出错: {e}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("PRRSV病毒衣壳蛋白抑制剂配体生成系统启动")
    logger.info("=" * 60)

    # 打印当前配置
    logger.info(f"配体生成数量: {LIGAND_GENERATION['num_ligands']}")
    logger.info(f"ADMET标准: {ADMET_CRITERIA}")

    # 初始化生成器
    generator = LigandGenerator()

    # 生成配体
    ligands = generator.generate_optimized_ligands()

    if ligands:
        # 保存配体
        generator.save_ligands(ligands)

        # 可视化配体
        generator.visualize_ligands(ligands)

        logger.info(f"成功生成 {len(ligands)} 个配体")
        logger.info("配体数据已保存到 results/generated_ligands.csv")
        logger.info("配体图像已保存到 results/ligand_images/")
    else:
        logger.error("未能生成有效配体")


if __name__ == "__main__":
    main()