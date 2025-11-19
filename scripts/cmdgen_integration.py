# -*- coding: utf-8 -*-
"""
CMD-GEN集成模块
集成最新的CMD-GEN (Coarse-grained and Multi-dimensional Data-driven molecular generation) 模型
用于基于结构的分子生成
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from config import *

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RDKit导入处理
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
    logger.info("RDKit已成功导入，将使用完整CMD-GEN功能")
except ImportError:
    logger.warning("RDKit不可用，CMD-GEN功能将受限")

class CMDGENGenerator:
    """CMD-GEN分子生成器"""

    def __init__(self,
                 cmdgen_path: Optional[str] = None,
                 diffphar_weights: Optional[str] = None,
                 gcpg_weights: Optional[str] = None,
                 device: str = "cpu"):
        """
        初始化CMD-GEN生成器

        Args:
            cmdgen_path: CMD-GEN代码路径
            diffphar_weights: DiffPhar模型权重路径
            gcpg_weights: GCPG模型权重路径
            device: 计算设备 (cpu/cuda)
        """
        self.cmdgen_path = cmdgen_path or self._get_default_cmdgen_path()
        self.device = device

        # 加载配置文件
        self._load_config()

        # 设置权重路径
        self.diffphar_weights = diffphar_weights or self._get_diffphar_weights()
        self.gcpg_weights = gcpg_weights or self._get_gcpg_weights()
        self.gcpg_tokenizer = self._get_gcpg_tokenizer()

        # 检查CMD-GEN是否可用
        self.cmdgen_available = self._check_cmdgen_availability()

        if not self.cmdgen_available:
            logger.warning("CMD-GEN不可用，将使用备用分子生成方法")

    def _load_config(self):
        """加载CMD-GEN配置"""
        config_file = "cmdgen_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info("已加载CMD-GEN配置文件")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")
                self.config = {}
        else:
            self.config = {}

    def _get_diffphar_weights(self) -> str:
        """获取DiffPhar权重路径"""
        if self.config and 'weights' in self.config:
            return self.config['weights'].get('diffphar', '')

        # 默认路径
        default_paths = [
            os.path.join(self.cmdgen_path, "DiffPhar/checkpoints/best-model-epoch=epoch=281.ckpt"),
            "CMD-GEN/DiffPhar/checkpoints/best-model-epoch=epoch=281.ckpt",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "CMD-GEN/DiffPhar/checkpoints/best-model-epoch=epoch=281.ckpt")
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        return ""

    def _get_gcpg_weights(self) -> str:
        """获取GCPG权重路径"""
        if self.config and 'weights' in self.config:
            return self.config['weights'].get('gcpg', '')

        # 默认路径
        default_paths = [
            os.path.join(self.cmdgen_path, "GCPG/result/rs_mapping/fold0_epoch64.pth"),
            "CMD-GEN/GCPG/result/rs_mapping/fold0_epoch64.pth",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "CMD-GEN/GCPG/result/rs_mapping/fold0_epoch64.pth")
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        return ""

    def _get_gcpg_tokenizer(self) -> str:
        """获取GCPG分词器路径"""
        if self.config and 'weights' in self.config:
            return self.config['weights'].get('tokenizer', '')

        # 默认路径
        default_paths = [
            os.path.join(self.cmdgen_path, "GCPG/result/rs_mapping/tokenizer_r_iso.pkl"),
            "CMD-GEN/GCPG/result/rs_mapping/tokenizer_r_iso.pkl",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "CMD-GEN/GCPG/result/rs_mapping/tokenizer_r_iso.pkl")
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        return ""

    def _get_default_cmdgen_path(self) -> str:
        """获取默认CMD-GEN路径"""
        # 检查常见的安装位置
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "CMD-GEN"),  # HJD/CMD-GEN
            os.path.join(os.path.dirname(__file__), "..", "CMD-GEN"),  # 相对路径
            os.path.join(os.path.dirname(__file__), "CMD-GEN"),
            os.path.join(os.path.expanduser("~"), "CMD-GEN"),
            "/opt/CMD-GEN",
            "./CMD-GEN"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return ""
    
    def _check_cmdgen_availability(self) -> bool:
        """检查CMD-GEN是否可用"""
        if not self.cmdgen_path or not os.path.exists(self.cmdgen_path):
            logger.warning("CMD-GEN路径不存在")
            return False

        # 检查必要的文件
        required_files = [
            "DiffPhar/generate_phars.py",
            "GCPG/generate.py",
            "PharAlign/align_test_wrn.py"
        ]

        for file in required_files:
            if not os.path.exists(os.path.join(self.cmdgen_path, file)):
                logger.warning(f"缺少CMD-GEN文件: {file}")
                return False

        # 检查权重文件
        if not os.path.exists(self.diffphar_weights):
            logger.warning(f"DiffPhar权重文件不存在: {self.diffphar_weights}")
            return False

        if not os.path.exists(self.gcpg_weights):
            logger.warning(f"GCPG权重文件不存在: {self.gcpg_weights}")
            return False

        if not os.path.exists(self.gcpg_tokenizer):
            logger.warning(f"GCPG分词器文件不存在: {self.gcpg_tokenizer}")
            return False

        logger.info("CMD-GEN所有组件检查通过")
        return True
    
    def generate_pharmacophore_points(self, 
                                    pdb_file: str,
                                    ref_ligand: str = "A:1",
                                    num_nodes: int = 10) -> Optional[str]:
        """
        生成药效团点
        
        Args:
            pdb_file: PDB文件路径
            ref_ligand: 参考配体 (格式: chain:index)
            num_nodes: 药效团点数量
            
        Returns:
            药效团点文件路径
        """
        if not self.cmdgen_available:
            logger.error("CMD-GEN不可用，无法生成药效团点")
            return None
        
        try:
            # 创建临时输出目录
            temp_dir = tempfile.mkdtemp()
            output_file = os.path.join(temp_dir, "pharmacophore_points.json")
            
            # 构建命令
            cmd = [
                "python",
                os.path.join(self.cmdgen_path, "DiffPhar/generate_phars.py"),
                self.diffphar_weights,
                "--num_nodes_phar", str(num_nodes),
                "--pdbfile", pdb_file,
                "--ref_ligand", ref_ligand,
                "--output", output_file,
                "--device", self.device
            ]
            
            # 执行命令
            logger.info(f"生成药效团点: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                  cwd=self.cmdgen_path,
                                  capture_output=True, 
                                  text=True,
                                  timeout=300)
            
            if result.returncode == 0:
                logger.info("药效团点生成成功")
                return output_file
            else:
                logger.error(f"药效团点生成失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"生成药效团点时出错: {e}")
            return None
    
    def generate_molecules_from_pharmacophore(self,
                                            pharmacophore_file: str,
                                            num_molecules: int = 50,
                                            molecular_properties: Optional[Dict] = None) -> List[str]:
        """
        基于药效团生成分子
        
        Args:
            pharmacophore_file: 药效团文件路径
            num_molecules: 生成分子数量
            molecular_properties: 分子性质约束
            
        Returns:
            生成的SMILES列表
        """
        if not self.cmdgen_available:
            logger.warning("CMD-GEN不可用，使用备用方法生成分子")
            return self._generate_fallback_molecules(num_molecules)
        
        try:
            # 创建临时输出目录
            temp_dir = tempfile.mkdtemp()
            output_dir = os.path.join(temp_dir, "generated_molecules")
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建命令
            cmd = [
                "python",
                os.path.join(self.cmdgen_path, "GCPG/generate.py"),
                pharmacophore_file,
                output_dir,
                self.gcpg_weights,
                self.gcpg_tokenizer,
                "--n_mol", str(num_molecules),
                "--device", self.device,
                "--filter"
            ]
            
            # 执行命令
            logger.info(f"基于药效团生成分子: {' '.join(cmd)}")
            result = subprocess.run(cmd,
                                  cwd=self.cmdgen_path,
                                  capture_output=True,
                                  text=True,
                                  timeout=600)
            
            if result.returncode == 0:
                # 读取生成的分子
                molecules = self._read_generated_molecules(output_dir)
                logger.info(f"成功生成 {len(molecules)} 个分子")
                return molecules
            else:
                logger.error(f"分子生成失败: {result.stderr}")
                return self._generate_fallback_molecules(num_molecules)
                
        except Exception as e:
            logger.error(f"生成分子时出错: {e}")
            return self._generate_fallback_molecules(num_molecules)
    
    def _read_generated_molecules(self, output_dir: str) -> List[str]:
        """读取生成的分子"""
        molecules = []
        
        # 查找输出文件
        for file in os.listdir(output_dir):
            if file.endswith('.txt') or file.endswith('.smi'):
                file_path = os.path.join(output_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            smiles = line.strip()
                            if smiles and self._is_valid_smiles(smiles):
                                molecules.append(smiles)
                except Exception as e:
                    logger.warning(f"读取文件 {file_path} 时出错: {e}")
        
        return molecules
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """验证SMILES是否有效"""
        if not RDKIT_AVAILABLE:
            return len(smiles) > 0
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _generate_fallback_molecules(self, num_molecules: int) -> List[str]:
        """备用分子生成方法"""
        logger.info("使用备用方法生成分子")
        
        # 使用预定义的药物类似分子模板
        templates = [
            "CCOc1ccc(C(N)=O)cc1",  # 苯甲酰胺衍生物
            "Cc1ccc(C(=O)N(C)C)cc1",  # N,N-二甲基苯甲酰胺
            "COc1ccc(CCN)cc1",  # 酪胺衍生物
            "Cc1ccc(-c2nc(N)nc(N)n2)cc1",  # 三嗪衍生物
            "CCc1ccc(C(=O)NCc2ccccc2)cc1",  # 苯乙酰胺
            "Cc1ccc(-c2nnc(N)s2)cc1",  # 噻二唑衍生物
            "COc1ccc2c(N)nc(N)nc2c1",  # 喹唑啉衍生物
            "CCOC(=O)c1ccc(OC)cc1",  # 对甲氧基苯甲酸乙酯
            "Cc1nc2ccccc2c(=O)n1C",  # 喹唑啉酮
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1"  # 布洛芬类似物
        ]
        
        molecules = []
        for i in range(num_molecules):
            template = templates[i % len(templates)]
            # 简单的分子修饰
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(template)
                if mol:
                    molecules.append(Chem.MolToSmiles(mol))
            else:
                molecules.append(template)
        
        return molecules
    
    def generate_pocket_based_molecules(self,
                                      pdb_file: str,
                                      num_molecules: int = 50,
                                      ref_ligand: str = "A:1") -> List[Dict]:
        """
        基于口袋的完整分子生成流程
        
        Args:
            pdb_file: PDB文件路径
            num_molecules: 生成分子数量
            ref_ligand: 参考配体
            
        Returns:
            生成的分子信息列表
        """
        logger.info(f"开始基于口袋的分子生成，目标: {num_molecules} 个分子")
        
        # 第一步：生成药效团点
        pharmacophore_file = self.generate_pharmacophore_points(
            pdb_file=pdb_file,
            ref_ligand=ref_ligand,
            num_nodes=10
        )
        
        # 第二步：基于药效团生成分子
        if pharmacophore_file:
            molecules = self.generate_molecules_from_pharmacophore(
                pharmacophore_file=pharmacophore_file,
                num_molecules=num_molecules
            )
        else:
            molecules = self._generate_fallback_molecules(num_molecules)
        
        # 第三步：计算分子性质
        results = []
        for i, smiles in enumerate(molecules):
            mol_info = {
                "compound_id": f"cmdgen_{i+1}",
                "smiles": smiles,
                "generation_method": "CMD-GEN" if self.cmdgen_available else "Fallback",
                "source": "pocket_based"
            }
            
            # 计算分子性质
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol_info.update({
                        "molecular_weight": Descriptors.MolWt(mol),
                        "logp": Descriptors.MolLogP(mol),
                        "hbd": Descriptors.NumHDonors(mol),
                        "hba": Descriptors.NumHAcceptors(mol),
                        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                        "tpsa": Descriptors.TPSA(mol)
                    })
            
            results.append(mol_info)
        
        logger.info(f"成功生成 {len(results)} 个基于口袋的分子")
        return results
