#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子工具函数
包含分子处理、转换和分析的实用函数
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def smiles_to_graph(smiles: str) -> Optional[Dict]:
    """
    将SMILES字符串转换为图表示
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        图字典或None（如果转换失败）
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 节点特征（原子特征）
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                atom.GetIsAromatic(),
                atom.GetMass(),
                atom.GetTotalNumHs()
            ]
            atom_features.append(features)
        
        # 边特征（键特征）
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 添加双向边
            edge_indices.extend([[i, j], [j, i]])
            
            bond_features = [
                bond.GetBondType().real,
                bond.GetIsConjugated(),
                bond.IsInRing()
            ]
            edge_features.extend([bond_features, bond_features])
        
        return {
            'node_features': torch.tensor(atom_features, dtype=torch.float),
            'edge_index': torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            'edge_features': torch.tensor(edge_features, dtype=torch.float),
            'smiles': smiles,
            'num_atoms': mol.GetNumAtoms()
        }
        
    except ImportError:
        logger.warning("RDKit不可用，无法转换SMILES到图")
        return None
    except Exception as e:
        logger.warning(f"SMILES转换失败: {smiles}, 错误: {e}")
        return None

def graph_to_smiles(graph_data: Dict) -> Optional[str]:
    """
    将图表示转换为SMILES字符串
    
    Args:
        graph_data: 图数据字典
        
    Returns:
        SMILES字符串或None
    """
    # 这是一个复杂的逆向过程，通常需要专门的图到分子转换算法
    # 这里返回原始SMILES（如果存在）
    return graph_data.get('smiles')

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    标准化SMILES字符串
    
    Args:
        smiles: 输入SMILES
        
    Returns:
        标准化的SMILES或None
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        logger.warning("RDKit不可用，无法标准化SMILES")
        return smiles
    except:
        return None

def calculate_molecular_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """
    计算分子描述符
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        描述符字典或None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol)
        }
        
        # Lipinski规则
        descriptors['lipinski_violations'] = sum([
            descriptors['molecular_weight'] > 500,
            descriptors['logp'] > 5,
            descriptors['hbd'] > 5,
            descriptors['hba'] > 10
        ])
        
        return descriptors
        
    except ImportError:
        logger.warning("RDKit不可用，无法计算分子描述符")
        return None
    except Exception as e:
        logger.warning(f"计算描述符失败: {smiles}, 错误: {e}")
        return None

def filter_molecules(smiles_list: List[str], 
                    criteria: Optional[Dict[str, Tuple[float, float]]] = None) -> List[str]:
    """
    根据分子性质过滤分子
    
    Args:
        smiles_list: SMILES列表
        criteria: 过滤条件 {property: (min_val, max_val)}
        
    Returns:
        过滤后的SMILES列表
    """
    if criteria is None:
        criteria = {
            'molecular_weight': (150, 500),
            'logp': (-2, 5),
            'hbd': (0, 5),
            'hba': (0, 10)
        }
    
    filtered_smiles = []
    
    for smiles in smiles_list:
        descriptors = calculate_molecular_descriptors(smiles)
        if descriptors is None:
            continue
            
        # 检查所有条件
        passes_filter = True
        for prop, (min_val, max_val) in criteria.items():
            if prop in descriptors:
                value = descriptors[prop]
                if not (min_val <= value <= max_val):
                    passes_filter = False
                    break
        
        if passes_filter:
            filtered_smiles.append(smiles)
    
    logger.info(f"分子过滤: {len(smiles_list)} -> {len(filtered_smiles)}")
    return filtered_smiles

def calculate_similarity(smiles1: str, smiles2: str, method: str = 'tanimoto') -> Optional[float]:
    """
    计算两个分子的相似性
    
    Args:
        smiles1: 第一个SMILES
        smiles2: 第二个SMILES
        method: 相似性方法
        
    Returns:
        相似性分数或None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import DataStructs
        from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        # 计算分子指纹
        fp1 = GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # 计算相似性
        if method == 'tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == 'dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            return DataStructs.TanimotoSimilarity(fp1, fp2)
            
    except ImportError:
        logger.warning("RDKit不可用，无法计算分子相似性")
        return None
    except Exception as e:
        logger.warning(f"相似性计算失败: {e}")
        return None

def generate_conformers(smiles: str, num_conformers: int = 10) -> Optional[List[np.ndarray]]:
    """
    生成分子构象
    
    Args:
        smiles: SMILES字符串
        num_conformers: 构象数量
        
    Returns:
        构象坐标列表或None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 生成构象
        conformer_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_conformers, randomSeed=42
        )
        
        if not conformer_ids:
            return None
        
        # 优化构象
        for conf_id in conformer_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
        
        # 提取坐标
        conformers = []
        for conf_id in conformer_ids:
            conf = mol.GetConformer(conf_id)
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            conformers.append(np.array(coords))
        
        return conformers
        
    except ImportError:
        logger.warning("RDKit不可用，无法生成构象")
        return None
    except Exception as e:
        logger.warning(f"构象生成失败: {smiles}, 错误: {e}")
        return None

def validate_smiles(smiles: str) -> bool:
    """
    验证SMILES字符串是否有效
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        是否有效
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # 如果没有RDKit，进行基本验证
        return isinstance(smiles, str) and len(smiles) > 0
    except:
        return False

def enumerate_stereoisomers(smiles: str) -> List[str]:
    """
    枚举立体异构体
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        立体异构体SMILES列表
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        
        opts = StereoEnumerationOptions(maxIsomers=20)
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        
        isomer_smiles = []
        for isomer in isomers:
            smi = Chem.MolToSmiles(isomer, isomericSmiles=True)
            isomer_smiles.append(smi)
        
        return list(set(isomer_smiles))  # 去重
        
    except ImportError:
        logger.warning("RDKit不可用，无法枚举立体异构体")
        return [smiles]
    except Exception as e:
        logger.warning(f"立体异构体枚举失败: {smiles}, 错误: {e}")
        return [smiles]
