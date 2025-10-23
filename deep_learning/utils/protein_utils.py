#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蛋白质工具函数
包含蛋白质结构处理和分析的实用函数
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_protein_structure(pdb_path: Union[str, Path]) -> Optional[Dict]:
    """
    加载蛋白质结构
    
    Args:
        pdb_path: PDB文件路径
        
    Returns:
        蛋白质结构字典或None
    """
    try:
        # 尝试使用BioPython
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        # 提取原子坐标和信息
        atoms = []
        coordinates = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append({
                            'name': atom.get_name(),
                            'element': atom.element,
                            'residue': residue.get_resname(),
                            'residue_id': residue.get_id()[1],
                            'chain': chain.get_id(),
                            'bfactor': atom.get_bfactor(),
                            'occupancy': atom.get_occupancy()
                        })
                        coordinates.append(atom.get_coord())
        
        return {
            'atoms': atoms,
            'coordinates': np.array(coordinates),
            'structure': structure,
            'num_atoms': len(atoms)
        }
        
    except ImportError:
        logger.warning("BioPython不可用，尝试简单解析")
        return _simple_pdb_parser(pdb_path)
    except Exception as e:
        logger.warning(f"PDB加载失败: {pdb_path}, 错误: {e}")
        return None

def _simple_pdb_parser(pdb_path: Union[str, Path]) -> Optional[Dict]:
    """
    简单的PDB解析器（不依赖BioPython）
    
    Args:
        pdb_path: PDB文件路径
        
    Returns:
        蛋白质结构字典或None
    """
    try:
        atoms = []
        coordinates = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_name = line[12:16].strip()
                    residue = line[17:20].strip()
                    chain = line[21].strip()
                    residue_id = int(line[22:26].strip())
                    
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.0
                    bfactor = float(line[60:66].strip()) if line[60:66].strip() else 0.0
                    element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                    
                    atoms.append({
                        'name': atom_name,
                        'element': element,
                        'residue': residue,
                        'residue_id': residue_id,
                        'chain': chain,
                        'bfactor': bfactor,
                        'occupancy': occupancy
                    })
                    coordinates.append([x, y, z])
        
        return {
            'atoms': atoms,
            'coordinates': np.array(coordinates),
            'structure': None,
            'num_atoms': len(atoms)
        }
        
    except Exception as e:
        logger.warning(f"简单PDB解析失败: {pdb_path}, 错误: {e}")
        return None

def extract_binding_site(protein_data: Dict, 
                        ligand_coords: np.ndarray,
                        radius: float = 5.0) -> Optional[Dict]:
    """
    提取结合位点
    
    Args:
        protein_data: 蛋白质数据
        ligand_coords: 配体坐标
        radius: 结合位点半径
        
    Returns:
        结合位点数据或None
    """
    if protein_data is None or 'coordinates' not in protein_data:
        return None
    
    protein_coords = protein_data['coordinates']
    
    # 计算距离
    distances = np.linalg.norm(
        protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :],
        axis=2
    )
    
    # 找到在半径内的原子
    min_distances = np.min(distances, axis=1)
    binding_site_mask = min_distances <= radius
    
    if not np.any(binding_site_mask):
        return None
    
    # 提取结合位点原子
    binding_site_atoms = [protein_data['atoms'][i] for i in range(len(protein_data['atoms'])) if binding_site_mask[i]]
    binding_site_coords = protein_coords[binding_site_mask]
    
    return {
        'atoms': binding_site_atoms,
        'coordinates': binding_site_coords,
        'indices': np.where(binding_site_mask)[0],
        'num_atoms': len(binding_site_atoms)
    }

def calculate_protein_features(protein_data: Dict) -> Optional[torch.Tensor]:
    """
    计算蛋白质特征
    
    Args:
        protein_data: 蛋白质数据
        
    Returns:
        特征张量或None
    """
    if protein_data is None or 'atoms' not in protein_data:
        return None
    
    # 氨基酸编码
    aa_encoding = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
    }
    
    # 原子类型编码
    atom_encoding = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5
    }
    
    features = []
    
    for atom in protein_data['atoms']:
        # 氨基酸类型
        aa_type = aa_encoding.get(atom['residue'], 20)  # 20为未知氨基酸
        
        # 原子类型
        atom_type = atom_encoding.get(atom['element'], 6)  # 6为其他原子
        
        # 原子特征
        atom_features = [
            aa_type,
            atom_type,
            atom['bfactor'] / 100.0,  # 标准化B因子
            atom['occupancy'],
            1.0 if atom['name'] == 'CA' else 0.0,  # 是否为Cα原子
            1.0 if atom['name'] in ['N', 'CA', 'C', 'O'] else 0.0  # 是否为主链原子
        ]
        
        features.append(atom_features)
    
    return torch.tensor(features, dtype=torch.float)

def align_proteins(protein1_coords: np.ndarray, 
                  protein2_coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    对齐两个蛋白质结构
    
    Args:
        protein1_coords: 第一个蛋白质坐标
        protein2_coords: 第二个蛋白质坐标
        
    Returns:
        (对齐后的第二个蛋白质坐标, RMSD)
    """
    # 简单的质心对齐
    centroid1 = np.mean(protein1_coords, axis=0)
    centroid2 = np.mean(protein2_coords, axis=0)
    
    # 平移到质心
    coords1_centered = protein1_coords - centroid1
    coords2_centered = protein2_coords - centroid2
    
    # 使用Kabsch算法进行旋转对齐（简化版本）
    try:
        # 计算协方差矩阵
        H = coords2_centered.T @ coords1_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 确保是右手坐标系
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 应用旋转
        coords2_aligned = coords2_centered @ R.T + centroid1
        
        # 计算RMSD
        rmsd = np.sqrt(np.mean(np.sum((coords1_centered - (coords2_aligned - centroid1))**2, axis=1)))
        
        return coords2_aligned, rmsd
        
    except Exception as e:
        logger.warning(f"蛋白质对齐失败: {e}")
        # 返回简单的质心对齐结果
        coords2_aligned = coords2_centered + centroid1
        rmsd = np.sqrt(np.mean(np.sum((coords1_centered - coords2_centered)**2, axis=1)))
        return coords2_aligned, rmsd

def calculate_secondary_structure(protein_data: Dict) -> Optional[List[str]]:
    """
    计算二级结构（简化版本）
    
    Args:
        protein_data: 蛋白质数据
        
    Returns:
        二级结构列表或None
    """
    # 这里应该实现DSSP算法或使用现有工具
    # 简化版本：基于Cα原子距离的粗略估计
    
    if protein_data is None or 'coordinates' not in protein_data:
        return None
    
    # 提取Cα原子
    ca_atoms = []
    ca_coords = []
    
    for i, atom in enumerate(protein_data['atoms']):
        if atom['name'] == 'CA':
            ca_atoms.append(atom)
            ca_coords.append(protein_data['coordinates'][i])
    
    if len(ca_coords) < 3:
        return ['C'] * len(ca_atoms)  # 全部标记为coil
    
    ca_coords = np.array(ca_coords)
    secondary_structure = []
    
    for i in range(len(ca_coords)):
        # 简单的二级结构预测（基于局部几何）
        if i < 2 or i >= len(ca_coords) - 2:
            secondary_structure.append('C')  # coil
        else:
            # 计算局部曲率
            v1 = ca_coords[i] - ca_coords[i-1]
            v2 = ca_coords[i+1] - ca_coords[i]
            
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            
            if angle > 2.5:  # 大角度变化
                secondary_structure.append('C')  # coil
            elif angle < 1.0:  # 小角度变化
                secondary_structure.append('H')  # helix
            else:
                secondary_structure.append('E')  # sheet
    
    return secondary_structure

def calculate_surface_area(protein_data: Dict, probe_radius: float = 1.4) -> Optional[float]:
    """
    计算蛋白质表面积（简化版本）
    
    Args:
        protein_data: 蛋白质数据
        probe_radius: 探针半径
        
    Returns:
        表面积或None
    """
    # 这里应该实现更精确的表面积计算算法
    # 简化版本：基于原子数量的估计
    
    if protein_data is None or 'num_atoms' not in protein_data:
        return None
    
    # 粗略估计：每个原子贡献约20平方埃的表面积
    estimated_surface_area = protein_data['num_atoms'] * 20.0
    
    return estimated_surface_area
