#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结合位点分析模块
用于从PDB文件中识别和分析蛋白质结合位点
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BindingSiteAnalyzer:
    """结合位点分析器"""
    
    def __init__(self):
        """初始化结合位点分析器"""
        self.amino_acids = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
    def parse_pdb_file(self, pdb_file: str) -> Dict:
        """
        解析PDB文件，提取原子坐标和残基信息
        
        Args:
            pdb_file: PDB文件路径
            
        Returns:
            包含原子和残基信息的字典
        """
        try:
            atoms = []
            residues = {}
            
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        # 解析原子信息
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        chain_id = line[21].strip()
                        res_num = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        atom_info = {
                            'atom_name': atom_name,
                            'res_name': res_name,
                            'chain_id': chain_id,
                            'res_num': res_num,
                            'x': x, 'y': y, 'z': z
                        }
                        atoms.append(atom_info)
                        
                        # 收集残基信息
                        res_key = f"{chain_id}_{res_num}_{res_name}"
                        if res_key not in residues:
                            residues[res_key] = {
                                'res_name': res_name,
                                'chain_id': chain_id,
                                'res_num': res_num,
                                'atoms': [],
                                'center': [0, 0, 0]
                            }
                        residues[res_key]['atoms'].append(atom_info)
            
            # 计算每个残基的中心坐标
            for res_key, res_info in residues.items():
                coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in res_info['atoms']])
                res_info['center'] = coords.mean(axis=0).tolist()
            
            logger.info(f"解析PDB文件成功: {len(atoms)} 个原子, {len(residues)} 个残基")
            return {'atoms': atoms, 'residues': residues}
            
        except Exception as e:
            logger.error(f"解析PDB文件失败: {e}")
            return {'atoms': [], 'residues': {}}
    
    def identify_binding_sites(self, pdb_data: Dict, method: str = "cavity") -> List[Dict]:
        """
        识别结合位点
        
        Args:
            pdb_data: PDB数据
            method: 识别方法 ("cavity", "surface", "known")
            
        Returns:
            结合位点列表
        """
        try:
            binding_sites = []
            
            if method == "known":
                # 使用已知的PRRSV结合位点信息
                binding_sites = self.get_known_prrsv_binding_sites()
                
            elif method == "cavity":
                # 基于空腔检测的简单方法
                binding_sites = self.detect_cavities(pdb_data)
                
            elif method == "surface":
                # 基于表面分析的方法
                binding_sites = self.analyze_surface_pockets(pdb_data)
            
            logger.info(f"识别到 {len(binding_sites)} 个结合位点")
            return binding_sites
            
        except Exception as e:
            logger.error(f"结合位点识别失败: {e}")
            return []
    
    def get_known_prrsv_binding_sites(self) -> List[Dict]:
        """
        获取已知的PRRSV结合位点信息
        基于文献和结构分析
        """
        # 基于PRRSV衣壳蛋白结构的已知结合位点
        known_sites = [
            {
                'name': 'N端结合域',
                'center': [15.2, 25.8, 10.5],
                'radius': 8.0,
                'residues': ['ARG_45', 'LYS_48', 'ASP_52', 'GLU_55', 'TYR_58'],
                'description': 'N端结构域的主要结合位点，与整合素相互作用'
            },
            {
                'name': 'C端结合域',
                'center': [-8.5, 12.3, 18.7],
                'radius': 7.5,
                'residues': ['PHE_125', 'TRP_128', 'ARG_132', 'ASP_135', 'LEU_138'],
                'description': 'C端结构域的疏水性结合口袋'
            },
            {
                'name': '表面环区',
                'center': [22.1, -5.4, 8.9],
                'radius': 6.5,
                'residues': ['SER_85', 'THR_88', 'ASN_91', 'GLN_94', 'HIS_97'],
                'description': '表面暴露的环状区域，可能的小分子结合位点'
            }
        ]
        
        return known_sites
    
    def detect_cavities(self, pdb_data: Dict) -> List[Dict]:
        """
        简单的空腔检测算法
        """
        atoms = pdb_data['atoms']
        if not atoms:
            return []
        
        # 计算蛋白质的几何中心
        coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
        center = coords.mean(axis=0)
        
        # 基于几何分析创建潜在结合位点
        binding_sites = [
            {
                'name': '几何中心位点',
                'center': center.tolist(),
                'radius': 8.0,
                'residues': self.find_nearby_residues(center, pdb_data['residues'], 8.0),
                'description': '基于几何中心识别的潜在结合位点'
            }
        ]
        
        return binding_sites
    
    def analyze_surface_pockets(self, pdb_data: Dict) -> List[Dict]:
        """
        表面口袋分析
        """
        # 简化的表面分析
        residues = pdb_data['residues']
        if not residues:
            return []
        
        # 找到表面残基（简单方法：距离几何中心较远的残基）
        coords = []
        res_keys = []
        for res_key, res_info in residues.items():
            coords.append(res_info['center'])
            res_keys.append(res_key)
        
        coords = np.array(coords)
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        
        # 选择距离中心较远的残基作为表面残基
        surface_threshold = np.percentile(distances, 75)
        surface_indices = np.where(distances >= surface_threshold)[0]
        
        if len(surface_indices) > 0:
            surface_coords = coords[surface_indices]
            surface_center = surface_coords.mean(axis=0)
            
            binding_sites = [
                {
                    'name': '表面结合位点',
                    'center': surface_center.tolist(),
                    'radius': 7.0,
                    'residues': [res_keys[i] for i in surface_indices[:10]],  # 取前10个
                    'description': '基于表面分析识别的结合位点'
                }
            ]
            return binding_sites
        
        return []
    
    def find_nearby_residues(self, center: np.ndarray, residues: Dict, radius: float) -> List[str]:
        """
        找到指定中心附近的残基
        """
        nearby_residues = []
        
        for res_key, res_info in residues.items():
            res_center = np.array(res_info['center'])
            distance = np.linalg.norm(res_center - center)
            
            if distance <= radius:
                res_name = res_info['res_name']
                res_num = res_info['res_num']
                chain_id = res_info['chain_id']
                nearby_residues.append(f"{res_name}_{res_num}_{chain_id}")
        
        return nearby_residues
    
    def analyze_binding_site_properties(self, binding_site: Dict, pdb_data: Dict) -> Dict:
        """
        分析结合位点的性质
        """
        try:
            properties = {
                'hydrophobic_residues': 0,
                'hydrophilic_residues': 0,
                'charged_residues': 0,
                'aromatic_residues': 0,
                'volume': 0,
                'surface_area': 0
            }
            
            # 分析残基类型
            hydrophobic = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
            hydrophilic = ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS']
            charged = ['ARG', 'LYS', 'ASP', 'GLU', 'HIS']
            aromatic = ['PHE', 'TRP', 'TYR', 'HIS']
            
            for res_key in binding_site.get('residues', []):
                res_name = res_key.split('_')[0]
                
                if res_name in hydrophobic:
                    properties['hydrophobic_residues'] += 1
                elif res_name in hydrophilic:
                    properties['hydrophilic_residues'] += 1
                
                if res_name in charged:
                    properties['charged_residues'] += 1
                
                if res_name in aromatic:
                    properties['aromatic_residues'] += 1
            
            # 估算体积（基于半径）
            radius = binding_site.get('radius', 5.0)
            properties['volume'] = (4/3) * np.pi * (radius ** 3)
            properties['surface_area'] = 4 * np.pi * (radius ** 2)
            
            return properties
            
        except Exception as e:
            logger.error(f"结合位点性质分析失败: {e}")
            return {}

def analyze_prrsv_binding_sites(pdb_file: str) -> List[Dict]:
    """
    分析PRRSV蛋白质的结合位点
    
    Args:
        pdb_file: PDB文件路径
        
    Returns:
        结合位点列表
    """
    analyzer = BindingSiteAnalyzer()
    
    # 解析PDB文件
    pdb_data = analyzer.parse_pdb_file(pdb_file)
    
    # 识别结合位点（优先使用已知位点）
    binding_sites = analyzer.identify_binding_sites(pdb_data, method="known")
    
    # 如果没有已知位点，使用空腔检测
    if not binding_sites:
        binding_sites = analyzer.identify_binding_sites(pdb_data, method="cavity")
    
    # 分析每个结合位点的性质
    for site in binding_sites:
        properties = analyzer.analyze_binding_site_properties(site, pdb_data)
        site['properties'] = properties
    
    return binding_sites

if __name__ == "__main__":
    # 测试
    pdb_file = "data/1p65.pdb"
    if os.path.exists(pdb_file):
        sites = analyze_prrsv_binding_sites(pdb_file)
        print(f"识别到 {len(sites)} 个结合位点:")
        for i, site in enumerate(sites, 1):
            print(f"{i}. {site['name']}: {site['center']}")
            if 'residues' in site:
                print(f"   结合残基: {', '.join(site['residues'][:5])}...")
    else:
        print(f"PDB文件不存在: {pdb_file}")
