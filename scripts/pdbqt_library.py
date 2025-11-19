#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预生成PDBQT文件库
包含已知药物类似分子的PDBQT格式文件
"""

import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PDBQTLibrary:
    """预生成PDBQT文件库"""
    
    def __init__(self):
        """初始化PDBQT库"""
        self.pdbqt_templates = self._load_pdbqt_templates()
        logger.info(f"加载了 {len(self.pdbqt_templates)} 个PDBQT模板")
    
    def _load_pdbqt_templates(self) -> Dict[str, Dict]:
        """加载PDBQT模板"""
        templates = {
            'benzene': {
                'smiles': 'c1ccccc1',
                'name': 'Benzene',
                'pdbqt_content': '''REMARK  Name = c1ccccc1
ROOT
HETATM    1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.0000  C
HETATM    2  C   LIG A   1       1.400   0.000   0.000  1.00  0.00     0.0000  C
HETATM    3  C   LIG A   1       2.100   1.200   0.000  1.00  0.00     0.0000  C
HETATM    4  C   LIG A   1       1.400   2.400   0.000  1.00  0.00     0.0000  C
HETATM    5  C   LIG A   1       0.000   2.400   0.000  1.00  0.00     0.0000  C
HETATM    6  C   LIG A   1      -0.700   1.200   0.000  1.00  0.00     0.0000  C
ENDROOT
TORSDOF 0''',
                'interaction_type': 'hydrophobic',
                'mw': 78.11
            },

            'benzamide': {
                'smiles': 'c1ccc(cc1)C(=O)N',
                'name': 'Benzamide',
                'pdbqt_content': '''REMARK  Name = c1ccc(cc1)C(=O)N
ROOT
HETATM    1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.0000  C
HETATM    2  C   LIG A   1       1.400   0.000   0.000  1.00  0.00     0.0000  C
HETATM    3  C   LIG A   1       2.100   1.200   0.000  1.00  0.00     0.0000  C
HETATM    4  C   LIG A   1       1.400   2.400   0.000  1.00  0.00     0.0000  C
HETATM    5  C   LIG A   1       0.000   2.400   0.000  1.00  0.00     0.0000  C
HETATM    6  C   LIG A   1      -0.700   1.200   0.000  1.00  0.00     0.0000  C
HETATM    7  C   LIG A   1       3.500   1.200   0.000  1.00  0.00     0.0000  C
HETATM    8  O   LIG A   1       4.200   0.000   0.000  1.00  0.00     0.0000  O
HETATM    9  N   LIG A   1       4.200   2.400   0.000  1.00  0.00     0.0000  N
ENDROOT
TORSDOF 1''',
                'interaction_type': 'hydrogen_bond',
                'mw': 121.14
            }
        }
        
        return templates
    
    def get_pdbqt_by_interaction_type(self, interaction_type: str) -> List[Dict]:
        """根据相互作用类型获取PDBQT模板"""
        return [template for template in self.pdbqt_templates.values() 
                if template['interaction_type'] == interaction_type]
    
    def get_pdbqt_by_size(self, min_mw: float = 90, max_mw: float = 200) -> List[Dict]:
        """根据分子量范围获取PDBQT模板"""
        return [template for template in self.pdbqt_templates.values() 
                if min_mw <= template['mw'] <= max_mw]
    
    def get_all_templates(self) -> List[Dict]:
        """获取所有PDBQT模板"""
        return list(self.pdbqt_templates.values())
    
    def save_pdbqt_file(self, template_name: str, output_path: str) -> bool:
        """保存PDBQT文件"""
        try:
            if template_name not in self.pdbqt_templates:
                logger.error(f"未找到模板: {template_name}")
                return False
            
            template = self.pdbqt_templates[template_name]
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 写入PDBQT文件
            with open(output_path, 'w') as f:
                f.write(template['pdbqt_content'])
            
            logger.debug(f"保存PDBQT文件: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存PDBQT文件失败: {e}")
            return False
