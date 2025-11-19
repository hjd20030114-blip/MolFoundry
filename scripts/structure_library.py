#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预生成的3D结构库
包含已知能够成功生成3D构象的药物类似分子
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class StructureLibrary:
    """预生成的3D结构库"""
    
    def __init__(self):
        """初始化结构库"""
        self.drug_like_molecules = self._load_drug_like_molecules()
        logger.info(f"加载了 {len(self.drug_like_molecules)} 个药物类似分子")
    
    def _load_drug_like_molecules(self) -> List[Dict]:
        """加载药物类似分子库"""
        # 这些分子都是经过验证能够成功生成3D构象的
        molecules = [
            {
                'smiles': 'CCc1ccc(cc1)C(=O)O',
                'name': 'Ibuprofen-like',
                'mw': 178.23,
                'logp': 3.5,
                'interaction_type': 'hydrophobic'
            },
            {
                'smiles': 'COc1ccc(cc1)CCN',
                'name': 'Mescaline-like',
                'mw': 181.23,
                'logp': 1.2,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'Cc1ccc(cc1)C(=O)Nc2ccccc2',
                'name': 'Tolyl-benzamide',
                'mw': 211.26,
                'logp': 3.1,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'CCOc1ccc(cc1)C(=O)N',
                'name': 'Ethoxy-benzamide',
                'mw': 179.22,
                'logp': 1.8,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'Cc1nc2ccccc2c(=O)n1C',
                'name': 'Methylquinazolone',
                'mw': 188.23,
                'logp': 1.5,
                'interaction_type': 'van_der_waals'
            },
            {
                'smiles': 'COc1ccc2c(c1)c(C)cn2C',
                'name': 'Methoxyindole',
                'mw': 175.23,
                'logp': 2.1,
                'interaction_type': 'hydrophobic'
            },
            {
                'smiles': 'Cc1ccc2c(c1)nc(C)n2C',
                'name': 'Dimethylbenzimidazole',
                'mw': 161.20,
                'logp': 1.7,
                'interaction_type': 'van_der_waals'
            },
            {
                'smiles': 'CCc1ccc(cc1)OCc2ccccc2',
                'name': 'Ethylphenyl-benzylether',
                'mw': 212.29,
                'logp': 4.2,
                'interaction_type': 'hydrophobic'
            },
            {
                'smiles': 'COc1ccc(cc1)CCc2ccccc2',
                'name': 'Methoxyphenethyl-benzene',
                'mw': 212.29,
                'logp': 3.8,
                'interaction_type': 'hydrophobic'
            },
            {
                'smiles': 'Cc1ccc(cc1)COc2ccccc2C',
                'name': 'Tolyl-cresyl-ether',
                'mw': 212.29,
                'logp': 3.9,
                'interaction_type': 'hydrophobic'
            },
            {
                'smiles': 'CCc1ccc(cc1)C(=O)NCc2ccccc2',
                'name': 'Ethylbenzoyl-benzylamine',
                'mw': 253.34,
                'logp': 3.7,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'COc1ccc(cc1)C(=O)OCC',
                'name': 'Methoxy-benzoate-ethyl',
                'mw': 180.20,
                'logp': 2.3,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'Cc1ccc(cc1)C(=O)N(C)C',
                'name': 'Tolyl-dimethylamide',
                'mw': 163.22,
                'logp': 1.9,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'c1ccc(cc1)C(=O)N[C@@H](Cc2ccccc2)C(=O)O',
                'name': 'Benzoyl-phenylalanine',
                'mw': 269.30,
                'logp': 2.8,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'COc1ccc(cc1)C(=O)N[C@@H](C)C(=O)O',
                'name': 'Methoxybenzoyl-alanine',
                'mw': 223.23,
                'logp': 1.2,
                'interaction_type': 'hydrogen_bond'
            },
            {
                'smiles': 'Cc1ccc(cc1)S(=O)(=O)N[C@@H](C)C(=O)O',
                'name': 'Tosyl-alanine',
                'mw': 243.28,
                'logp': 0.8,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'c1ccc2c(c1)nc(N)nc2N',
                'name': 'Diaminoquinazoline',
                'mw': 160.18,
                'logp': -0.5,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'COc1ccc2c(c1)nc(N)nc2N',
                'name': 'Methoxydiaminoquinazoline',
                'mw': 190.20,
                'logp': 0.2,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'Cc1nc2ccccc2c(=O)n1C',
                'name': 'Methylquinazolone',
                'mw': 188.23,
                'logp': 1.5,
                'interaction_type': 'van_der_waals'
            },
            {
                'smiles': 'CCc1ccc2nc(C)nc(N)c2c1',
                'name': 'Ethylaminoquinazoline',
                'mw': 201.27,
                'logp': 1.8,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'Cc1ccc(cc1)c2nc(N)nc(N)n2',
                'name': 'Tolyl-triazine-diamine',
                'mw': 201.23,
                'logp': 1.1,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'COc1ccc(cc1)c2nnc(N)s2',
                'name': 'Methoxyphenyl-thiadiazole',
                'mw': 207.24,
                'logp': 1.9,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'Cc1ccc(cc1)C2=NN=C(N)S2',
                'name': 'Tolyl-thiadiazole-amine',
                'mw': 191.25,
                'logp': 1.7,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'CCc1ccc(cc1)c2nc(C)nc(N)c2',
                'name': 'Ethylphenyl-pyrimidine',
                'mw': 215.30,
                'logp': 2.3,
                'interaction_type': 'electrostatic'
            },
            {
                'smiles': 'COc1ccc(cc1)c2nc(N)nc(C)c2',
                'name': 'Methoxyphenyl-pyrimidine',
                'mw': 201.23,
                'logp': 1.5,
                'interaction_type': 'electrostatic'
            }
        ]
        
        return molecules
    
    def get_molecules_by_interaction_type(self, interaction_type: str) -> List[Dict]:
        """根据相互作用类型获取分子"""
        return [mol for mol in self.drug_like_molecules if mol['interaction_type'] == interaction_type]
    
    def get_molecules_by_size(self, min_mw: float = 150, max_mw: float = 300) -> List[Dict]:
        """根据分子量范围获取分子"""
        return [mol for mol in self.drug_like_molecules if min_mw <= mol['mw'] <= max_mw]
    
    def get_molecules_by_logp(self, min_logp: float = -1, max_logp: float = 5) -> List[Dict]:
        """根据LogP范围获取分子"""
        return [mol for mol in self.drug_like_molecules if min_logp <= mol['logp'] <= max_logp]
    
    def get_filtered_molecules(self, interaction_type: str = None, 
                             min_mw: float = 150, max_mw: float = 300,
                             min_logp: float = -1, max_logp: float = 5) -> List[Dict]:
        """根据多个条件筛选分子"""
        filtered = self.drug_like_molecules
        
        if interaction_type:
            filtered = [mol for mol in filtered if mol['interaction_type'] == interaction_type]
        
        filtered = [mol for mol in filtered if min_mw <= mol['mw'] <= max_mw]
        filtered = [mol for mol in filtered if min_logp <= mol['logp'] <= max_logp]
        
        return filtered
    
    def get_random_molecules(self, count: int = 10) -> List[Dict]:
        """随机获取指定数量的分子"""
        import random
        return random.sample(self.drug_like_molecules, min(count, len(self.drug_like_molecules)))
