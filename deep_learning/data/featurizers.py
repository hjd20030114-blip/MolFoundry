"""
分子和蛋白质特征化器
将原始数据转换为模型可用的特征表示
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.rdchem import Mol
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available. Molecular featurization will be limited.")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import distances
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    print("Warning: MDAnalysis not available. Protein analysis will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class FeaturizationConfig:
    """特征化配置"""
    # 分子特征
    use_morgan_fingerprints: bool = True
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    use_maccs_keys: bool = True
    use_descriptors: bool = True
    use_3d_descriptors: bool = False
    
    # 蛋白质特征
    use_amino_acid_properties: bool = True
    use_secondary_structure: bool = True
    use_surface_accessibility: bool = True
    pocket_radius: float = 5.0
    
    # 相互作用特征
    interaction_cutoff: float = 4.0
    use_pharmacophore: bool = True

class MolecularFeaturizer:
    """分子特征化器"""
    
    def __init__(self, config: FeaturizationConfig):
        self.config = config
        
        # 原子特征映射
        self.atom_features = {
            'atomic_num': list(range(1, 119)),  # 原子序数
            'degree': [0, 1, 2, 3, 4, 5],  # 度数
            'formal_charge': [-2, -1, 0, 1, 2],  # 形式电荷
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ] if HAS_RDKIT else [],
            'is_aromatic': [False, True],
            'is_in_ring': [False, True]
        }
        
        # 键特征映射
        self.bond_features = {
            'bond_type': [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ] if HAS_RDKIT else [],
            'is_conjugated': [False, True],
            'is_in_ring': [False, True]
        }
    
    def featurize_molecule(self, mol: Union[str, 'Mol']) -> Dict[str, torch.Tensor]:
        """分子特征化"""
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for molecular featurization")
        
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            raise ValueError("Invalid molecule")
        
        features = {}
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_feat = self._get_atom_features(atom)
            atom_features.append(atom_feat)
        
        features['atom_features'] = torch.tensor(atom_features, dtype=torch.float32)
        features['num_atoms'] = len(atom_features)
        
        # 键特征和邻接矩阵
        bond_features, edge_indices = self._get_bond_features(mol)
        features['bond_features'] = torch.tensor(bond_features, dtype=torch.float32)
        features['edge_indices'] = torch.tensor(edge_indices, dtype=torch.long)
        
        # 分子指纹
        if self.config.use_morgan_fingerprints:
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.config.morgan_radius, nBits=self.config.morgan_nbits
            )
            features['morgan_fingerprint'] = torch.tensor(
                np.array(morgan_fp), dtype=torch.float32
            )
        
        if self.config.use_maccs_keys:
            maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            features['maccs_fingerprint'] = torch.tensor(
                np.array(maccs_fp), dtype=torch.float32
            )
        
        # 分子描述符
        if self.config.use_descriptors:
            descriptors = self._calculate_descriptors(mol)
            features['descriptors'] = torch.tensor(descriptors, dtype=torch.float32)
        
        # 3D描述符（如果有3D坐标）
        if self.config.use_3d_descriptors and mol.GetNumConformers() > 0:
            descriptors_3d = self._calculate_3d_descriptors(mol)
            features['descriptors_3d'] = torch.tensor(descriptors_3d, dtype=torch.float32)
        
        # 分子坐标
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            features['coordinates'] = torch.tensor(coords, dtype=torch.float32)
        
        return features
    
    def _get_atom_features(self, atom) -> List[float]:
        """获取原子特征"""
        features = []
        
        # 原子序数 (one-hot)
        atomic_num = atom.GetAtomicNum()
        features.extend(self._one_hot_encode(atomic_num, self.atom_features['atomic_num']))
        
        # 度数
        degree = atom.GetDegree()
        features.extend(self._one_hot_encode(degree, self.atom_features['degree']))
        
        # 形式电荷
        formal_charge = atom.GetFormalCharge()
        features.extend(self._one_hot_encode(formal_charge, self.atom_features['formal_charge']))
        
        # 杂化类型
        hybridization = atom.GetHybridization()
        features.extend(self._one_hot_encode(hybridization, self.atom_features['hybridization']))
        
        # 芳香性
        is_aromatic = atom.GetIsAromatic()
        features.extend(self._one_hot_encode(is_aromatic, self.atom_features['is_aromatic']))
        
        # 是否在环中
        is_in_ring = atom.IsInRing()
        features.extend(self._one_hot_encode(is_in_ring, self.atom_features['is_in_ring']))
        
        # 其他特征
        features.append(atom.GetMass())  # 原子质量
        features.append(atom.GetTotalValence())  # 总价
        features.append(atom.GetNumRadicalElectrons())  # 自由基电子数
        
        return features
    
    def _get_bond_features(self, mol) -> Tuple[List[List[float]], List[List[int]]]:
        """获取键特征和边索引"""
        bond_features = []
        edge_indices = []
        
        for bond in mol.GetBonds():
            # 键特征
            bond_feat = []
            
            # 键类型
            bond_type = bond.GetBondType()
            bond_feat.extend(self._one_hot_encode(bond_type, self.bond_features['bond_type']))
            
            # 共轭性
            is_conjugated = bond.GetIsConjugated()
            bond_feat.extend(self._one_hot_encode(is_conjugated, self.bond_features['is_conjugated']))
            
            # 是否在环中
            is_in_ring = bond.IsInRing()
            bond_feat.extend(self._one_hot_encode(is_in_ring, self.bond_features['is_in_ring']))
            
            # 添加两个方向的边
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            bond_features.append(bond_feat)
            bond_features.append(bond_feat)  # 无向图，两个方向
            
            edge_indices.append([begin_atom, end_atom])
            edge_indices.append([end_atom, begin_atom])
        
        return bond_features, edge_indices
    
    def _calculate_descriptors(self, mol) -> List[float]:
        """计算分子描述符"""
        descriptors = []
        
        # 基本描述符
        descriptors.append(Descriptors.MolWt(mol))  # 分子量
        descriptors.append(Descriptors.MolLogP(mol))  # LogP
        descriptors.append(Descriptors.NumHDonors(mol))  # 氢键供体
        descriptors.append(Descriptors.NumHAcceptors(mol))  # 氢键受体
        descriptors.append(Descriptors.NumRotatableBonds(mol))  # 可旋转键
        descriptors.append(Descriptors.TPSA(mol))  # 拓扑极性表面积
        descriptors.append(Descriptors.NumAromaticRings(mol))  # 芳香环数
        descriptors.append(Descriptors.NumSaturatedRings(mol))  # 饱和环数
        descriptors.append(Descriptors.FractionCsp3(mol))  # sp3碳比例
        descriptors.append(Descriptors.BalabanJ(mol))  # Balaban指数
        
        return descriptors
    
    def _calculate_3d_descriptors(self, mol) -> List[float]:
        """计算3D描述符"""
        descriptors_3d = []
        
        try:
            # 3D描述符
            descriptors_3d.append(rdMolDescriptors.CalcPMI1(mol))  # 主惯性矩1
            descriptors_3d.append(rdMolDescriptors.CalcPMI2(mol))  # 主惯性矩2
            descriptors_3d.append(rdMolDescriptors.CalcPMI3(mol))  # 主惯性矩3
            descriptors_3d.append(rdMolDescriptors.CalcRadiusOfGyration(mol))  # 回转半径
            descriptors_3d.append(rdMolDescriptors.CalcInertialShapeFactor(mol))  # 惯性形状因子
            descriptors_3d.append(rdMolDescriptors.CalcEccentricity(mol))  # 偏心率
            descriptors_3d.append(rdMolDescriptors.CalcAsphericity(mol))  # 非球形度
            descriptors_3d.append(rdMolDescriptors.CalcSpherocityIndex(mol))  # 球形指数
        except:
            # 如果计算失败，用零填充
            descriptors_3d = [0.0] * 8
        
        return descriptors_3d
    
    def _one_hot_encode(self, value, choices: List) -> List[float]:
        """One-hot编码"""
        encoding = [0.0] * len(choices)
        try:
            index = choices.index(value)
            encoding[index] = 1.0
        except ValueError:
            # 如果值不在选择列表中，保持全零
            pass
        return encoding

class ProteinFeaturizer:
    """蛋白质特征化器"""
    
    def __init__(self, config: FeaturizationConfig):
        self.config = config
        
        # 氨基酸性质
        self.aa_properties = {
            'A': [0.31, -0.74, 0.0, 0.0, 0.0],  # 疏水性, 体积, 极性, 电荷, 芳香性
            'R': [-1.01, 0.06, 1.0, 1.0, 0.0],
            'N': [-0.60, -0.34, 1.0, 0.0, 0.0],
            'D': [-0.77, -0.54, 1.0, -1.0, 0.0],
            'C': [1.54, -0.04, 0.0, 0.0, 0.0],
            'Q': [-0.22, 0.58, 1.0, 0.0, 0.0],
            'E': [-0.64, 0.13, 1.0, -1.0, 0.0],
            'G': [0.0, -1.0, 0.0, 0.0, 0.0],
            'H': [0.13, 0.11, 1.0, 0.5, 1.0],
            'I': [1.80, 0.73, 0.0, 0.0, 0.0],
            'L': [1.70, 0.53, 0.0, 0.0, 0.0],
            'K': [-0.99, 0.30, 1.0, 1.0, 0.0],
            'M': [1.23, 0.52, 0.0, 0.0, 0.0],
            'F': [1.79, 0.35, 0.0, 0.0, 1.0],
            'P': [0.72, -0.07, 0.0, 0.0, 0.0],
            'S': [-0.04, -0.52, 1.0, 0.0, 0.0],
            'T': [0.26, -0.22, 1.0, 0.0, 0.0],
            'W': [2.25, 1.0, 0.0, 0.0, 1.0],
            'Y': [0.96, 0.17, 1.0, 0.0, 1.0],
            'V': [1.22, 0.07, 0.0, 0.0, 0.0]
        }
    
    def featurize_protein(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """蛋白质特征化"""
        if not HAS_MDANALYSIS:
            raise ImportError("MDAnalysis is required for protein featurization")
        
        # 加载蛋白质结构
        u = mda.Universe(pdb_file)
        protein = u.select_atoms("protein")
        
        features = {}
        
        # 残基特征
        residue_features = []
        residue_coords = []
        
        for residue in protein.residues:
            # 氨基酸类型特征
            aa_type = residue.resname
            if len(aa_type) == 3:
                aa_type = self._three_to_one_letter(aa_type)
            
            if aa_type in self.aa_properties:
                aa_feat = self.aa_properties[aa_type]
            else:
                aa_feat = [0.0] * 5  # 未知氨基酸
            
            residue_features.append(aa_feat)
            
            # 残基坐标（CA原子）
            ca_atoms = residue.atoms.select_atoms("name CA")
            if len(ca_atoms) > 0:
                ca_coord = ca_atoms.positions[0]
                residue_coords.append(ca_coord)
            else:
                residue_coords.append([0.0, 0.0, 0.0])
        
        features['residue_features'] = torch.tensor(residue_features, dtype=torch.float32)
        features['residue_coordinates'] = torch.tensor(residue_coords, dtype=torch.float32)
        features['num_residues'] = len(residue_features)
        
        # 原子特征
        atom_features = []
        atom_coords = []
        
        for atom in protein.atoms:
            # 原子类型特征（简化）
            element = atom.element
            atom_feat = self._get_atom_type_features(element)
            atom_features.append(atom_feat)
            atom_coords.append(atom.position)
        
        features['atom_features'] = torch.tensor(atom_features, dtype=torch.float32)
        features['atom_coordinates'] = torch.tensor(atom_coords, dtype=torch.float32)
        features['num_atoms'] = len(atom_features)
        
        return features
    
    def extract_pocket(
        self,
        protein_features: Dict[str, torch.Tensor],
        ligand_coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """提取结合口袋"""
        protein_coords = protein_features['atom_coordinates']
        
        # 计算距离
        distances = torch.cdist(ligand_coords, protein_coords)
        min_distances = distances.min(dim=0)[0]
        
        # 选择口袋内的原子
        pocket_mask = min_distances <= self.config.pocket_radius
        pocket_indices = torch.where(pocket_mask)[0]
        
        pocket_features = {
            'atom_features': protein_features['atom_features'][pocket_indices],
            'atom_coordinates': protein_features['atom_coordinates'][pocket_indices],
            'pocket_indices': pocket_indices,
            'num_pocket_atoms': len(pocket_indices)
        }
        
        return pocket_features
    
    def _three_to_one_letter(self, three_letter: str) -> str:
        """三字母氨基酸代码转单字母"""
        mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return mapping.get(three_letter, 'X')
    
    def _get_atom_type_features(self, element: str) -> List[float]:
        """获取原子类型特征"""
        # 简化的原子特征
        element_features = {
            'C': [1, 0, 0, 0, 0],
            'N': [0, 1, 0, 0, 0],
            'O': [0, 0, 1, 0, 0],
            'S': [0, 0, 0, 1, 0],
            'P': [0, 0, 0, 0, 1]
        }
        return element_features.get(element, [0, 0, 0, 0, 0])

class InteractionFeaturizer:
    """相互作用特征化器"""
    
    def __init__(self, config: FeaturizationConfig):
        self.config = config
    
    def featurize_interaction(
        self,
        protein_features: Dict[str, torch.Tensor],
        ligand_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """特征化蛋白质-配体相互作用"""
        
        protein_coords = protein_features['atom_coordinates']
        ligand_coords = ligand_features['coordinates']
        
        # 计算距离矩阵
        distance_matrix = torch.cdist(protein_coords, ligand_coords)
        
        # 识别相互作用
        interaction_mask = distance_matrix <= self.config.interaction_cutoff
        
        # 相互作用特征
        features = {
            'distance_matrix': distance_matrix,
            'interaction_mask': interaction_mask,
            'num_interactions': interaction_mask.sum().item()
        }
        
        # 相互作用指纹
        interaction_fingerprint = self._compute_interaction_fingerprint(
            protein_features, ligand_features, interaction_mask
        )
        features['interaction_fingerprint'] = interaction_fingerprint
        
        return features
    
    def _compute_interaction_fingerprint(
        self,
        protein_features: Dict[str, torch.Tensor],
        ligand_features: Dict[str, torch.Tensor],
        interaction_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算相互作用指纹"""
        # 简化的相互作用指纹
        fingerprint = torch.zeros(1024)  # 1024位指纹
        
        # 基于相互作用类型设置指纹位
        # 这里是简化实现，实际应该更复杂
        num_interactions = interaction_mask.sum().item()
        if num_interactions > 0:
            # 设置一些位表示有相互作用
            fingerprint[:min(num_interactions, 1024)] = 1.0
        
        return fingerprint
