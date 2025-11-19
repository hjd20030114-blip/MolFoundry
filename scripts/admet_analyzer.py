# -*- coding: utf-8 -*-
# type: ignore
"""
PRRSV病毒衣壳蛋白抑制剂ADMET分析模块
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from .config import ADMET_CRITERIA, RESULTS_DIR, OUTPUT_FILES
except ImportError:
    from config import ADMET_CRITERIA, RESULTS_DIR, OUTPUT_FILES

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDKit导入处理
RDKIT_AVAILABLE = False
PAINS_CATALOG = None
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import QED
    from rdkit.Chem import FilterCatalog
    RDKIT_AVAILABLE = True
    # 初始化PAINS筛查目录
    try:
        _params = FilterCatalog.FilterCatalogParams()
        _params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        PAINS_CATALOG = FilterCatalog.FilterCatalog(_params)
    except Exception:
        PAINS_CATALOG = None
    logger.info("RDKit已成功导入，将使用完整ADMET分析功能")
except ImportError:
    logger.warning("RDKit不可用，将使用简化模式进行ADMET分析")
    logger.info("如需完整功能，请安装RDKit: pip install rdkit-pypi")
    # 创建虚拟的Chem和Descriptors模块以避免错误
    class MockChem:
        @staticmethod
        def MolFromSmiles(smiles):
            return None
    
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

class ADMETAnalyzer:
    """ADMET分析器类"""
    
    def __init__(self):
        """初始化ADMET分析器"""
        self.results = []
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    def calculate_admet_properties(self, smiles: str) -> Optional[Dict]:
        """计算单个分子的ADMET性质"""
        if not RDKIT_AVAILABLE:
            return self._calculate_simple_properties(smiles)
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 验证分子
            if not self._is_valid_molecule(mol):
                return None
            
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
                "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
            }
            
            # 添加额外的药物性质
            try:
                properties["molar_refractivity"] = Descriptors.MolMR(mol)
                properties["heavy_atom_count"] = Descriptors.HeavyAtomCount(mol)
                properties["heteroatom_count"] = Descriptors.NumHeteroatoms(mol)
                properties["fraction_csp3"] = Descriptors.FractionCsp3(mol)
                # SlogP_VSA1 为基于SlogP分箱的VSA片段之一，避免误导命名
                properties["slogp_vsa1"] = Descriptors.SlogP_VSA1(mol)
                # QED（药物相似性评分）
                try:
                    properties["qed"] = float(QED.qed(mol))
                except Exception:
                    properties["qed"] = np.nan
            except:
                # 如果某些描述符不可用，使用默认值
                properties["molar_refractivity"] = 0.0
                properties["heavy_atom_count"] = 0
                properties["heteroatom_count"] = 0
                properties["fraction_csp3"] = 0.0
                properties["slogp_vsa1"] = 0.0
                properties["qed"] = np.nan
            
            # Lipinski规则检查
            lipinski_violations = self._check_lipinski_rules(mol)
            properties["lipinski_compliant"] = lipinski_violations <= 1
            properties["lipinski_violations"] = lipinski_violations
            try:
                properties["lipinski_violation_details"] = ",".join(self._lipinski_violation_details(mol))
            except Exception:
                properties["lipinski_violation_details"] = ""

            # Veber/Egan 规则
            try:
                properties["veber_compliant"] = (properties["tpsa"] <= 140) and (properties["rotatable_bonds"] <= 10)
                properties["egan_compliant"] = (properties["logp"] <= 5.88) and (properties["tpsa"] <= 131)
            except Exception:
                properties["veber_compliant"] = False
                properties["egan_compliant"] = False

            # 溶解度预测（ESOL近似公式）
            try:
                ap = 0.0
                try:
                    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
                    heavy = properties.get("heavy_atom_count", 0) or Descriptors.HeavyAtomCount(mol)
                    ap = float(aromatic_atoms) / float(heavy) if heavy > 0 else 0.0
                except Exception:
                    ap = 0.0

                mw = properties.get("molecular_weight", 0.0)
                logp = properties.get("logp", 0.0)
                rb = properties.get("rotatable_bonds", 0)
                logS = 0.16 - 0.63 * float(logp) - 0.0062 * float(mw) + 0.066 * float(rb) + 0.74 * float(ap)
                properties["predicted_logS"] = float(logS)
                # 分类（越大越易溶）
                if logS > 0.5:
                    sol_class = "极易溶"
                elif logS > 0.0:
                    sol_class = "易溶"
                elif logS > -2.0:
                    sol_class = "中等溶解度"
                elif logS > -4.0:
                    sol_class = "低溶解度"
                else:
                    sol_class = "极低溶解度"
                properties["solubility_class"] = sol_class
            except Exception:
                properties["predicted_logS"] = 0.0
                properties["solubility_class"] = "未知"

            # 毒性结构警示（简单SMARTS筛查）
            try:
                alerts = []
                patterns = {
                    "硝基": "[N+](=O)[O-]",
                    "偶氮": "N=N",
                    "烯酮(Michael受体)": "C=CC(=O)",
                    "烷基卤化物": "[CX4][Cl,Br,I]",
                    "环氧": "C1OC1",
                    "仲胺-芳胺": "[a][N;H1]",
                    "硫脲": "NC(=S)N"
                }
                for tag, smarts in patterns.items():
                    try:
                        patt = Chem.MolFromSmarts(smarts)
                        if patt is not None and mol.HasSubstructMatch(patt):
                            alerts.append(tag)
                    except Exception:
                        continue
                # PAINS 筛查
                pains_alerts = []
                if PAINS_CATALOG is not None:
                    try:
                        matches = PAINS_CATALOG.GetMatches(mol)
                        for m in matches:
                            pains_alerts.append(m.GetDescription())
                    except Exception:
                        pass
                risk_level = "低"
                total_alerts = len(alerts) + len(pains_alerts)
                if total_alerts >= 3:
                    risk_level = "高"
                elif total_alerts >= 1:
                    risk_level = "中"
                properties["toxicity_risk_level"] = risk_level
                properties["toxicity_alerts_count"] = total_alerts
                properties["toxicity_alerts"] = ",".join(alerts + pains_alerts)
            except Exception:
                properties["toxicity_risk_level"] = "未知"
                properties["toxicity_alerts_count"] = 0
                properties["toxicity_alerts"] = ""
            
            return properties
            
        except Exception as e:
            logger.error(f"计算ADMET性质时出错: {e}")
            return None
    
    def _is_valid_molecule(self, mol) -> bool:
        """验证分子是否有效"""
        try:
            if mol is None:
                return False
            
            # 检查分子是否可以被清理
            mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol_clean is None:
                return False
            
            # 放宽分子大小限制
            if mol_clean.GetNumAtoms() < 3 or mol_clean.GetNumAtoms() > 150:
                return False
            
            # 简化价态检查，避免弃用警告
            try:
                # 尝试计算分子性质，如果失败说明分子无效
                Descriptors.MolWt(mol_clean)
                return True
            except:
                return False
            
        except Exception:
            return False
    
    def _calculate_simple_properties(self, smiles: str) -> Optional[Dict]:
        """简化模式下的性质计算"""
        try:
            # 基础计算
            properties = {
                "molecular_weight": self._estimate_molecular_weight(smiles),
                "logp": self._estimate_logp(smiles),
                "hbd": self._count_hydrogen_bond_donors(smiles),
                "hba": self._count_hydrogen_bond_acceptors(smiles),
                "rotatable_bonds": self._count_rotatable_bonds(smiles),
                "rings": self._count_rings(smiles),
                "aromatic_rings": self._count_aromatic_rings(smiles),
                "tpsa": self._estimate_tpsa(smiles),
                "lipinski_compliant": True,  # 简化模式默认符合
                "canonical_smiles": smiles,
                "qed": np.nan,
            }

            # 溶解度预测（使用估算的理化性质）
            try:
                mw = properties["molecular_weight"]
                logp = properties["logp"]
                rb = properties["rotatable_bonds"]
                # 简易芳香比例近似
                aromatic_atoms_approx = smiles.count('c')
                heavy_approx = sum(smiles.count(x) for x in ['C','N','O','S','F']) + smiles.count('Cl')
                ap = float(aromatic_atoms_approx) / float(heavy_approx) if heavy_approx > 0 else 0.0
                logS = 0.16 - 0.63 * float(logp) - 0.0062 * float(mw) + 0.066 * float(rb) + 0.74 * float(ap)
                properties["predicted_logS"] = float(logS)
                if logS > 0.5:
                    sol_class = "极易溶"
                elif logS > 0.0:
                    sol_class = "易溶"
                elif logS > -2.0:
                    sol_class = "中等溶解度"
                elif logS > -4.0:
                    sol_class = "低溶解度"
                else:
                    sol_class = "极低溶解度"
                properties["solubility_class"] = sol_class
            except Exception:
                properties["predicted_logS"] = 0.0
                properties["solubility_class"] = "未知"

            # Veber/Egan 规则（基于估算值）
            try:
                properties["veber_compliant"] = (properties["tpsa"] <= 140) and (properties["rotatable_bonds"] <= 10)
                properties["egan_compliant"] = (properties["logp"] <= 5.88) and (properties["tpsa"] <= 131)
            except Exception:
                properties["veber_compliant"] = False
                properties["egan_compliant"] = False

            # 毒性结构警示（字符串启发式）
            try:
                alerts = []
                checks = {
                    "硝基": "[N+](=O)[O-]",
                    "偶氮": "N=N",
                    "烯酮(Michael受体)": "C=CC(=O)",
                    "烷基卤化物_Cl": "CCl",
                    "烷基卤化物_Br": "CBr",
                    "烷基卤化物_I": "CI",
                    "环氧": "C1OC1",
                }
                for tag, key in checks.items():
                    if key in smiles:
                        alerts.append(tag)
                risk_level = "低"
                if len(alerts) >= 3:
                    risk_level = "高"
                elif len(alerts) >= 1:
                    risk_level = "中"
                properties["toxicity_risk_level"] = risk_level
                properties["toxicity_alerts_count"] = len(alerts)
                properties["toxicity_alerts"] = ",".join(alerts)
            except Exception:
                properties["toxicity_risk_level"] = "未知"
                properties["toxicity_alerts_count"] = 0
                properties["toxicity_alerts"] = ""
            return properties
            
        except Exception as e:
            logger.error(f"简化性质计算时出错: {e}")
            return None
    
    def _estimate_molecular_weight(self, smiles: str) -> float:
        """估算分子量"""
        atomic_masses = {'C': 12.01, 'H': 1.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'F': 19.00, 'Cl': 35.45}
        total_mass = sum(atomic_masses.get(atom, 0) * smiles.count(atom) for atom in atomic_masses)
        return total_mass
    
    def _estimate_logp(self, smiles: str) -> float:
        """估算LogP值"""
        logp = 0.0
        # 芳香环、烷基增加疏水性
        logp += smiles.count('c1ccccc1') * 1.8
        logp += smiles.count('C') * 0.02
        # 含氧、含氮降低疏水性
        logp -= smiles.count('O') * 0.8
        logp -= smiles.count('N') * 0.5
        # 含硫/卤素略增
        logp += smiles.count('S') * 0.3
        logp += smiles.count('F') * 0.2
        logp += smiles.count('Cl') * 0.4
        logp += smiles.count('Br') * 0.6
        logp += smiles.count('I') * 0.8
        return float(logp)

    def _lipinski_violation_details(self, mol) -> List[str]:
        """返回违反的Lipinski条目"""
        details = []
        try:
            if Descriptors.MolWt(mol) > 500: details.append('MW>500')
            if Descriptors.MolLogP(mol) > 5: details.append('LogP>5')
            if Descriptors.NumHDonors(mol) > 5: details.append('HBD>5')
            if Descriptors.NumHAcceptors(mol) > 10: details.append('HBA>10')
        except Exception:
            pass
        return details
    
    def _count_hydrogen_bond_donors(self, smiles: str) -> int:
        """计算氢键供体数"""
        return smiles.count('O') + smiles.count('N') + smiles.count('S')
    
    def _count_hydrogen_bond_acceptors(self, smiles: str) -> int:
        """计算氢键受体数"""
        return smiles.count('O') + smiles.count('N') + smiles.count('S')
    
    def _count_rotatable_bonds(self, smiles: str) -> int:
        """计算可旋转键数"""
        return smiles.count('C') // 4
    
    def _count_rings(self, smiles: str) -> int:
        """计算环数"""
        return smiles.count('1') // 2
    
    def _count_aromatic_rings(self, smiles: str) -> int:
        """计算芳香环数"""
        return smiles.count('c1ccccc1')
    
    def _estimate_tpsa(self, smiles: str) -> float:
        """估算拓扑极性表面积"""
        tpsa = 0.0
        tpsa += smiles.count('O') * 20.0
        tpsa += smiles.count('N') * 17.0
        tpsa += smiles.count('S') * 38.0
        return tpsa
    
    def _check_lipinski_rules(self, mol) -> int:
        """检查Lipinski规则"""
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        return violations
    
    def batch_admet_analysis(self, ligands_data: List[Dict]) -> pd.DataFrame:
        """批量ADMET分析"""
        logger.info(f"开始ADMET分析 {len(ligands_data)} 个配体...")
        
        results = []
        for i, ligand_data in enumerate(ligands_data):
            try:
                smiles = ligand_data["smiles"]
                admet_properties = self.calculate_admet_properties(smiles)
                
                if admet_properties:
                    result_data = {
                        "ligand_id": ligand_data.get("ligand_id", f"ligand_{i+1}"),
                        "smiles": smiles,
                        **ligand_data,
                        **admet_properties
                    }
                    results.append(result_data)
                    
            except Exception as e:
                logger.error(f"分析配体 {i+1} 时出错: {e}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            logger.info(f"ADMET分析完成，成功分析 {len(results)} 个配体")
            return df
        else:
            logger.warning("没有成功的ADMET分析结果")
            return pd.DataFrame()
    
    def save_admet_results(self, results_df: pd.DataFrame, report: Dict):
        """保存ADMET分析结果"""
        try:
            results_file = os.path.join(RESULTS_DIR, OUTPUT_FILES["admet_results"])
            results_df.to_csv(results_file, index=False)
            logger.info(f"ADMET结果已保存到: {results_file}")
        except Exception as e:
            logger.error(f"保存ADMET结果时出错: {e}")

# 注意：本模块无测试入口，作为库供其他流程调用。