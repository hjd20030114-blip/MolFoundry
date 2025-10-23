"""
分子质量评估器
评估生成分子的质量、多样性、成药性等
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import inspect

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen
    from rdkit.Contrib.SA_Score import sascorer
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

logger = logging.getLogger(__name__)

class MolecularEvaluator:
    """分子质量评估器"""
    
    def __init__(self):
        if not HAS_RDKIT:
            logger.warning("RDKit not available. Some evaluation functions will be limited.")
    
    def evaluate_binding_affinity_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """评估结合亲和力预测模型"""
        model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 将数据移到设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                # 仅传入模型 forward 接受的参数，排除监督目标等
                try:
                    sig = inspect.signature(model.forward)
                    allowed = set(sig.parameters.keys())
                    model_inputs = {k: v for k, v in batch.items() if k in allowed}
                except (ValueError, TypeError):
                    # 回退：明确排除常见监督键
                    exclude = {'binding_affinity', 'labels', 'targets'}
                    model_inputs = {k: v for k, v in batch.items() if k not in exclude}

                outputs = model(**model_inputs)

                if 'binding_affinity' in outputs:
                    pred = outputs['binding_affinity'].cpu().numpy()
                    target = batch['binding_affinity'].cpu().numpy()
                    
                    predictions.extend(pred.flatten())
                    targets.extend(target.flatten())
        
        if not predictions:
            return {'error': 'No predictions available'}
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算评估指标
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # 相关系数
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # R²分数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation),
            'r2_score': float(r2_score),
            'num_samples': len(predictions)
        }
    
    def evaluate_generated_molecules(
        self,
        molecules: List[str],
        reference_molecules: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """评估生成分子的质量"""
        if not HAS_RDKIT:
            return {'error': 'RDKit not available'}
        
        results = {}
        
        # 有效性 (Validity)
        valid_molecules = []
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules.append(smiles)
        
        validity = len(valid_molecules) / len(molecules) if molecules else 0
        results['validity'] = validity
        
        if not valid_molecules:
            return results
        
        # 唯一性 (Uniqueness)
        unique_molecules = list(set(valid_molecules))
        uniqueness = len(unique_molecules) / len(valid_molecules)
        results['uniqueness'] = uniqueness
        
        # 新颖性 (Novelty)
        if reference_molecules:
            reference_set = set(reference_molecules)
            novel_molecules = [mol for mol in unique_molecules if mol not in reference_set]
            novelty = len(novel_molecules) / len(unique_molecules)
            results['novelty'] = novelty
        
        # 分子性质统计
        properties = self._calculate_molecular_properties(unique_molecules)
        results.update(properties)
        
        return results
    
    def _calculate_molecular_properties(self, molecules: List[str]) -> Dict[str, float]:
        """计算分子性质统计"""
        if not HAS_RDKIT:
            return {}
        
        properties = {
            'molecular_weight': [],
            'logp': [],
            'hbd': [],  # 氢键供体
            'hba': [],  # 氢键受体
            'tpsa': [],  # 拓扑极性表面积
            'rotatable_bonds': [],
            'qed_score': [],
            'sa_score': []
        }
        
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                # 基本描述符
                properties['molecular_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Crippen.MolLogP(mol))
                properties['hbd'].append(Descriptors.NumHDonors(mol))
                properties['hba'].append(Descriptors.NumHAcceptors(mol))
                properties['tpsa'].append(Descriptors.TPSA(mol))
                properties['rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
                
                # QED分数
                qed_score = QED.qed(mol)
                properties['qed_score'].append(qed_score)
                
                # SA分数
                sa_score = sascorer.calculateScore(mol)
                properties['sa_score'].append(sa_score)
                
            except Exception as e:
                logger.warning(f"Error calculating properties for {smiles}: {e}")
                continue
        
        # 计算统计量
        stats = {}
        for prop, values in properties.items():
            if values:
                stats[f'{prop}_mean'] = np.mean(values)
                stats[f'{prop}_std'] = np.std(values)
                stats[f'{prop}_min'] = np.min(values)
                stats[f'{prop}_max'] = np.max(values)
        
        return stats
    
    def calculate_lipinski_compliance(self, molecules: List[str]) -> Dict[str, float]:
        """计算Lipinski规则符合性"""
        if not HAS_RDKIT:
            return {'error': 'RDKit not available'}
        
        compliant_count = 0
        total_count = 0
        
        violations = {
            'mw_violations': 0,      # 分子量 > 500
            'logp_violations': 0,    # LogP > 5
            'hbd_violations': 0,     # HBD > 5
            'hba_violations': 0      # HBA > 10
        }
        
        for smiles in molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            total_count += 1
            
            try:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # 检查违规
                violation_count = 0
                if mw > 500:
                    violations['mw_violations'] += 1
                    violation_count += 1
                if logp > 5:
                    violations['logp_violations'] += 1
                    violation_count += 1
                if hbd > 5:
                    violations['hbd_violations'] += 1
                    violation_count += 1
                if hba > 10:
                    violations['hba_violations'] += 1
                    violation_count += 1
                
                # Lipinski规则允许最多1个违规
                if violation_count <= 1:
                    compliant_count += 1
                    
            except Exception as e:
                logger.warning(f"Error calculating Lipinski compliance for {smiles}: {e}")
                continue
        
        compliance_rate = compliant_count / total_count if total_count > 0 else 0
        
        results = {
            'lipinski_compliance_rate': compliance_rate,
            'total_molecules': total_count,
            'compliant_molecules': compliant_count
        }
        
        # 添加违规统计
        for violation_type, count in violations.items():
            results[violation_type] = count
            results[f'{violation_type}_rate'] = count / total_count if total_count > 0 else 0
        
        return results
    
    def calculate_diversity_metrics(self, molecules: List[str]) -> Dict[str, float]:
        """计算分子多样性指标"""
        if not HAS_RDKIT:
            return {'error': 'RDKit not available'}
        
        try:
            from rdkit.Chem import rdMolDescriptors
            from rdkit import DataStructs
            
            # 计算分子指纹
            fingerprints = []
            for smiles in molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints.append(fp)
            
            if len(fingerprints) < 2:
                return {'diversity_score': 0.0}
            
            # 计算成对相似性
            similarities = []
            for i in range(len(fingerprints)):
                for j in range(i + 1, len(fingerprints)):
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    similarities.append(similarity)
            
            # 多样性 = 1 - 平均相似性
            avg_similarity = np.mean(similarities)
            diversity_score = 1 - avg_similarity
            
            return {
                'diversity_score': diversity_score,
                'average_similarity': avg_similarity,
                'num_comparisons': len(similarities)
            }
            
        except ImportError:
            return {'error': 'Required RDKit modules not available'}
        except Exception as e:
            logger.error(f"Error calculating diversity metrics: {e}")
            return {'error': str(e)}
    
    def generate_evaluation_report(
        self,
        molecules: List[str],
        reference_molecules: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """生成完整的评估报告"""
        report = {}
        
        # 基本质量评估
        quality_metrics = self.evaluate_generated_molecules(molecules, reference_molecules)
        report['quality_metrics'] = quality_metrics
        
        # Lipinski规则符合性
        lipinski_metrics = self.calculate_lipinski_compliance(molecules)
        report['lipinski_metrics'] = lipinski_metrics
        
        # 多样性评估
        diversity_metrics = self.calculate_diversity_metrics(molecules)
        report['diversity_metrics'] = diversity_metrics
        
        # 生成摘要
        summary = self._generate_summary(report)
        report['summary'] = summary
        
        # 保存报告
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_summary(self, report: Dict) -> Dict[str, str]:
        """生成评估摘要"""
        summary = {}
        
        quality = report.get('quality_metrics', {})
        lipinski = report.get('lipinski_metrics', {})
        diversity = report.get('diversity_metrics', {})
        
        # 质量评级
        validity = quality.get('validity', 0)
        if validity >= 0.9:
            summary['validity_grade'] = "优秀"
        elif validity >= 0.7:
            summary['validity_grade'] = "良好"
        elif validity >= 0.5:
            summary['validity_grade'] = "一般"
        else:
            summary['validity_grade'] = "较差"
        
        # 成药性评级
        compliance_rate = lipinski.get('lipinski_compliance_rate', 0)
        if compliance_rate >= 0.8:
            summary['drug_likeness_grade'] = "优秀"
        elif compliance_rate >= 0.6:
            summary['drug_likeness_grade'] = "良好"
        elif compliance_rate >= 0.4:
            summary['drug_likeness_grade'] = "一般"
        else:
            summary['drug_likeness_grade'] = "较差"
        
        # 多样性评级
        diversity_score = diversity.get('diversity_score', 0)
        if diversity_score >= 0.7:
            summary['diversity_grade'] = "优秀"
        elif diversity_score >= 0.5:
            summary['diversity_grade'] = "良好"
        elif diversity_score >= 0.3:
            summary['diversity_grade'] = "一般"
        else:
            summary['diversity_grade'] = "较差"
        
        return summary
