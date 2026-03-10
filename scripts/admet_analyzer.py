# -*- coding: utf-8 -*-
# type: ignore
"""
ADMET analysis module for PRRSV nucleocapsid protein inhibitors.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from .config import ADMET_CRITERIA, RESULTS_DIR, OUTPUT_FILES
except ImportError:
    from config import ADMET_CRITERIA, RESULTS_DIR, OUTPUT_FILES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDKit import handling
RDKIT_AVAILABLE = False
PAINS_CATALOG = None
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import QED
    from rdkit.Chem import FilterCatalog
    RDKIT_AVAILABLE = True
    # Initialize PAINS filter catalog
    try:
        _params = FilterCatalog.FilterCatalogParams()
        _params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        PAINS_CATALOG = FilterCatalog.FilterCatalog(_params)
    except Exception:
        PAINS_CATALOG = None
    logger.info("RDKit successfully imported, full ADMET analysis enabled")
except ImportError:
    logger.warning("RDKit not available, using simplified ADMET analysis mode")
    logger.info("For full functionality, install RDKit: pip install rdkit-pypi")
    # Create mock Chem and Descriptors modules to avoid errors
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
    """ADMET analyzer class."""
    
    def __init__(self):
        """Initialize ADMET analyzer."""
        self.results = []
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    def calculate_admet_properties(self, smiles: str) -> Optional[Dict]:
        """Calculate ADMET properties for a single molecule."""
        if not RDKIT_AVAILABLE:
            return self._calculate_simple_properties(smiles)
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Validate molecule
            if not self._is_valid_molecule(mol):
                return None
            
            # Calculate basic properties
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
            
            # Add additional drug-like properties
            try:
                properties["molar_refractivity"] = Descriptors.MolMR(mol)
                properties["heavy_atom_count"] = Descriptors.HeavyAtomCount(mol)
                properties["heteroatom_count"] = Descriptors.NumHeteroatoms(mol)
                properties["fraction_csp3"] = Descriptors.FractionCsp3(mol)
                # SlogP_VSA1: one of the VSA fragments binned by SlogP
                properties["slogp_vsa1"] = Descriptors.SlogP_VSA1(mol)
                # QED (Quantitative Estimate of Drug-likeness)
                try:
                    properties["qed"] = float(QED.qed(mol))
                except Exception:
                    properties["qed"] = np.nan
            except:
                # If some descriptors are unavailable, use default values
                properties["molar_refractivity"] = 0.0
                properties["heavy_atom_count"] = 0
                properties["heteroatom_count"] = 0
                properties["fraction_csp3"] = 0.0
                properties["slogp_vsa1"] = 0.0
                properties["qed"] = np.nan
            
            # Lipinski rule-of-five check
            lipinski_violations = self._check_lipinski_rules(mol)
            properties["lipinski_compliant"] = lipinski_violations <= 1
            properties["lipinski_violations"] = lipinski_violations
            try:
                properties["lipinski_violation_details"] = ",".join(self._lipinski_violation_details(mol))
            except Exception:
                properties["lipinski_violation_details"] = ""

            # Veber/Egan rules
            try:
                properties["veber_compliant"] = (properties["tpsa"] <= 140) and (properties["rotatable_bonds"] <= 10)
                properties["egan_compliant"] = (properties["logp"] <= 5.88) and (properties["tpsa"] <= 131)
            except Exception:
                properties["veber_compliant"] = False
                properties["egan_compliant"] = False

            # Solubility prediction (ESOL approximation formula)
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
                # Classification (higher = more soluble)
                if logS > 0.5:
                    sol_class = "Highly Soluble"
                elif logS > 0.0:
                    sol_class = "Soluble"
                elif logS > -2.0:
                    sol_class = "Moderately Soluble"
                elif logS > -4.0:
                    sol_class = "Poorly Soluble"
                else:
                    sol_class = "Insoluble"
                properties["solubility_class"] = sol_class
            except Exception:
                properties["predicted_logS"] = 0.0
                properties["solubility_class"] = "Unknown"

            # Toxicity structural alerts (simple SMARTS screening)
            try:
                alerts = []
                patterns = {
                    "Nitro": "[N+](=O)[O-]",
                    "Azo": "N=N",
                    "Enone (Michael acceptor)": "C=CC(=O)",
                    "Alkyl halide": "[CX4][Cl,Br,I]",
                    "Epoxide": "C1OC1",
                    "Secondary/aromatic amine": "[a][N;H1]",
                    "Thiourea": "NC(=S)N"
                }
                for tag, smarts in patterns.items():
                    try:
                        patt = Chem.MolFromSmarts(smarts)
                        if patt is not None and mol.HasSubstructMatch(patt):
                            alerts.append(tag)
                    except Exception:
                        continue
        # PAINS screening
                pains_alerts = []
                if PAINS_CATALOG is not None:
                    try:
                        matches = PAINS_CATALOG.GetMatches(mol)
                        for m in matches:
                            pains_alerts.append(m.GetDescription())
                    except Exception:
                        pass
                risk_level = "Low"
                total_alerts = len(alerts) + len(pains_alerts)
                if total_alerts >= 3:
                    risk_level = "High"
                elif total_alerts >= 1:
                    risk_level = "Medium"
                properties["toxicity_risk_level"] = risk_level
                properties["toxicity_alerts_count"] = total_alerts
                properties["toxicity_alerts"] = ",".join(alerts + pains_alerts)
            except Exception:
                properties["toxicity_risk_level"] = "Unknown"
                properties["toxicity_alerts_count"] = 0
                properties["toxicity_alerts"] = ""
            
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating ADMET properties: {e}")
            return None
    
    def _is_valid_molecule(self, mol) -> bool:
        """Validate whether the molecule is valid."""
        try:
            if mol is None:
                return False
            
            # Check if the molecule can be sanitized
            mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol_clean is None:
                return False
            
            # Relaxed molecule size limits
            if mol_clean.GetNumAtoms() < 3 or mol_clean.GetNumAtoms() > 150:
                return False
            
            # Simplified valence check to avoid deprecation warnings
            try:
                # Try computing molecular properties; failure indicates invalid molecule
                Descriptors.MolWt(mol_clean)
                return True
            except:
                return False
            
        except Exception:
            return False
    
    def _calculate_simple_properties(self, smiles: str) -> Optional[Dict]:
        """Property calculation in simplified mode."""
        try:
            # Basic calculations
            properties = {
                "molecular_weight": self._estimate_molecular_weight(smiles),
                "logp": self._estimate_logp(smiles),
                "hbd": self._count_hydrogen_bond_donors(smiles),
                "hba": self._count_hydrogen_bond_acceptors(smiles),
                "rotatable_bonds": self._count_rotatable_bonds(smiles),
                "rings": self._count_rings(smiles),
                "aromatic_rings": self._count_aromatic_rings(smiles),
                "tpsa": self._estimate_tpsa(smiles),
                "lipinski_compliant": True,  # Simplified mode assumes compliant
                "canonical_smiles": smiles,
                "qed": np.nan,
            }

            # Solubility prediction (using estimated physicochemical properties)
            try:
                mw = properties["molecular_weight"]
                logp = properties["logp"]
                rb = properties["rotatable_bonds"]
                # Approximate aromatic fraction
                aromatic_atoms_approx = smiles.count('c')
                heavy_approx = sum(smiles.count(x) for x in ['C','N','O','S','F']) + smiles.count('Cl')
                ap = float(aromatic_atoms_approx) / float(heavy_approx) if heavy_approx > 0 else 0.0
                logS = 0.16 - 0.63 * float(logp) - 0.0062 * float(mw) + 0.066 * float(rb) + 0.74 * float(ap)
                properties["predicted_logS"] = float(logS)
                if logS > 0.5:
                    sol_class = "Highly Soluble"
                elif logS > 0.0:
                    sol_class = "Soluble"
                elif logS > -2.0:
                    sol_class = "Moderately Soluble"
                elif logS > -4.0:
                    sol_class = "Poorly Soluble"
                else:
                    sol_class = "Insoluble"
                properties["solubility_class"] = sol_class
            except Exception:
                properties["predicted_logS"] = 0.0
                properties["solubility_class"] = "Unknown"

            # Veber/Egan rules (based on estimated values)
            try:
                properties["veber_compliant"] = (properties["tpsa"] <= 140) and (properties["rotatable_bonds"] <= 10)
                properties["egan_compliant"] = (properties["logp"] <= 5.88) and (properties["tpsa"] <= 131)
            except Exception:
                properties["veber_compliant"] = False
                properties["egan_compliant"] = False

            # Toxicity structural alerts (string heuristics)
            try:
                alerts = []
                checks = {
                    "Nitro": "[N+](=O)[O-]",
                    "Azo": "N=N",
                    "Enone (Michael acceptor)": "C=CC(=O)",
                    "Alkyl halide_Cl": "CCl",
                    "Alkyl halide_Br": "CBr",
                    "Alkyl halide_I": "CI",
                    "Epoxide": "C1OC1",
                }
                for tag, key in checks.items():
                    if key in smiles:
                        alerts.append(tag)
                risk_level = "Low"
                if len(alerts) >= 3:
                    risk_level = "High"
                elif len(alerts) >= 1:
                    risk_level = "Medium"
                properties["toxicity_risk_level"] = risk_level
                properties["toxicity_alerts_count"] = len(alerts)
                properties["toxicity_alerts"] = ",".join(alerts)
            except Exception:
                properties["toxicity_risk_level"] = "Unknown"
                properties["toxicity_alerts_count"] = 0
                properties["toxicity_alerts"] = ""
            return properties
            
        except Exception as e:
            logger.error(f"Error in simplified property calculation: {e}")
            return None
    
    def _estimate_molecular_weight(self, smiles: str) -> float:
        """Estimate molecular weight."""
        atomic_masses = {'C': 12.01, 'H': 1.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'F': 19.00, 'Cl': 35.45}
        total_mass = sum(atomic_masses.get(atom, 0) * smiles.count(atom) for atom in atomic_masses)
        return total_mass
    
    def _estimate_logp(self, smiles: str) -> float:
        """Estimate LogP value."""
        logp = 0.0
        # Aromatic rings, alkyl groups increase hydrophobicity
        logp += smiles.count('c1ccccc1') * 1.8
        logp += smiles.count('C') * 0.02
        # Oxygen, nitrogen decrease hydrophobicity
        logp -= smiles.count('O') * 0.8
        logp -= smiles.count('N') * 0.5
        # Sulfur/halogens slightly increase
        logp += smiles.count('S') * 0.3
        logp += smiles.count('F') * 0.2
        logp += smiles.count('Cl') * 0.4
        logp += smiles.count('Br') * 0.6
        logp += smiles.count('I') * 0.8
        return float(logp)

    def _lipinski_violation_details(self, mol) -> List[str]:
        """Return violated Lipinski criteria."""
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
        """Count hydrogen bond donors."""
        return smiles.count('O') + smiles.count('N') + smiles.count('S')
    
    def _count_hydrogen_bond_acceptors(self, smiles: str) -> int:
        """Count hydrogen bond acceptors."""
        return smiles.count('O') + smiles.count('N') + smiles.count('S')
    
    def _count_rotatable_bonds(self, smiles: str) -> int:
        """Count rotatable bonds."""
        return smiles.count('C') // 4
    
    def _count_rings(self, smiles: str) -> int:
        """Count rings."""
        return smiles.count('1') // 2
    
    def _count_aromatic_rings(self, smiles: str) -> int:
        """Count aromatic rings."""
        return smiles.count('c1ccccc1')
    
    def _estimate_tpsa(self, smiles: str) -> float:
        """Estimate topological polar surface area."""
        tpsa = 0.0
        tpsa += smiles.count('O') * 20.0
        tpsa += smiles.count('N') * 17.0
        tpsa += smiles.count('S') * 38.0
        return tpsa
    
    def _check_lipinski_rules(self, mol) -> int:
        """Check Lipinski rule-of-five."""
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        return violations
    
    def batch_admet_analysis(self, ligands_data: List[Dict]) -> pd.DataFrame:
        """Batch ADMET analysis."""
        logger.info(f"Starting ADMET analysis for {len(ligands_data)} ligands...")
        
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
                logger.error(f"Error analyzing ligand {i+1}: {e}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            logger.info(f"ADMET analysis complete, successfully analyzed {len(results)} ligands")
            return df
        else:
            logger.warning("No successful ADMET analysis results")
            return pd.DataFrame()
    
    def save_admet_results(self, results_df: pd.DataFrame, report: Dict):
        """Save ADMET analysis results."""
        try:
            results_file = os.path.join(RESULTS_DIR, OUTPUT_FILES["admet_results"])
            results_df.to_csv(results_file, index=False)
            logger.info(f"ADMET results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Error saving ADMET results: {e}")

# Note: This module has no test entry point; it is used as a library by other workflows.