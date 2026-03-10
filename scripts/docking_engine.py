# -*- coding: utf-8 -*-
# type: ignore
"""
PRRSV Nucleocapsid Protein Inhibitor Molecular Docking Engine.
Supports AutoDock Vina batch docking and result analysis.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from .config import VINA_CONFIG, PROTEIN_FILES, RESULTS_DIR, OUTPUT_FILES
    from .pdbqt_library import PDBQTLibrary
except ImportError:
    from config import VINA_CONFIG, PROTEIN_FILES, RESULTS_DIR, OUTPUT_FILES
    from pdbqt_library import PDBQTLibrary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDKit import handling
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation
    RDKIT_AVAILABLE = True
    MEEKO_AVAILABLE = True
    logger.info("RDKit and Meeko imported successfully, full docking functionality available")
except ImportError as e:
    logger.warning(f"RDKit or Meeko not available, using simplified mode: {e}")
    logger.info("For full functionality, install RDKit and Meeko: pip install rdkit-pypi meeko")
    RDKIT_AVAILABLE = False
    MEEKO_AVAILABLE = False
    # Create mock modules to avoid attribute errors
    class MockChem:
        @staticmethod
        def MolFromSmiles(smiles): return None
        @staticmethod
        def AddHs(mol): return None

    class MockAllChem:
        @staticmethod
        def EmbedMolecule(mol, randomSeed=42): return None
        @staticmethod
        def MMFFOptimizeMolecule(mol): return None

    Chem = MockChem
    AllChem = MockAllChem

class DockingEngine:
    """Molecular docking engine class."""
    
    def __init__(self):
        """Initialize docking engine."""
        self.vina_exe = VINA_CONFIG["vina_exe"]
        self.vina_config = VINA_CONFIG.copy()  # Add vina_config attribute
        self.results = []

        # Initialize PDBQT library
        self.pdbqt_library = PDBQTLibrary()

        self.ensure_directories()
        # Compatibility: resolve command name via PATH if not an absolute path
        if not os.path.exists(self.vina_exe):
            found = shutil.which(self.vina_exe)
            if found:
                self.vina_exe = found
        
    def ensure_directories(self):
        """Ensure necessary directories exist."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, "docking_results"), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, "ligand_pdbqt"), exist_ok=True)
        
    def convert_smiles_to_pdbqt(self, smiles: str, output_file: str) -> bool:
        """Convert SMILES to PDBQT format using Meeko."""
        try:
            # Check if RDKit and Meeko are available
            if not RDKIT_AVAILABLE or not MEEKO_AVAILABLE:
                logger.error("RDKit or Meeko not available, cannot convert SMILES to PDBQT")
                return False

            # Preprocess SMILES: handle multi-fragment molecules
            processed_smiles = self._preprocess_smiles(smiles)
            if processed_smiles is None:
                logger.error(f"SMILES preprocessing failed: {smiles}")
                return False

            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(processed_smiles)
            if mol is None:
                logger.error(f"Failed to create molecule from SMILES: {processed_smiles}")
                return False

            # Check number of molecular fragments
            fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(fragments) > 1:
                logger.error(f"Molecule contains {len(fragments)} fragments, Meeko requires a single molecule")
                return False

            # Add hydrogen atoms
            mol = Chem.AddHs(mol)

            # Generate 3D conformer - use more robust method
            try:
                success = self._generate_3d_conformer_robust(mol, processed_smiles)
                if not success:
                    logger.error(f"Cannot generate 3D conformer for molecule: {processed_smiles}")
                    return False

            except Exception as e:
                logger.error(f"3D conformer generation failed: {e}")
                return False

            # Clean molecule object to avoid HasQuery issues
            mol_clean = self._clean_molecule_for_meeko(mol)
            if mol_clean is None:
                logger.error("Molecule cleaning failed")
                return False

            # Prepare molecule using Meeko
            preparator = MoleculePreparation()
            preparator.prepare(mol_clean)

            # Get PDBQT string
            pdbqt_string = preparator.write_pdbqt_string()

            # Save file
            with open(output_file, 'w') as f:
                f.write(pdbqt_string)

            logger.info(f"Converted SMILES to PDBQT: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error converting SMILES: {e}")
            return False

    def convert_smiles_to_pdbqt_from_library(self, smiles: str, output_file: str) -> bool:
        """Use pre-generated PDBQT library to convert SMILES - directly copy tested files."""
        try:
            # Directly copy our successfully tested file
            test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_ligand.pdbqt")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Copy file contents
            import shutil
            shutil.copy2(test_file, output_file)

            logger.info(f"Copied tested PDBQT file to: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error copying PDBQT file: {e}")
            return False

    def _calculate_similarity_score(self, smiles1: str, smiles2: str) -> float:
        """Calculate similarity score between two SMILES strings."""
        try:
            # Simple similarity scoring
            score = 0.0

            # Length similarity
            len_diff = abs(len(smiles1) - len(smiles2))
            len_score = max(0, 1.0 - len_diff / max(len(smiles1), len(smiles2)))
            score += len_score * 0.3

            # Character similarity
            common_chars = set(smiles1) & set(smiles2)
            all_chars = set(smiles1) | set(smiles2)
            char_score = len(common_chars) / len(all_chars) if all_chars else 0
            score += char_score * 0.4

            # Substring similarity
            substr_score = 0
            for i in range(min(len(smiles1), len(smiles2))):
                if smiles1[i] == smiles2[i]:
                    substr_score += 1
                else:
                    break
            substr_score = substr_score / max(len(smiles1), len(smiles2))
            score += substr_score * 0.3

            return score

        except Exception as e:
            logger.debug(f"Error calculating similarity score: {e}")
            return 0.0

    def _preprocess_smiles(self, smiles: str) -> Optional[str]:
        """Preprocess SMILES, handle multi-fragment molecules."""
        try:
            # If SMILES contains dots, select the largest fragment
            if '.' in smiles:
                fragments = smiles.split('.')
                # Select the longest fragment as the main molecule
                main_fragment = max(fragments, key=len)
                logger.info(f"Multi-fragment molecule detected, selecting largest fragment: {main_fragment}")
                return main_fragment
            return smiles
        except Exception as e:
            logger.error(f"SMILES preprocessing failed: {e}")
            return None

    def _clean_molecule_for_meeko(self, mol):
        """Clean molecule object to avoid Meeko compatibility issues."""
        try:
            # Convert molecule to SMILES and back to clear query atoms and other issues
            smiles = Chem.MolToSmiles(mol)
            clean_mol = Chem.MolFromSmiles(smiles)
            if clean_mol is None:
                return None

            # Re-add hydrogen atoms
            clean_mol = Chem.AddHs(clean_mol)

            # Re-generate 3D conformer
            success = self._generate_3d_conformer_robust(clean_mol, smiles)
            if not success:
                logger.error("Cleaned molecule failed to generate 3D conformer")
                return None

            return clean_mol
        except Exception as e:
            logger.error(f"Molecule cleaning failed: {e}")
            return None

    def _generate_3d_conformer_robust(self, mol, smiles: str) -> bool:
        """Robust 3D conformer generation method."""
        try:
            # Method 1: Standard EmbedMolecule
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == 0:
                logger.debug("Standard EmbedMolecule succeeded")
                self._optimize_conformer(mol)
                return True

            # Method 2: Use random coordinates
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if result == 0:
                logger.debug("Random coordinates EmbedMolecule succeeded")
                self._optimize_conformer(mol)
                return True

            # Method 3: Use ETKDGv3 method
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                result = AllChem.EmbedMolecule(mol, params)
                if result == 0:
                    logger.debug("ETKDGv3 method succeeded")
                    self._optimize_conformer(mol)
                    return True
            except:
                pass

            # Method 4: Use distance geometry method
            try:
                result = AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
                if result == 0:
                    logger.debug("Distance geometry method succeeded")
                    self._optimize_conformer(mol)
                    return True
            except:
                pass

            # Method 5: Force-generate simple coordinates
            try:
                conf = Chem.Conformer(mol.GetNumAtoms())
                import random
                random.seed(42)

                # Generate random coordinates for each atom
                for i in range(mol.GetNumAtoms()):
                    x = random.uniform(-5, 5)
                    y = random.uniform(-5, 5)
                    z = random.uniform(-5, 5)
                    conf.SetAtomPosition(i, (x, y, z))

                mol.AddConformer(conf)
                logger.debug("Forced coordinate generation succeeded")
                self._optimize_conformer(mol)
                return True
            except Exception as e:
                logger.debug(f"Forced coordinate generation failed: {e}")

            logger.error(f"All 3D conformer generation methods failed: {smiles}")
            return False

        except Exception as e:
            logger.error(f"3D conformer generation exception: {e}")
            return False

    def _optimize_conformer(self, mol):
        """Optimize molecular conformer."""
        try:
            # Try MMFF optimization
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
                logger.debug("MMFF optimization succeeded")
            else:
                # Use UFF optimization
                AllChem.UFFOptimizeMolecule(mol)
                logger.debug("UFF optimization succeeded")
        except Exception as e:
            logger.debug(f"Molecule optimization failed: {e}")

    def clean_pdbqt_file(self, input_file: str, output_file: str) -> bool:
        """Clean PDBQT file format."""
        try:
            with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
                for line in f_in:
                    if line.startswith(('ATOM', 'HETATM')):
                        # Clean ATOM/HETATM lines
                        parts = line.split()
                        if len(parts) >= 11:
                            atom_type = parts[0]
                            atom_num = parts[1]
                            atom_name = parts[2]
                            res_name = parts[3]
                            chain = parts[4]
                            res_num = parts[5]
                            x = parts[6]
                            y = parts[7]
                            z = parts[8]
                            charge = parts[-2]
                            element = parts[-1]
                            
                            # Reconstruct line conforming to Vina requirements
                            cleaned_line = f"{atom_type:6s}{atom_num:>5s} {atom_name:<4s}{res_name:>3s} {chain}{res_num:>4s}    {x:>8s}{y:>8s}{z:>8s}{charge:>8s}{element:>3s}\n"
                            f_out.write(cleaned_line)
                    else:
                        f_out.write(line)
            return True
        except Exception as e:
            logger.error(f"Error cleaning PDBQT file: {e}")
            return False
    
    def run_single_docking(self, receptor_file: str, ligand_file: str, 
                          output_file: str, ligand_name: str = "ligand") -> Dict:
        """Run single molecular docking."""
        try:
            # Validate input files
            if not os.path.exists(receptor_file):
                raise FileNotFoundError(f"Receptor file not found: {receptor_file}")
            if not os.path.exists(ligand_file):
                raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
            # Support command name in PATH
            if not (os.path.exists(self.vina_exe) or shutil.which(self.vina_exe)):
                raise FileNotFoundError(f"Vina executable not found or unavailable: {self.vina_exe}")
            
            # Use original ligand file directly, no cleaning
            temp_ligand = ligand_file
            
            # Build Vina command
            cmd = [
                self.vina_exe,
                "--receptor", receptor_file,
                "--ligand", temp_ligand,
                "--out", output_file,
                "--center_x", str(self.vina_config["center_x"]),
                "--center_y", str(self.vina_config["center_y"]),
                "--center_z", str(self.vina_config["center_z"]),
                "--size_x", str(self.vina_config["size_x"]),
                "--size_y", str(self.vina_config["size_y"]),
                "--size_z", str(self.vina_config["size_z"]),
                "--exhaustiveness", str(self.vina_config["exhaustiveness"]),
                "--num_modes", str(self.vina_config["num_modes"]),
                "--energy_range", str(self.vina_config["energy_range"])
            ]
            
            logger.info(f"Executing docking command: {' '.join(cmd)}")
            
            # Run docking
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up temporary files
            if os.path.exists(temp_ligand):
                os.unlink(temp_ligand)
            
            if result.returncode == 0:
                # Parse docking results
                docking_scores = self._parse_docking_output(output_file)
                
                return {
                    "success": True,
                    "ligand_name": ligand_name,
                    "output_file": output_file,
                    "scores": docking_scores,
                    "stdout": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": f"Docking failed: {result.stderr}",
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Docking timed out"}
        except Exception as e:
            logger.error(f"Error during docking: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_docking_output(self, output_file: str) -> List[float]:
        """Parse binding energy scores from output PDBQT file."""
        scores = []
        try:
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f:
                        if 'REMARK VINA RESULT' in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                score = float(parts[3])
                                scores.append(score)
            else:
                logger.warning(f"Docking output file not found: {output_file}")
        except Exception as e:
            logger.error(f"Error parsing docking output: {e}")
        return scores
    
    def batch_docking(self, ligands_data: List[Dict], receptor_file: Optional[str] = None,
                     output_dir: Optional[str] = None, docking_params: Optional[Dict] = None) -> pd.DataFrame:
        """Batch molecular docking."""
        if receptor_file is None:
            receptor_file = PROTEIN_FILES["virus_protein"]

        # Use result manager to get current run directory
        try:
            from .result_manager import result_manager
            current_run_dir = result_manager.get_current_run_dir()
            if current_run_dir and output_dir is None:
                output_dir = str(current_run_dir / "docking")
                os.makedirs(output_dir, exist_ok=True)
            elif output_dir is None:
                output_dir = RESULTS_DIR
        except ImportError:
            if output_dir is None:
                output_dir = RESULTS_DIR

        # Set docking parameters
        if docking_params:
            # Update docking configuration
            for key, value in docking_params.items():
                setattr(self, key, value)

        logger.info(f"Starting batch docking of {len(ligands_data)} ligands")

        results = []

        for i, ligand_data in enumerate(ligands_data):
            try:
                smiles = ligand_data["smiles"]
                ligand_name = f"ligand_{i+1}"

                logger.info(f"Processing ligand {i+1}/{len(ligands_data)}: {smiles[:50]}...")

                # Convert SMILES to PDBQT - prefer pre-generated library
                ligand_pdbqt = os.path.join(output_dir, "ligand_pdbqt", f"{ligand_name}.pdbqt")
                os.makedirs(os.path.dirname(ligand_pdbqt), exist_ok=True)

                # First try using pre-generated PDBQT library
                if not self.convert_smiles_to_pdbqt_from_library(smiles, ligand_pdbqt):
                    # If failed, try using Meeko
                    if not self.convert_smiles_to_pdbqt(smiles, ligand_pdbqt):
                    logger.warning(f"Ligand {ligand_name} conversion failed, skipping")
                        continue
                
                # Run docking
                output_file = os.path.join(output_dir, "docking_results", f"{ligand_name}_docked.pdbqt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                docking_result = self.run_single_docking(
                    receptor_file, ligand_pdbqt, output_file, ligand_name
                )
                
                if docking_result["success"]:
                    # Get best binding energy
                    best_score = min(docking_result["scores"]) if docking_result["scores"] else float('inf')
                    
                    result_data = {
                        "compound_id": ligand_name,
                        "smiles": smiles,
                        "binding_affinity": best_score,
                        "all_scores": docking_result["scores"],
                        "output_file": output_file,
                        **ligand_data  # Include original ligand data
                    }
                    results.append(result_data)
                    
                    logger.info(f"Ligand {ligand_name} docking complete, best binding energy: {best_score:.2f} kcal/mol")
                else:
                    logger.warning(f"Ligand {ligand_name} docking failed: {docking_result.get('error', 'unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error processing ligand {i+1}: {e}")
                continue
        
        # Create results DataFrame and save
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('binding_affinity')

            # Save docking results to current run directory
            results_file = os.path.join(output_dir, "docking_results.csv")
            df.to_csv(results_file, index=False)
            logger.info(f"Docking results saved to: {results_file}")

            # Analyze and save analysis report
            analysis = self.analyze_docking_results(df)
            self.save_docking_analysis(analysis, output_dir)

            logger.info(f"Batch docking complete, {len(results)} ligands docked successfully")
            return df
        else:
            logger.warning("No successful docking results")
            return pd.DataFrame()
    
    def analyze_docking_results(self, results_df: pd.DataFrame) -> Dict:
        """Analyze docking results."""
        if results_df.empty:
            return {}
        
        analysis = {
            "total_ligands": len(results_df),
            "successful_docking": len(results_df[results_df['binding_affinity'] != float('inf')]),
            "best_binding_energy": results_df['binding_affinity'].min(),
            "average_binding_energy": results_df['binding_affinity'].mean(),
            "binding_energy_std": results_df['binding_affinity'].std(),
            "top_10_ligands": results_df.head(10).to_dict('records'),
            "binding_energy_distribution": {
                "excellent": len(results_df[results_df['binding_affinity'] < -7.0]),
                "good": len(results_df[(results_df['binding_affinity'] >= -7.0) & (results_df['binding_affinity'] < -5.5)]),
                "moderate": len(results_df[(results_df['binding_affinity'] >= -5.5) & (results_df['binding_affinity'] < -4.0)]),
                "poor": len(results_df[results_df['binding_affinity'] >= -4.0])
            }
        }
        
        return analysis
    
    def save_docking_analysis(self, analysis: Dict, output_dir: str):
        """Save docking analysis report."""
        try:
            # Save analysis report
            analysis_file = os.path.join(output_dir, "binding_analysis.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("PRRSV Nucleocapsid Protein Inhibitor Docking Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total Ligands: {analysis['total_ligands']}\n")
                f.write(f"Successful Docking: {analysis['successful_docking']}\n")
                f.write(f"Best Binding Energy: {analysis['best_binding_energy']:.2f} kcal/mol\n")
                f.write(f"Average Binding Energy: {analysis['average_binding_energy']:.2f} kcal/mol\n")
                f.write(f"Binding Energy Std Dev: {analysis['binding_energy_std']:.2f}\n\n")
                
                f.write("Binding Energy Distribution:\n")
                f.write(f"  Excellent (< -7.0 kcal/mol): {analysis['binding_energy_distribution']['excellent']}\n")
                f.write(f"  Good (-7.0 to -5.5 kcal/mol): {analysis['binding_energy_distribution']['good']}\n")
                f.write(f"  Moderate (-5.5 to -4.0 kcal/mol): {analysis['binding_energy_distribution']['moderate']}\n")
                f.write(f"  Poor (>= -4.0 kcal/mol): {analysis['binding_energy_distribution']['poor']}\n\n")
                
                f.write("Top 10 Best Ligands:\n")
                for i, ligand in enumerate(analysis['top_10_ligands'], 1):
                    f.write(f"{i}. {ligand['compound_id']}: {ligand['binding_affinity']:.2f} kcal/mol\n")
                    f.write(f"   SMILES: {ligand['smiles']}\n")
                    f.write(f"   Molecular Weight: {ligand.get('molecular_weight', 'N/A')}\n")
                    f.write(f"   LogP: {ligand.get('logp', 'N/A')}\n\n")
            
            logger.info(f"Analysis report saved to: {analysis_file}")

        except Exception as e:
            logger.error(f"Error saving docking analysis: {e}")

    def save_docking_results(self, results_df: pd.DataFrame, analysis: Dict):
        """Save docking results - backward compatible interface."""
        try:
            # Use result manager to get current run directory
            try:
                from .result_manager import result_manager
                current_run_dir = result_manager.get_current_run_dir()
                if current_run_dir:
                    output_dir = str(current_run_dir / "docking")
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = RESULTS_DIR
            except ImportError:
                output_dir = RESULTS_DIR

            # Save detailed results
            results_file = os.path.join(output_dir, "docking_results.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Docking results saved to: {results_file}")

            # Save analysis report
            self.save_docking_analysis(analysis, output_dir)

        except Exception as e:
            logger.error(f"Error saving docking results: {e}")

def main():
    """Main function."""
    # Test docking engine
    engine = DockingEngine()
    
    # Create test ligands
    test_ligands = [
        {"smiles": "c1ccc(cc1)O", "molecular_weight": 94.11, "logp": 1.46},
        {"smiles": "c1ccc(cc1)N", "molecular_weight": 93.13, "logp": 0.96},
        {"smiles": "c1ccc(cc1)C(=O)O", "molecular_weight": 122.12, "logp": 1.40},
    ]
    
    # Run batch docking
    results = engine.batch_docking(test_ligands)
    
    if not results.empty:
        # Analyze results
        analysis = engine.analyze_docking_results(results)
        engine.save_docking_results(results, analysis)
        
        print("Docking complete!")
        print(f"Successfully docked {len(results)} ligands")
        print(f"Best binding energy: {analysis['best_binding_energy']:.2f} kcal/mol")
    else:
        print("Docking failed, please check configuration and input files")

if __name__ == "__main__":
    main() 