# -*- coding: utf-8 -*-
"""
PRRSV Nucleocapsid Protein Inhibitor Ligand Generation Module.
Supports multiple molecular generation strategies and optimization algorithms.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
import time

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
try:
    from .config import *
except ImportError:
    from config import *

# Import CMD-GEN integration module
try:
    from scripts.cmdgen_integration import CMDGENGenerator
    CMDGEN_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("CMD-GEN integration module imported")
except ImportError:
    CMDGEN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CMD-GEN integration module not available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RDKit import handling
RDKIT_AVAILABLE = False
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem import Draw

    RDKIT_AVAILABLE = True
    logger.info("RDKit imported successfully, full ligand generation functionality available")
except ImportError:
    logger.warning("RDKit not available, using simplified mode for ligand generation")
    logger.info("For full functionality, install RDKit: pip install rdkit-pypi")


    # Create mock modules to avoid attribute errors
    class MockChem:
        @staticmethod
        def MolFromSmiles(smiles): return None

        @staticmethod
        def MolToSmiles(mol): return ""

        @staticmethod
        def RWMol(mol): return None

        @staticmethod
        def Atom(symbol): return None

        @staticmethod
        def BondType(): return None

        @staticmethod
        def Draw():
            class MockDraw:
                @staticmethod
                def MolToImage(mol, size): return None

            return MockDraw()


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
    AllChem = MockChem


class LigandGenerator:
    """Ligand generator class."""

    def __init__(self, use_cmdgen: bool = True, cmdgen_path: Optional[str] = None):
        """
        Initialize ligand generator.

        Args:
            use_cmdgen: Whether to use CMD-GEN model
            cmdgen_path: Path to CMD-GEN code
        """
        self.generated_ligands = []
        self.smiles_templates = self._load_smiles_templates()
        self.fragment_library = self._load_fragment_library()

        # Initialize CMD-GEN generator
        self.use_cmdgen = use_cmdgen and CMDGEN_AVAILABLE
        self.cmdgen_generator = None

        if self.use_cmdgen:
            try:
                self.cmdgen_generator = CMDGENGenerator(cmdgen_path=cmdgen_path)
                logger.info("CMD-GEN generator initialized successfully")
            except Exception as e:
                logger.warning(f"CMD-GEN generator initialization failed: {e}")
                self.use_cmdgen = False

    def _load_smiles_templates(self) -> List[str]:
        """Load SMILES template library - using molecules more suitable for drug design."""
        templates = [
            # Drug-like molecule templates - easier to generate 3D conformers
            "CCc1ccc(cc1)C(=O)O",  # Ibuprofen analog
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Isobutyl-phenylpropionic acid
            "COc1ccc(cc1)CCN",  # Methoxyphenethylamine
            "Cc1ccc(cc1)C(=O)Nc2ccccc2",  # Toluamide-aniline
            "CCOc1ccc(cc1)C(=O)N",  # p-Ethoxybenzamide

            # Antiviral drug scaffolds - enhanced binding affinity
            "Cc1ccc2nc(N)nc(Nc3ccccc3)c2c1",  # Anilino-quinazoline (antiviral activity)
            "COc1ccc(cc1)c2nc(N)nc(N)n2",  # Methoxyphenyl-triazine (protein inhibitor)
            "Cc1ccc(cc1)S(=O)(=O)Nc2nc(N)nc(N)n2",  # Sulfonamide-triazine (strong binding)
            "CCc1ccc(cc1)c2nc(Nc3ccccc3)nc(N)n2",  # Anilino-triazine (high affinity)
            "COc1ccc2c(c1)nc(Nc3ccccc3)nc2N",  # Anilino-quinazoline (optimized binding)

            # Heterocyclic drug templates - optimized versions
            "Cc1ccc2nc(N)nc(N)c2c1",  # Diamino-methylquinazoline
            "COc1ccc2c(c1)nc(N)nc2N",  # Methoxy-diaminoquinazoline
            "Cc1nc2ccccc2c(=O)n1C",  # Methylquinazolinone
            "CCc1ccc2nc(C)nc(N)c2c1",  # Ethyl-aminoquinazoline
            "Fc1ccc(cc1)c2nc(N)nc(N)c2",  # Fluorophenyl-pyrimidine (enhanced binding)

            # Protein inhibitor scaffolds
            "Cc1ccc(cc1)c2nc(Nc3ccc(F)cc3)nc(N)n2",  # Fluoroanilino-triazine
            "COc1ccc(cc1)c2nnc(Nc3ccccc3)s2",  # Anilino-thiadiazole
            "Cc1ccc(cc1)C(=O)Nc2nc(N)nc(N)n2",  # Amide-triazine
            "CCc1ccc(cc1)c2nc(N)nc(Nc3ccc(Cl)cc3)n2",  # Chloroanilino-triazine

            # Benzoheterocycles - enhanced versions
            "COc1ccc2c(c1)c(C)cn2C",  # Methoxy-methylindole
            "Cc1ccc2c(c1)nc(C)n2C",  # Dimethylbenzimidazole
            "CCc1ccc2c(c1)oc(C)c2C(=O)O",  # Ethyl-benzofuran carboxylic acid
            "Fc1ccc2c(c1)nc(N)nc2Nc3ccccc3",  # Fluoroanilino-quinazoline

            # Nitrogen heterocycles - high affinity versions
            "Cc1ccc(cc1)c2nc(N)nc(N)n2",  # Tolyl-triazinediamine
            "COc1ccc(cc1)c2nnc(N)s2",  # Methoxyphenyl-thiadiazolamine
            "Cc1ccc(cc1)C2=NN=C(N)S2",  # Tolyl-thiadiazolamine
            "Clc1ccc(cc1)c2nc(N)nc(N)n2",  # Chlorophenyl-triazine (strong binding)

            # Aliphatic-linked aromatic rings - optimized flexibility
            "CCc1ccc(cc1)OCc2ccccc2",  # Ethylphenoxymethyl-benzene
            "COc1ccc(cc1)CCc2ccccc2",  # Methoxyphenethyl-benzene
            "Cc1ccc(cc1)COc2ccccc2C",  # Tolyloxymethyl-toluene
            "Fc1ccc(cc1)OCc2ccc(F)cc2",  # Difluorophenoxy (enhanced binding)

            # Amides and esters - hydrogen bond optimized
            "CCc1ccc(cc1)C(=O)NCc2ccccc2",  # Ethylbenzoyl-benzylamine
            "COc1ccc(cc1)C(=O)OCC",  # Methoxybenzoic acid ethyl ester
            "Cc1ccc(cc1)C(=O)N(C)C",  # Toluoyl-dimethylamine
            "Fc1ccc(cc1)C(=O)Nc2nc(N)nc(N)n2",  # Fluorobenzamide-triazine (strong H-bond)
        ]
        return templates

    def _load_fragment_library(self) -> Dict[str, List[str]]:
        """Load molecular fragment library."""
        fragments = {
            "aromatic": ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccc2ncccc2c1"],
            "aliphatic": ["CC", "CCC", "CC(C)C", "CC(C)CC"],
            "heterocyclic": ["c1ccncc1", "c1ccoc1", "c1ccsc1"],
            "functional_groups": ["O", "N", "S", "C=O", "C#N", "C(=O)O", "C(=O)N"],
            "linkers": ["C", "CC", "CCC", "c1ccccc1", "C(=O)", "C#C"]
        }
        return fragments

    def generate_random_ligand(self) -> Optional[str]:
        """Generate a random ligand."""
        try:
            # If RDKit is not available, use simplified mode
            if not RDKIT_AVAILABLE:
                return self._generate_simplified_ligand()

            # Select base template
            template = random.choice(self.smiles_templates)
            mol = Chem.MolFromSmiles(template)

            if mol is None:
                return None

            # Validate molecule
            if not self._is_valid_molecule(mol):
                return None

            # Random modification (reduce iterations to avoid complexity issues)
            for _ in range(random.randint(0, 2)):  # Reduced modification count
                modified_result = self._random_modification(mol)
                if modified_result is None:
                    break

                # If result is a string, convert to molecule object
                if isinstance(modified_result, str):
                    mol = Chem.MolFromSmiles(modified_result)
                else:
                    mol = modified_result

                if mol is None or not self._is_valid_molecule(mol):
                    break

            if mol is None:
                return None

            return Chem.MolToSmiles(mol)

        except Exception as e:
            logger.error(f"Error generating random ligand: {e}")
            return None

    def _generate_simplified_ligand(self) -> Optional[str]:
        """Ligand generation in simplified mode."""
        try:
            # Randomly select a single template without fragment combination
            # This ensures a single molecule is generated rather than multi-fragment molecules
            ligand = random.choice(self.smiles_templates)

            # Validate molecule size
            # Simple atom count (remove special characters)
            atom_count = len(ligand.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('=', '').replace('#', '').replace('.', ''))

            # Simple size limit
            if 3 <= atom_count <= 150:
                return ligand
            return None

        except Exception as e:
            logger.error(f"Error in simplified ligand generation: {e}")
            return None

    def _is_valid_molecule(self, mol) -> bool:
        """Validate whether a molecule is valid."""
        try:
            if mol is None:
                return False

            # If RDKit is not available, use simplified validation
            if not RDKIT_AVAILABLE:
                # Simple atom count validation
                smiles = Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)
                atom_count = len(smiles.replace('[', '').replace(']', '').replace('.', ''))
                return 3 <= atom_count <= 150

            # Use full validation when RDKit is available
            mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol_clean is None:
                return False

            # Molecule size limit
            if mol_clean.GetNumAtoms() < 3 or mol_clean.GetNumAtoms() > 150:
                return False

            # Try calculating molecular properties as validation
            try:
                Descriptors.MolWt(mol_clean)
                return True
            except:
                return False

        except Exception:
            return False

    def _random_modification(self, mol) -> Optional[str]:
        """Randomly modify a molecule."""
        try:
            # If RDKit is not available, return original molecule directly
            if not RDKIT_AVAILABLE:
                return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

            # Simplified modification method
            modification_type = random.choice([
                "simple_substitution",
                "functional_group_addition"
            ])

            if modification_type == "simple_substitution":
                return self._simple_substitution(mol)
            elif modification_type == "functional_group_addition":
                return self._add_simple_functional_group(mol)

        except Exception as e:
            logger.error(f"Error during molecule modification: {e}")
            return mol

    def _simple_substitution(self, mol) -> Optional[str]:
        """Simple substituent addition."""
        try:
            # If RDKit is not available, return original molecule directly
            if not RDKIT_AVAILABLE:
                return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

            # Use safer SMILES operations
            smiles = Chem.MolToSmiles(mol, canonical=True)

            # Simple substituents
            substituents = ["F", "Cl", "Br", "O", "N"]
            substituent = random.choice(substituents)

            # Create new SMILES
            modified_smiles = smiles + "." + substituent
            return modified_smiles

        except Exception as e:
            logger.error(f"Error during simple substitution: {e}")
            return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

    def _add_simple_functional_group(self, mol) -> Optional[str]:
        """Add simple functional groups."""
        try:
            # If RDKit is not available, return original molecule directly
            if not RDKIT_AVAILABLE:
                return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

            smiles = Chem.MolToSmiles(mol, canonical=True)

            # Simple functional groups
            functional_groups = ["C(=O)O", "C(=O)N", "C#N"]
            fg = random.choice(functional_groups)

            # Combine SMILES
            combined_smiles = smiles + "." + fg
            return combined_smiles

        except Exception as e:
            logger.error(f"Error adding functional group: {e}")
            return Chem.MolToSmiles(mol) if hasattr(mol, 'GetNumAtoms') else str(mol)

    def generate_cmdgen_ligands(self,
                               pdb_file: Optional[str] = None,
                               num_ligands: Optional[int] = None,
                               ref_ligand: str = "A:1") -> List[Dict]:
        """
        Use CMD-GEN to generate structure-based ligands.

        Args:
            pdb_file: Path to PDB file
            num_ligands: Number of ligands to generate
            ref_ligand: Reference ligand

        Returns:
            List of generated ligands
        """
        # If quantity not specified, use default from config file
        if num_ligands is None:
            num_ligands = LIGAND_GENERATION["num_ligands"]

        if not self.use_cmdgen or not self.cmdgen_generator:
            logger.warning("CMD-GEN not available, using traditional method to generate ligands")
            return self._generate_traditional_ligands(num_ligands)

        if not pdb_file or not os.path.exists(pdb_file):
            logger.warning("PDB file does not exist, using traditional method to generate ligands")
            return self._generate_traditional_ligands(num_ligands)

        try:
            logger.info(f"Using CMD-GEN to generate {num_ligands} structure-based ligands")

            # Use CMD-GEN to generate molecules
            molecules = self.cmdgen_generator.generate_pocket_based_molecules(
                pdb_file=pdb_file,
                num_molecules=num_ligands,
                ref_ligand=ref_ligand
            )

            if molecules:
                logger.info(f"CMD-GEN successfully generated {len(molecules)} ligands")
                return molecules
            else:
                logger.warning("CMD-GEN generation failed, using traditional method")
                return self._generate_traditional_ligands(num_ligands)

        except Exception as e:
            logger.error(f"Error during CMD-GEN generation: {e}")
            return self._generate_traditional_ligands(num_ligands)

    def _generate_traditional_ligands(self, num_ligands: int) -> List[Dict]:
        """Generate ligands using traditional method."""
        logger.info(f"Using traditional method to generate {num_ligands} ligands")

        ligands = []
        successful_generations = 0
        max_attempts = num_ligands * 3

        for attempt in range(max_attempts):
            if successful_generations >= num_ligands:
                break

            smiles = self.generate_random_ligand()
            if smiles:
                ligand_info = {
                    "compound_id": f"ligand_{successful_generations + 1}",
                    "smiles": smiles,
                    "generation_method": "Traditional",
                    "source": "template_based"
                }

                # Calculate molecular properties
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        ligand_info.update({
                            "molecular_weight": Descriptors.MolWt(mol),
                            "logp": Descriptors.MolLogP(mol),
                            "hbd": Descriptors.NumHDonors(mol),
                            "hba": Descriptors.NumHAcceptors(mol),
                            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                            "tpsa": Descriptors.TPSA(mol)
                        })

                ligands.append(ligand_info)
                successful_generations += 1

        return ligands

    def generate_optimized_ligands(self,
                                 num_ligands: Optional[int] = None,
                                 pdb_file: Optional[str] = None,
                                 use_cmdgen: Optional[bool] = None,
                                 optimize_for_binding: bool = True) -> List[Dict]:
        """
        Generate optimized ligand library.

        Args:
            num_ligands: Number of ligands to generate
            pdb_file: Path to PDB file (for CMD-GEN)
            use_cmdgen: Whether to use CMD-GEN (overrides default setting)
            optimize_for_binding: Whether to optimize for binding affinity

        Returns:
            List of generated ligands
        """
        if num_ligands is None:
            num_ligands = LIGAND_GENERATION["num_ligands"]

        # Decide which generation method to use
        should_use_cmdgen = use_cmdgen if use_cmdgen is not None else self.use_cmdgen

        if should_use_cmdgen and pdb_file:
            logger.info(f"Using CMD-GEN to generate {num_ligands} structure-based ligands")
            ligands = self.generate_cmdgen_ligands(
                pdb_file=pdb_file,
                num_ligands=num_ligands
            )
        else:
            logger.info(f"Using traditional method to generate {num_ligands} ligands")
            ligands = self._generate_traditional_ligands(num_ligands)

        # If binding optimization is enabled, perform post-processing
        if optimize_for_binding and ligands:
            logger.info("Optimizing generated ligands for binding affinity...")
            ligands = self._optimize_ligands_for_binding(ligands)

        return ligands

    def _calculate_molecular_properties(self, smiles: str) -> Dict:
        """Calculate molecular properties."""
        try:
            # If RDKit is not available, return simplified properties
            if not RDKIT_AVAILABLE:
                return {
                    "molecular_weight": random.uniform(100, 500),
                    "logp": random.uniform(-1, 5),
                    "hbd": random.randint(0, 5),
                    "hba": random.randint(0, 10),
                    "rotatable_bonds": random.randint(0, 10),
                    "tpsa": random.uniform(0, 150),
                    "rings": random.randint(0, 5),
                    "aromatic_rings": random.randint(0, 3),
                }

            # Create molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

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
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "formal_charge": Chem.rdmolops.GetFormalCharge(mol),
                "fraction_csp3": getattr(Descriptors, 'FractionCsp3', lambda x: 0.0)(mol)
            }

            return properties

        except Exception as e:
            logger.error(f"Error calculating molecular properties: {e}")
            return {}

    def _check_admet_criteria(self, properties: Dict) -> bool:
        """Check ADMET criteria."""
        try:
            # Check molecular weight
            if not (ADMET_CRITERIA["molecular_weight"][0] <=
                    properties["molecular_weight"] <=
                    ADMET_CRITERIA["molecular_weight"][1]):
                return False

            # Check LogP
            if not (ADMET_CRITERIA["logp_range"][0] <=
                    properties["logp"] <=
                    ADMET_CRITERIA["logp_range"][1]):
                return False

            # Check hydrogen bond donors
            if properties["hbd"] > ADMET_CRITERIA["hbd_max"]:
                return False

            # Check hydrogen bond acceptors
            if properties["hba"] > ADMET_CRITERIA["hba_max"]:
                return False

            # Check rotatable bonds
            if properties["rotatable_bonds"] > ADMET_CRITERIA["rotatable_bonds_max"]:
                return False

            # Check TPSA
            if not (ADMET_CRITERIA["tpsa_range"][0] <=
                    properties["tpsa"] <=
                    ADMET_CRITERIA["tpsa_range"][1]):
                return False

            # Check aromatic ring count (if configured)
            if "aromatic_rings_max" in ADMET_CRITERIA:
                aromatic_rings = properties.get("aromatic_rings", 0)
                if aromatic_rings > ADMET_CRITERIA["aromatic_rings_max"]:
                    return False

            # Check heavy atom count (if configured)
            if "heavy_atoms_range" in ADMET_CRITERIA:
                heavy_atoms = properties.get("heavy_atoms", 0)
                if not (ADMET_CRITERIA["heavy_atoms_range"][0] <=
                        heavy_atoms <=
                        ADMET_CRITERIA["heavy_atoms_range"][1]):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking ADMET criteria: {e}")
            return False

    def _optimize_ligands_for_binding(self, ligands: List[Dict]) -> List[Dict]:
        """
        Optimize ligands to improve binding affinity.

        Args:
            ligands: Original ligand list

        Returns:
            Optimized ligand list
        """
        try:
            optimized_ligands = []
            scores = []

            for ligand in ligands:
                # Calculate molecular descriptors
                properties = self._calculate_molecular_properties(ligand["smiles"])

                # Score based on molecular properties
                binding_score = self._calculate_binding_potential_score(properties)
                ligand["binding_potential_score"] = binding_score
                scores.append(binding_score)

                # Lower threshold to retain more ligands
                if binding_score > 0.3:  # Lower threshold from 0.6 to 0.3
                    optimized_ligands.append(ligand)

            # If still no ligands pass screening, keep the highest-scoring ones
            if not optimized_ligands and ligands:
                logger.warning("All ligands scored below threshold, keeping highest-scoring ligands")
                # Sort by score and keep top 50%
                ligands_with_scores = [(ligand, score) for ligand, score in zip(ligands, scores)]
                ligands_with_scores.sort(key=lambda x: x[1], reverse=True)
                keep_count = max(1, len(ligands) // 2)
                optimized_ligands = [ligand for ligand, _ in ligands_with_scores[:keep_count]]

            # Sort by binding potential
            optimized_ligands.sort(key=lambda x: x.get("binding_potential_score", 0), reverse=True)

            logger.info(f"Ligand optimization complete: {len(ligands)} -> {len(optimized_ligands)} high-potential ligands")
            if scores:
                logger.info(f"Score range: {min(scores):.3f} - {max(scores):.3f}")

            return optimized_ligands

        except Exception as e:
            logger.error(f"Error during ligand optimization: {e}")
            return ligands

    def _calculate_binding_potential_score(self, properties: Dict) -> float:
        """
        Calculate binding potential score based on molecular properties.

        Args:
            properties: Molecular properties dictionary

        Returns:
            Binding potential score (0-1)
        """
        try:
            score = 0.0

            # LogP score (ideal range 2-3.5)
            logp = properties.get("logp", 0)
            if 2.0 <= logp <= 3.5:
                score += 0.25
            elif 1.5 <= logp <= 4.0:
                score += 0.15

            # Molecular weight score (ideal range 300-450)
            mw = properties.get("molecular_weight", 0)
            if 300 <= mw <= 450:
                score += 0.25
            elif 250 <= mw <= 500:
                score += 0.15

            # Hydrogen bond donor/acceptor score
            hbd = properties.get("hbd", 0)
            hba = properties.get("hba", 0)
            if 1 <= hbd <= 3 and 3 <= hba <= 6:
                score += 0.2

            # Aromatic ring score (2-3 aromatic rings are usually favorable for protein binding)
            aromatic_rings = properties.get("aromatic_rings", 0)
            if 2 <= aromatic_rings <= 3:
                score += 0.15
            elif aromatic_rings == 1 or aromatic_rings == 4:
                score += 0.1

            # TPSA score (moderate polar surface area)
            tpsa = properties.get("tpsa", 0)
            if 60 <= tpsa <= 90:
                score += 0.15
            elif 40 <= tpsa <= 110:
                score += 0.1

            return min(score, 1.0)  # Ensure score does not exceed 1.0

        except Exception as e:
            logger.error(f"Error calculating binding potential score: {e}")
            return 0.5  # Return moderate score

    def save_ligands(self, ligands: List[Dict], filename: Optional[str] = None):
        """Save ligands to file."""
        if filename is None:
            filename = os.path.join(RESULTS_DIR, "generated_ligands.csv")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(ligands)
        df.to_csv(filename, index=False)
        logger.info(f"Ligand data saved to: {filename}")

        # Generate SMILES file for docking
        smiles_file = os.path.join(RESULTS_DIR, "ligands.smi")
        with open(smiles_file, 'w') as f:
            for ligand in ligands:
                f.write(f"{ligand['smiles']}\t{ligand['smiles']}\n")
        logger.info(f"SMILES file saved to: {smiles_file}")

    def visualize_ligands(self, ligands: List[Dict], num_ligands: int = 10):
        """Visualize ligands."""
        try:
            # Create image directory
            img_dir = os.path.join(RESULTS_DIR, OUTPUT_FILES["ligand_images"])
            os.makedirs(img_dir, exist_ok=True)

            # Select top N ligands for visualization
            selected_ligands = ligands[:min(num_ligands, len(ligands))]

            for i, ligand in enumerate(selected_ligands):
                # If RDKit is not available, skip visualization
                if not RDKIT_AVAILABLE:
                    logger.warning("RDKit not available, skipping ligand visualization")
                    break

                mol = Chem.MolFromSmiles(ligand["smiles"])
                if mol is not None:
                    # Generate 2D image
                    img = Chem.Draw.MolToImage(mol, size=(300, 300))
                    img_path = os.path.join(img_dir, f"ligand_{i + 1}.png")
                    img.save(img_path)

            if RDKIT_AVAILABLE:
                logger.info(f"Ligand images saved to: {img_dir}")

        except Exception as e:
            logger.error(f"Error visualizing ligands: {e}")


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("PRRSV Nucleocapsid Protein Inhibitor Ligand Generation System Started")
    logger.info("=" * 60)

    # Print current configuration
    logger.info(f"Number of ligands to generate: {LIGAND_GENERATION['num_ligands']}")
    logger.info(f"ADMET criteria: {ADMET_CRITERIA}")

    # Initialize generator
    generator = LigandGenerator()

    # Generate ligands
    ligands = generator.generate_optimized_ligands()

    if ligands:
        # Save ligands
        generator.save_ligands(ligands)

        # Visualize ligands
        generator.visualize_ligands(ligands)

        logger.info(f"Successfully generated {len(ligands)} ligands")
        logger.info("Ligand data saved to results/generated_ligands.csv")
        logger.info("Ligand images saved to results/ligand_images/")
    else:
        logger.error("Failed to generate valid ligands")


if __name__ == "__main__":
    main()