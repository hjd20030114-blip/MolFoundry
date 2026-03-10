# -*- coding: utf-8 -*-
"""
Configuration file for the PRRSV nucleocapsid protein inhibitor design project.
"""

import os

# Project root directory - points to the HJD directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, SCRIPTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Protein file paths
PROTEIN_FILES = {
    "virus_protein": os.path.join(DATA_DIR, "1p65.pdbqt"),  # PRRSV nucleocapsid protein (primary target)
    "integrin_complex_1": os.path.join(DATA_DIR, "AF-Q9GLP0-F1-model_v4_rigid.pdbqt"),  # Integrin complex 1
    "integrin_complex_2": os.path.join(DATA_DIR, "AF-F1SR53-F1-model_v4.pdbqt"),  # Integrin complex 2
    "old_virus_protein": os.path.join(DATA_DIR, "new1p65.pdbqt"),  # Legacy virus protein file (backup)
}

# Sequence file paths
SEQUENCE_FILES = {
    "capsid": os.path.join(DATA_DIR, "capsid.fasta"),
    "integrin": os.path.join(DATA_DIR, "integrin.fasta"),
}

# Auto-detect Vina executable path across platforms
_default_vina = os.path.join(DATA_DIR, "vina.exe")
_project_parent = os.path.dirname(PROJECT_ROOT)
_mac_vina = os.path.join(_project_parent, "autodock_vina", "bin", "vina")
_vina_path = _default_vina
if os.path.exists(_mac_vina) and os.access(_mac_vina, os.X_OK):
    _vina_path = _mac_vina
elif os.path.exists(_default_vina) and os.access(_default_vina, os.X_OK):
    _vina_path = _default_vina
else:
    # Fall back to system PATH
    _vina_path = "vina"

# AutoDock Vina configuration
VINA_CONFIG = {
    # Binding site coordinates (based on 1p65.pdbqt structure)
    "center_x": -0.24,
    "center_y": 26.06,
    "center_z": 65.48,

    # Search box dimensions (Angstrom)
    "size_x": 55.85,
    "size_y": 49.77,
    "size_z": 70.43,

    # Docking parameters - optimized for better binding affinity
    "exhaustiveness": 16,  # Increased search exhaustiveness (8 -> 16)
    "num_modes": 20,       # Increased number of conformations (9 -> 20)
    "energy_range": 4,     # Increased energy range for more conformations

    # Vina executable path (auto-detected)
    "vina_exe": _vina_path,
}

# Ligand generation parameters
LIGAND_GENERATION = {
    "num_ligands": 100,  # Number of ligands to generate (default, can be overridden)
    "max_atoms": 50,    # Maximum number of atoms
    "min_atoms": 10,    # Minimum number of atoms
    "target_logp": 2.5, # Target LogP value
    "target_mw": 400,   # Target molecular weight
}

# ADMET screening criteria - optimized for antiviral drugs
ADMET_CRITERIA = {
    "logp_range": (1, 4.5),         # LogP range (antivirals typically need moderate lipophilicity)
    "molecular_weight": (200, 600), # MW range (expanded to include more active molecules)
    "hbd_max": 4,                   # Max H-bond donors (reduced for membrane permeability)
    "hba_max": 8,                   # Max H-bond acceptors (balanced for solubility/permeability)
    "rotatable_bonds_max": 8,       # Max rotatable bonds (slightly increased for binding flexibility)
    "tpsa_range": (30, 120),        # TPSA range (optimized for oral bioavailability)
    "aromatic_rings_max": 4,        # Max aromatic rings (controls molecular complexity)
    "heavy_atoms_range": (15, 45),  # Heavy atom count range (ensures moderate molecular size)
}

# Output file naming
OUTPUT_FILES = {
    "docking_results": "docking_results.csv",
    "admet_results": "admet_analysis.csv",
    "top_ligands": "top_ligands.csv",
    "binding_analysis": "binding_analysis.txt",
    "ligand_images": "ligand_images/",
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(RESULTS_DIR, "project.log"),
}

# Database configuration (optional, for storing results)
DATABASE_CONFIG = {
    "enabled": False,
    "type": "sqlite",
    "path": os.path.join(RESULTS_DIR, "prrsv_inhibitors.db"),
}