# MolFoundry — Structure-Aware Deep Learning Platform for PRRSV Inhibitor Design

## Overview

MolFoundry is an end-to-end drug design platform that integrates classical computational chemistry methods with modern deep learning techniques. It is specifically designed for the intelligent design and optimization of small-molecule inhibitors targeting the Porcine Reproductive and Respiratory Syndrome Virus (PRRSV) nucleocapsid (N) protein.

**Paper:** *MolFoundry: A Structure-Aware Generative Framework for De Novo Design of PRRSV Nucleocapsid Inhibitors*
**Authors:** Jindong Hao, Weibo Jin — College of Life Sciences and Medicine, Zhejiang Sci-tech University
**License:** MIT

## Key Features

### 🧬 Molecular Generation
- **CMD-GEN Integration**: Intelligent molecular generation based on DiffPhar and GCPG
- **Deep Learning Generation**: Molecular design driven by SE(3)-Equivariant GNNs
- **Pocket-Conditioned Generation**: Target-directed molecular generation based on protein binding sites

### 🎯 Molecular Docking
- **AutoDock Vina Integration**: High-precision molecular docking calculations
- **Batch Docking**: Support for simultaneous multi-ligand docking analysis
- **Binding Site Analysis**: Automatic identification and analysis of protein binding sites

### 💊 ADMET Analysis
- **Drug-likeness Assessment**: Lipinski's Rule of Five and other drug-likeness metrics
- **Toxicity Prediction**: Machine learning-based toxicity risk assessment
- **Pharmacokinetics**: Prediction of absorption, distribution, metabolism, and excretion properties

### 🤖 Deep Learning Modules
- **SE(3)-Equivariant GNN**: Equivariant graph neural network for binding affinity prediction
- **Pocket–Ligand Transformer**: Cross-attention mechanism for pocket-ligand interaction modeling
- **Multi-task Learning**: Multi-task discriminator framework (ADMET, synthetic accessibility, etc.)

### 📊 Visualization & Analysis
- **3D Molecular Visualization**: Interactive molecular structure display (py3Dmol)
- **Docking Result Visualization**: Protein–ligand complex rendering
- **Data Analysis Dashboards**: Real-time interactive charts (Plotly)

## Project Structure

```
HJD/
├── README.md
├── requirements.txt
├── start_project.py                 # Entry point (dependency check + UI launch)
├── unified_web_interface.py         # Unified Streamlit web interface
├── run_full_workflow.py             # One-click full pipeline (generation → docking → ADMET → 3D report)
├── deep_learning_pipeline.py        # Deep learning end-to-end pipeline (research version)
│
├── scripts/                         # Computational chemistry / engineering pipeline
│   ├── config.py                    # Paths, Vina parameters, ADMET filtering thresholds
│   ├── ligand_generator.py          # Ligand generation (template/fragment + CMD-GEN optional)
│   ├── molecular_docking.py         # AutoDock Vina docking (with robust fallback)
│   ├── admet_analyzer.py            # RDKit ADMET & rule-based evaluation (with simplified fallback)
│   ├── binding_site_analyzer.py     # Binding site parsing / geometry / surface analysis
│   ├── visualization_3d.py          # 3D molecule / complex / dashboard / comprehensive report
│   ├── result_manager.py            # Run-level result directory & metadata management
│   ├── cmdgen_integration.py        # External CMD-GEN integration (DiffPhar/GCPG)
│   ├── view_all_3d_results.py       # Batch 3D result viewer
│   └── streamlit_3d_viewer.py       # Standalone 3D viewer
│
├── deep_learning/                   # Deep learning research modules
│   ├── models/
│   │   ├── equivariant_gnn.py       # SE(3)-Equivariant GNN affinity scorer
│   │   ├── transformer.py           # Pocket–Ligand Cross-attention Transformer
│   │   ├── discriminator.py         # Multi-task discriminator (ADMET / synthetic difficulty, etc.)
│   └── data/featurizers.py          # Molecular / protein / interaction featurization
│
├── data/                            # Sample data & resources
│   ├── 1p65.pdb / 1p65.pdbqt / 1p65.cif
│   ├── AF-Q9GLP0-F1-model_v4.pdb(.pdbqt)  # Integrin-related structures
│   ├── AF-F1SR53-F1-model_v4.pdb(.pdbqt)
│   ├── capsid.fasta / integrin.fasta       # Sequence data
│   ├── ligands.sdf                         # Seed / example ligands
│   └── P-L/                                # Protein–ligand pairs (training/evaluation examples)
│
└── results/                         # Run outputs (auto-organized by date)
    └── run_YYYYMMDD_xxx/
        ├── ligands/                 # Generated ligands & .smi files
        ├── docking/                 # docking_results.csv
        ├── admet/                   # admet_results.csv
        ├── visualization_3d/        # HTML reports, dashboards, galleries
        └── reports/                 # Additional reports
```

## Quick Start

### 1. Requirements
- Python 3.9+
- PyTorch 2.0+
- RDKit
- Streamlit
- Additional dependencies listed in `requirements.txt`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the System
```bash
python start_project.py
```

### 4. Access the Web Interface
Open your browser and navigate to: http://localhost:8501

## Usage Guide

### Web Interface
1. **Home**: Platform overview and feature introduction
2. **Molecular Generation**: Generate molecules using CMD-GEN or deep learning models
3. **Molecular Docking**: Upload protein structures and ligands for docking analysis
4. **ADMET Analysis**: Evaluate drug-likeness and safety profiles of molecules
5. **Data Visualization**: Explore 3D molecular structures and analytical results
6. **Experiment Reports**: Generate comprehensive analysis reports

### Command-Line Interface
```python
# Molecular docking example
from scripts.molecular_docking import MolecularDocking

docker = MolecularDocking()
results = docker.dock_multiple_ligands("HJD/data/1p65.pdb", ["CCO", "CC(=O)O"])

# CMD-GEN molecular generation example (auto-fallback if CMD-GEN is not configured)
from scripts.cmdgen_integration import CMDGENGenerator

generator = CMDGENGenerator()
molecules = generator.generate_pocket_based_molecules(
    pdb_file="HJD/data/1p65.pdb", num_molecules=10, ref_ligand="A:1"
)
```

## Principles

This platform builds a reproducible end-to-end workflow around “structure-guided pocket-ligand discovery”:
- Structure-guided generation: Using binding pockets as conditions, candidate molecules are first generated or imported, then screened and optimized
- Physics-based scoring + rule-based filtering: AutoDock Vina for binding affinity scoring, combined with RDKit descriptors and Lipinski's rules for ADMET pre-screening
- Visualization feedback loop: 2D/3D outputs with dashboards and comprehensive reports to assist interpretation and manual review
- Traceable results: All intermediate and final outputs organized by run for reproducible experiments and comparisons

The platform includes two types of “molecular sources”:
1) Rule/template/fragment-driven chemical space expansion (built-in by default, ready to use)
2) External generative models (CMD-GEN: DiffPhar + GCPG; and deep learning research modules for future integration of SOTA methods)

## Datasets & Preprocessing

- Protein Structures
  - Examples: `data/1p65.pdb(.pdbqt)`, `data/AF-Q9GLP0-F1-model_v4.pdb(.pdbqt)`, `data/AF-F1SR53-F1-model_v4.pdb(.pdbqt)`
  - Preprocessing: Hydrogen addition, conversion to pdbqt (using Meeko/AutoDock toolchain), grid center and size configuration in `scripts/config.py`; binding site geometry/surface analysis available via `scripts/binding_site_analyzer.py`
- Ligands
  - Seeds/examples: `data/ligands.sdf`; runtime generation of larger candidate sets by `scripts/ligand_generator.py` (deduplication, normalization, SMILES/SD export)
  - 3D conformations: RDKit ETKDG generation + energy minimization; automatically handled during docking
- Protein–Ligand Pairs
  - Research use: `data/P-L/` (example data for binary classification or interaction modeling)
  - Features: See `deep_learning/data/featurizers.py`
- Results Directory
  - Each run automatically creates `results/run_YYYYMMDD_xxx/` containing `ligands/`, `docking/`, `admet/`, `visualization_2d|3d/`, `reports/`

## Methods & Models

- Computational Chemistry & Rule-Based Methods
  - Docking: AutoDock Vina (`scripts/molecular_docking.py` or `scripts/docking_engine.py` for batch processing, with Meeko/RDKit preprocessing)
  - ADMET: RDKit descriptors (MW, LogP, HBD/HBA, RotB, TPSA, aromaticity, etc.) and Lipinski's Rule of Five (`scripts/admet_analyzer.py`)
  - Visualization: 2D (`scripts/visualization/visualization_2d.py`), 3D (`scripts/visualization_3d.py`)
- External Generative Models (Optional)
  - CMD-GEN integration (`scripts/cmdgen_integration.py`):
    - DiffPhar: Pharmacophore inference from protein structures
    - GCPG: Molecule generation conditioned on pharmacophores (with filtering and fallback support)
- Deep Learning Research Modules (`deep_learning/models/`)
  - EquivariantGNN (SE(3)-Equivariant GNN): Takes protein/ligand graph structures and 3D coordinates as input, outputs node/graph-level representations and affinity scores
  - PocketLigandTransformer: Cross-modal cross-attention between pocket amino acid sequences/geometry and ligand tokens, outputs P-L interaction representations
  - MultiTaskDiscriminator: Multi-task scoring of generated/screened molecules for synthetic accessibility, toxicity, etc.

The above deep learning modules provide runnable research implementations/placeholders for easy replacement and extension.

## Core Model Flowcharts (PNG)

The following are flowcharts (PNG) of core deep learning models, located in `docs/diagrams/png/`:

- Pocket–Ligand Cross-attention Transformer

  ![Pocket–Ligand Transformer](docs/diagrams/png/fig_pl_transformer.png)

- SE(3)-Equivariant Graph Neural Network

  ![SE(3)-Equivariant GNN](docs/diagrams/png/fig_equivariant_gnn.png)

- Multi-Task Discriminator

  ![Multi-Task Discriminator](docs/diagrams/png/fig_multitask_discriminator.png)

## Architecture Diagrams (English, Landscape)

Images only (PNG, high resolution):

- End-to-End Pipeline

  ![End-to-End Pipeline](docs/diagrams/end_to_end_pipeline.png)

- Data Featurization to Model Inputs

  ![Data Featurization to Model Inputs](docs/diagrams/featurization_to_inputs.png)

- SE(3)-Equivariant GNN

  ![SE(3)-Equivariant GNN](docs/diagrams/se3_equivariant_gnn.png)

- Pocket–Ligand Cross-attention Transformer

  ![Pocket–Ligand Transformer](docs/diagrams/pocket_ligand_transformer.png)

- Multi-task Discriminator

  ![Multi-task Discriminator](docs/diagrams/multitask_discriminator.png)

## End-to-End Pipeline

1) Ligand Generation
   - Rule/template/fragment expansion or CMD-GEN generation; outputs CSV/SMILES/SD to `results/.../ligands/`
2) Molecular Docking
   - Invokes AutoDock Vina, automatically prepares pdbqt and 3D conformations; outputs `docking/docking_results.csv`
3) ADMET Analysis
   - Calculates descriptors and compliance; outputs `admet/admet_results.csv`
4) Visualization & Reports
   - 2D: Top-N individual/grid plots and HTML reports → `visualization_2d/`
   - 3D: Single molecules/complexes/binding sites + interactive dashboards/comprehensive reports → `visualization_3d/`
5) Result Management
   - All pipeline outputs archived by run, including `run_info.json` and key file manifests for reproducibility

One-click command-line execution:
```bash
python run_full_workflow.py
```
Or run interactively via the “Full Workflow” page in the web interface.

## Results & Examples

- A typical run generates:
  - Dozens to hundreds of candidate molecules (`ligands/`)
  - `docking_results.csv` (containing affinity, conformation info, pose paths)
  - `admet_results.csv` (descriptors and compliance labels)
  - `interactive_dashboard.html` and `comprehensive_3d_report.html` in `visualization_3d/`
- Top-N determination: Based on docking affinity (lower is better) combined with basic ADMET compliance filtering
- Values vary with protein, parameters, and generation strategy; refer to CSV/HTML files in the corresponding run directory

## Reproducibility & Extension

- Reproducibility:
  - Fix inputs (protein/pocket/parameters), run `run_full_workflow.py`, compare differences across generation strategies or docking parameters
- Extension Directions:
  - Integrate external SOTA models (interfaces provided in `deep_learning_pipeline.py` and `scripts/external_*_infer.py`)
  - Add custom ADMET tasks or replace scoring criteria
  - Batch evaluation across multiple proteins/pockets

## Acknowledgements

- AutoDock Vina, Meeko, RDKit, py3Dmol, Plotly
- CMD-GEN (DiffPhar/GCPG) as optional external generation capability

## Technical Features

### 🔬 Scientific Rigor
- Based on state-of-the-art deep learning and computational chemistry methods
- Integration of multiple mature drug design tools
- Rigorous validation and evaluation workflows

### 🚀 Efficiency & Intelligence
- GPU-accelerated deep learning computation
- Parallelized molecular docking and analysis
- Intelligent molecular generation and optimization

### 🌐 User-Friendly
- Intuitive web interface
- Real-time result visualization
- Detailed analysis report generation

### 🔧 Modular Design
- Loosely coupled modular architecture
- Easy to extend and maintain
- Support for custom configurations

## Development Team

This project is developed by a professional computational chemistry and artificial intelligence team, dedicated to providing advanced computational tools for PRRSV drug discovery.

## License

This project is for academic research use only.

## Contact Us

For questions or suggestions, please contact us via:
- Project Homepage: [GitHub Link]
- Email: [Contact Email]

---

## Project Summary

This project targets small-molecule discovery for PRRSV viral capsid protein–integrin PPI inhibitors, integrating a complete workflow of “deep learning generation → molecular docking → ADMET analysis → 2D/3D visualization → result management”. It supports batch generation and screening of thousands of molecules, emphasizing traceability, reproducibility, and extensibility.

### Feature Matrix
- **[Molecular Generation]** Deep learning generation (Transformer/Diffusion placeholder implementations + external model integration), rule/template library dynamic expansion (thousands of unique molecules), Top-K optimization and Final-N convergence.
- **[Molecular Docking]** AutoDock Vina batch docking; RDKit+Meeko conformation generation, hydrogenation, and optimization; results archived as CSV.
- **[ADMET Analysis]** RDKit descriptors and Lipinski compliance; automatically enables “simplified mode” when RDKit is unavailable.
- **[2D Visualization]** Top-N (up to 1000) individual and grid plots, HTML reports; automatically falls back to ligand CSV when docking results are unavailable.
- **[3D Visualization]** Single-molecule 3D, protein–ligand complexes, binding site analysis, Plotly interactive dashboards, comprehensive reports; falls back to ligand CSV when docking results are unavailable; outputs saved to corresponding run's `visualization_3d/`.
- **[Result Management]** Run-based directory structure, automatic copying of deep learning Phase2/Phase3 outputs to `ligands/`, full workflow traceability.

### Models & Algorithms (SOTA Model Integration Ready)
- **[Generative Models]** Currently provides “placeholder generation + chemical space dynamic expansion” capabilities, with reserved interfaces for external pocket-conditioned Transformer, Diffusion + Reinforcement Learning (RL) models (`deep_learning_pipeline.py`).
- **[Docking Scoring]** AutoDock Vina; RDKit and Meeko for 3D conformation generation and energy minimization (`scripts/docking_engine.py`).
- **[ADMET Metrics]** MW, LogP, HBD/HBA, RotB, TPSA, aromaticity and ring counts, Lipinski's rules (`scripts/admet_analyzer.py`).
- **[Visualization]** 2D based on RDKit rdMolDraw2D (Cairo/SVG) with PIL fallback; 3D based on py3Dmol and Plotly (`scripts/visualization_3d.py`).

### Key Innovations
- **[Dynamic Chemical Space Expansion]** Introduces scaffolds including benzoxazole/five-membered azoles/triazine/thiadiazole/biphenyl/terphenyl/adamantane/bicyclic structures; systematically generates o/m/p-disubstituted benzenes and trisubstituted templates, significantly increasing unique molecule count after deduplication (soft limit ~8000).
- **[Robust Fallback Strategy]** 2D/3D automatically falls back to ligand CSV (`ligands/`) when docking results are unavailable, without interrupting visualization and analysis; enables simplified mode when RDKit/py3Dmol are missing.
- **[High-Throughput Visualization]** 2D Top-N limit increased to 1000, supporting rapid screening of large-scale candidates.
- **[Full Workflow Traceability]** Run-level directory management with file copying and step recording.

### Generated Molecule Performance & Evaluation Metrics (Examples)
- **[Diversity]** Covers halogenated aromatic rings, cyano/nitro/amide groups, five/six-membered heterocycles, fused rings, polycyclic aromatic hydrocarbons, alicyclic structures (adamantane/bicyclic), etc.; o/m/p positional control enhances spatial configuration diversity.
- **[Typical Distribution]** MW commonly 100–400 Da (expandable to ~500); LogP mostly in −1~5 range; affinity used for Top-N ranking and filtering.
- **[Report Contents]**
  - 2D: Individual plots, grid plots, HTML summary reports (`visualization_2d/`)
  - 3D: Single-molecule 3D, complexes, binding sites, interactive dashboards and comprehensive reports (`visualization_3d/`)
- Note: Specific values vary with run parameters/models/docking conditions; refer to reports and dashboards.

### Typical Workflow
1. **Molecular Generation**: Set “number of molecules to generate” (UI supports up to 10000); Phase2 generation → Phase3 optimization (Top-K dynamically scales with request volume). Phase outputs CSV automatically copied to current run's `ligands/`.
2. **Molecular Docking**: AutoDock Vina batch docking, results written to `docking/docking_results.csv`.
3. **ADMET Analysis**: Batch calculation of descriptors and Lipinski compliance, generating analysis CSV/reports.
4. **Visualization**:
   - 2D: Affinity-sorted Top-N (≤1000) + grids + reports → `visualization_2d/`
   - 3D: Single molecules/complexes/sites + dashboards + comprehensive reports → `visualization_3d/`

### Engineering Structure (Key Points)
```
HJD/
├── unified_web_interface.py         # Unified web interface (parameters, fallback, run directory selector, etc.)
├── deep_learning_pipeline.py        # Generation and optimization pipeline (library expansion + external model integration interface)
├── scripts/
│   ├── docking_engine.py            # AutoDock Vina batch docking + Meeko/RDKit preprocessing
│   ├── molecule_2d_generator.py     # 2D visualization (Top-N≤1000; docking priority, ligand fallback)
│   ├── visualization_3d.py          # 3D visualization (single molecules/complexes/sites/dashboards/comprehensive reports)
│   ├── result_manager.py            # Run-level directory management with file copying and step recording
│   └── admet_analyzer.py            # ADMET descriptors and Lipinski rules (simplified mode fallback)
└── results/
    └── run_YYYYMMDD_xxx/
        ├── ligands/                 # Generation/optimization output CSV (Phase2/3 auto-copied here)
        ├── docking/                 # docking_results.csv
        ├── visualization_2d/        # individual/, grids/, reports/
        ├── visualization_3d/        # Single-page visualizations and comprehensive report HTML
        └── reports/                 # Other analysis reports
```

### Dependencies & Execution
- **[Environment]** Python 3.9+; recommended to install RDKit, py3Dmol, Plotly, and Streamlit.
- **[Installation]** `pip install -r requirements.txt`
- **[Launch]** `python start_project.py`, then access `http://localhost:8501` in browser
- **[Notes]**
  - 2D/3D visualization automatically falls back to current/latest run's `ligands/*.csv` when docking results are unavailable
  - 3D visualization reports are always saved to current run's `visualization_3d/`

### Example Visualizations (From a Sample Run)

> Note: The following example images are existing files in the repository. Actual runs will generate corresponding charts and reports in your `results/run_*/visualization_2d|3d/` directories.

- **2D Molecular Grid (Top-N)**
  ![2D Grid](results/run_20250918_008/visualization_2d/grids/Top_165_Molecules_grid.png)

- **ADMET/Scoring Example Plots**
  Binding Affinity Distribution:
  ![Binding Affinity Distribution](experiment_report/binding_affinity_distribution.png)

  Top-10 Molecules Bar Chart:
  ![Top 10 Molecules](experiment_report/top_10_molecules.png)

  Lipinski Compliance:
  ![Lipinski Compliance](experiment_report/lipinski_compliance.png)

  ADMET Properties (Example):
  ![ADMET Properties](experiment_report/admet_properties.png)

- **3D Interactive Dashboards/Reports**
  After generating 3D reports, the following will be created in the corresponding run's `visualization_3d/`:
  - `interactive_dashboard.html` (interactive)
  - `interactive_dashboard.png` (static export, requires `kaleido`)
  - `comprehensive_3d_report.html` (comprehensive report)

> For static PNG export, ensure `kaleido` is installed (already included in `requirements.txt`). If not installed, PNG export will be automatically skipped without affecting HTML report generation.

---

**PRRSV Deep Learning Inhibitor Design Platform - Making Drug Design Smarter**
