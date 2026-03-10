#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaffold-Level Novelty Analysis (Reviewer R2-Q8)

Distinguishes scaffold-level novelty from fingerprint-level novelty:
  1. Murcko scaffold extraction (generic + side-chain-bearing)
  2. Scaffold novelty: fraction of generated scaffolds NOT seen in training set
  3. Fingerprint novelty: Tanimoto distance-based novelty (ECFP4)
  4. Scaffold diversity: number of unique scaffolds / total molecules
  5. Publication-ready figures: scaffold frequency distribution, Venn diagram

Usage:
  python scripts/scaffold_novelty.py \
      --generated results/run_xxx/ligands/generated.csv \
      --training data/ligands.sdf \
      --output_dir results/scaffold_analysis
  python scripts/scaffold_novelty.py --demo  # run with demo SMILES
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ========================= Scaffold Extraction =========================

def get_murcko_scaffold(smiles: str, generic: bool = False) -> Optional[str]:
    """Extract Murcko scaffold from SMILES.

    Args:
        smiles: input SMILES string
        generic: if True, return generic scaffold (all atoms -> C, all bonds -> single)

    Returns:
        Scaffold SMILES or None if extraction fails
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        scaffold_smi = Chem.MolToSmiles(core)
        return scaffold_smi if scaffold_smi else None
    except Exception:
        return None


def get_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprint (ECFP4 by default)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ========================= Novelty Metrics =========================

def compute_scaffold_novelty(
    generated_smiles: List[str],
    training_smiles: List[str],
    generic: bool = False,
) -> Dict:
    """Compute scaffold-level novelty metrics.

    Returns dict with:
        - scaffold_novelty: fraction of generated scaffolds not in training
        - scaffold_diversity: unique scaffolds / total valid molecules
        - n_unique_scaffolds: count of unique scaffolds in generated set
        - n_novel_scaffolds: count of scaffolds not seen in training
        - novel_scaffold_list: list of novel scaffold SMILES
    """
    # Extract scaffolds
    gen_scaffolds = []
    for smi in generated_smiles:
        sc = get_murcko_scaffold(smi, generic=generic)
        if sc is not None:
            gen_scaffolds.append(sc)

    train_scaffolds = set()
    for smi in training_smiles:
        sc = get_murcko_scaffold(smi, generic=generic)
        if sc is not None:
            train_scaffolds.add(sc)

    unique_gen = set(gen_scaffolds)
    novel = unique_gen - train_scaffolds

    n_valid = len(gen_scaffolds)
    scaffold_novelty = len(novel) / max(len(unique_gen), 1)
    scaffold_diversity = len(unique_gen) / max(n_valid, 1)

    # Frequency distribution
    scaffold_counts = Counter(gen_scaffolds)
    top_scaffolds = scaffold_counts.most_common(20)

    return {
        "scaffold_novelty": round(scaffold_novelty, 4),
        "scaffold_diversity": round(scaffold_diversity, 4),
        "n_total_valid": n_valid,
        "n_unique_scaffolds": len(unique_gen),
        "n_training_scaffolds": len(train_scaffolds),
        "n_novel_scaffolds": len(novel),
        "n_shared_scaffolds": len(unique_gen & train_scaffolds),
        "top_20_scaffolds": [{"scaffold": s, "count": c} for s, c in top_scaffolds],
        "novel_scaffold_list": sorted(novel)[:50],
        "generic_scaffold": generic,
    }


def compute_fingerprint_novelty(
    generated_smiles: List[str],
    training_smiles: List[str],
    threshold: float = 0.4,
) -> Dict:
    """Compute fingerprint-level (Tanimoto) novelty.

    A generated molecule is 'novel' if its max Tanimoto similarity
    to ANY training molecule is below `threshold`.

    Returns dict with:
        - fp_novelty: fraction of generated molecules that are novel
        - mean_nearest_tanimoto: avg nearest-neighbor similarity to training
        - internal_diversity: 1 - avg pairwise Tanimoto within generated set
    """
    gen_fps = []
    for smi in generated_smiles:
        fp = get_fingerprint(smi)
        if fp is not None:
            gen_fps.append(fp)

    train_fps = []
    for smi in training_smiles:
        fp = get_fingerprint(smi)
        if fp is not None:
            train_fps.append(fp)

    if not gen_fps:
        return {"fp_novelty": 0.0, "mean_nearest_tanimoto": 0.0,
                "internal_diversity": 0.0, "n_valid_generated": 0}

    # Nearest-neighbor similarity to training set
    nn_sims = []
    novel_count = 0
    for gfp in gen_fps:
        if train_fps:
            max_sim = max(tanimoto_similarity(gfp, tfp) for tfp in train_fps)
        else:
            max_sim = 0.0
        nn_sims.append(max_sim)
        if max_sim < threshold:
            novel_count += 1

    # Internal diversity (sample if large)
    n = len(gen_fps)
    if n > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 500, replace=False)
        sample = [gen_fps[i] for i in idx]
    else:
        sample = gen_fps

    pair_sims = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            pair_sims.append(tanimoto_similarity(sample[i], sample[j]))
    internal_div = 1.0 - (np.mean(pair_sims) if pair_sims else 0.0)

    return {
        "fp_novelty": round(novel_count / len(gen_fps), 4),
        "fp_novelty_threshold": threshold,
        "mean_nearest_tanimoto": round(float(np.mean(nn_sims)), 4),
        "std_nearest_tanimoto": round(float(np.std(nn_sims)), 4),
        "internal_diversity": round(float(internal_div), 4),
        "n_valid_generated": len(gen_fps),
        "n_training": len(train_fps),
    }


def run_full_scaffold_analysis(
    generated_smiles: List[str],
    training_smiles: List[str],
    output_dir: str = "results/scaffold_analysis",
) -> Dict:
    """Run complete scaffold + fingerprint novelty analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Scaffold & Fingerprint Novelty Analysis (R2-Q8)")
    print("=" * 60)

    # 1. Scaffold novelty (side-chain-bearing)
    print("\n1. Murcko Scaffold Novelty (with side chains):")
    sc_results = compute_scaffold_novelty(generated_smiles, training_smiles,
                                          generic=False)
    print(f"   Scaffold novelty:   {sc_results['scaffold_novelty']:.4f}")
    print(f"   Scaffold diversity: {sc_results['scaffold_diversity']:.4f}")
    print(f"   Unique scaffolds:   {sc_results['n_unique_scaffolds']}")
    print(f"   Novel scaffolds:    {sc_results['n_novel_scaffolds']}")
    print(f"   Shared scaffolds:   {sc_results['n_shared_scaffolds']}")

    # 2. Generic scaffold novelty
    print("\n2. Generic Scaffold Novelty:")
    gen_sc = compute_scaffold_novelty(generated_smiles, training_smiles,
                                      generic=True)
    print(f"   Generic scaffold novelty:   {gen_sc['scaffold_novelty']:.4f}")
    print(f"   Generic unique scaffolds:   {gen_sc['n_unique_scaffolds']}")
    print(f"   Generic novel scaffolds:    {gen_sc['n_novel_scaffolds']}")

    # 3. Fingerprint novelty
    print("\n3. Fingerprint (ECFP4) Novelty:")
    fp_results = compute_fingerprint_novelty(generated_smiles, training_smiles)
    print(f"   FP novelty (Tc<0.4):     {fp_results['fp_novelty']:.4f}")
    print(f"   Mean nearest Tanimoto:   {fp_results['mean_nearest_tanimoto']:.4f}")
    print(f"   Internal diversity:      {fp_results['internal_diversity']:.4f}")

    # 4. Comparison summary
    print("\n4. Scaffold vs Fingerprint Novelty Comparison:")
    print(f"   {'Metric':<30s} {'Scaffold':>10s} {'Fingerprint':>12s}")
    print(f"   {'-'*52}")
    print(f"   {'Novelty':<30s} {sc_results['scaffold_novelty']:>10.4f}"
          f" {fp_results['fp_novelty']:>12.4f}")
    print(f"   {'Diversity':<30s} {sc_results['scaffold_diversity']:>10.4f}"
          f" {fp_results['internal_diversity']:>12.4f}")

    # Combine results
    combined = {
        "murcko_scaffold": sc_results,
        "generic_scaffold": gen_sc,
        "fingerprint_ecfp4": fp_results,
    }

    # Save JSON
    json_path = os.path.join(output_dir, "scaffold_novelty_results.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results: {json_path}")

    # Save Markdown
    _write_scaffold_markdown(combined, output_dir)

    # Generate figures
    try:
        _generate_scaffold_figures(combined, sc_results, output_dir)
    except ImportError as e:
        print(f"[WARNING] Figures skipped (missing dependency): {e}")

    return combined



# ========================= Output Helpers =========================

def _write_scaffold_markdown(combined: Dict, output_dir: str):
    """Write Markdown summary of scaffold novelty analysis."""
    md_path = os.path.join(output_dir, "scaffold_novelty_summary.md")
    sc = combined["murcko_scaffold"]
    gs = combined["generic_scaffold"]
    fp = combined["fingerprint_ecfp4"]

    lines = [
        "# Scaffold & Fingerprint Novelty Analysis\n",
        "## Summary\n",
        "| Metric | Murcko Scaffold | Generic Scaffold | ECFP4 Fingerprint |",
        "|:-------|----------------:|-----------------:|------------------:|",
        f"| Novelty | {sc['scaffold_novelty']:.4f} | {gs['scaffold_novelty']:.4f}"
        f" | {fp['fp_novelty']:.4f} |",
        f"| Diversity | {sc['scaffold_diversity']:.4f} | {gs['scaffold_diversity']:.4f}"
        f" | {fp['internal_diversity']:.4f} |",
        f"| Unique count | {sc['n_unique_scaffolds']} | {gs['n_unique_scaffolds']}"
        f" | {fp['n_valid_generated']} |",
        f"| Novel count | {sc['n_novel_scaffolds']} | {gs['n_novel_scaffolds']} | — |",
        "",
        "## Top-20 Most Frequent Scaffolds\n",
        "| Rank | Scaffold SMILES | Count |",
        "|-----:|:----------------|------:|",
    ]
    for i, item in enumerate(sc.get("top_20_scaffolds", []), 1):
        lines.append(f"| {i} | `{item['scaffold']}` | {item['count']} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown summary: {md_path}")


def _generate_scaffold_figures(combined: Dict, sc_results: Dict, output_dir: str):
    """Generate scaffold analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})

    # Figure 1: Scaffold vs Fingerprint novelty comparison bar chart
    sc = combined["murcko_scaffold"]
    gs = combined["generic_scaffold"]
    fp = combined["fingerprint_ecfp4"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Novelty comparison
    labels = ["Murcko\nScaffold", "Generic\nScaffold", "ECFP4\nFingerprint"]
    novelty_vals = [sc["scaffold_novelty"], gs["scaffold_novelty"], fp["fp_novelty"]]
    colors = ["#2ECC71", "#3498DB", "#E74C3C"]
    axes[0].bar(labels, novelty_vals, color=colors, width=0.5, edgecolor="black")
    axes[0].set_ylabel("Novelty")
    axes[0].set_title("Novelty: Scaffold vs Fingerprint", fontweight="bold")
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(novelty_vals):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # Diversity comparison
    div_vals = [sc["scaffold_diversity"], gs["scaffold_diversity"],
                fp["internal_diversity"]]
    axes[1].bar(labels, div_vals, color=colors, width=0.5, edgecolor="black")
    axes[1].set_ylabel("Diversity")
    axes[1].set_title("Diversity: Scaffold vs Fingerprint", fontweight="bold")
    axes[1].set_ylim(0, 1.1)
    for i, v in enumerate(div_vals):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "scaffold_vs_fingerprint_novelty.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # Figure 2: Top-20 scaffold frequency
    top = sc_results.get("top_20_scaffolds", [])
    if top:
        fig, ax = plt.subplots(figsize=(10, 6))
        scaff_labels = [f"S{i+1}" for i in range(len(top))]
        counts = [item["count"] for item in top]
        ax.barh(scaff_labels[::-1], counts[::-1], color="#3498DB", edgecolor="black")
        ax.set_xlabel("Frequency")
        ax.set_title("Top-20 Most Frequent Murcko Scaffolds", fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "top20_scaffold_frequency.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")

    # Figure 3: Venn-style overlap
    fig, ax = plt.subplots(figsize=(6, 5))
    n_gen_only = sc["n_novel_scaffolds"]
    n_shared = sc["n_shared_scaffolds"]
    n_train_only = sc["n_training_scaffolds"] - n_shared
    data = [n_train_only, n_shared, n_gen_only]
    labels_v = ["Training\nonly", "Shared", "Generated\nonly (Novel)"]
    colors_v = ["#95A5A6", "#F39C12", "#2ECC71"]
    ax.bar(labels_v, data, color=colors_v, width=0.5, edgecolor="black")
    for i, v in enumerate(data):
        ax.text(i, v + max(data) * 0.02, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Number of Unique Scaffolds")
    ax.set_title("Scaffold Overlap: Generated vs Training", fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "scaffold_overlap.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


# ========================= SMILES Loading =========================

def load_smiles_from_file(filepath: str) -> List[str]:
    """Load SMILES from CSV, SDF, or plain text file."""
    ext = os.path.splitext(filepath)[1].lower()
    smiles_list = []

    if ext == ".sdf":
        try:
            from rdkit import Chem
            suppl = Chem.SDMolSupplier(filepath)
            for mol in suppl:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    if smi:
                        smiles_list.append(smi)
        except ImportError:
            logger.error("RDKit required to read SDF files.")
    elif ext == ".csv":
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = (row.get("smiles") or row.get("SMILES")
                       or row.get("Smiles") or "")
                if smi.strip():
                    smiles_list.append(smi.strip())
    else:  # plain text, one SMILES per line
        with open(filepath, "r") as f:
            for line in f:
                smi = line.strip()
                if smi and not smi.startswith("#"):
                    smiles_list.append(smi)

    print(f"  Loaded {len(smiles_list)} SMILES from {filepath}")
    return smiles_list


# Demo SMILES for testing without real data
DEMO_GENERATED = [
    "Cc1ccc(-c2nc(N)nc(Nc3ccc(F)cc3)n2)cc1",
    "COc1ccc2c(N)nc(Nc3ccccc3)nc2c1",
    "c1ccc(-c2ccnc(Nc3cccc(O)c3)n2)cc1",
    "CC(=O)Nc1ccc(-c2nc3ccccc3n2C)cc1",
    "Fc1ccc(-c2nnc(Nc3ccccc3)o2)cc1",
    "c1ccc(-c2cnc(Nc3ccc(Cl)cc3)nc2N)cc1",
    "COc1cc(-c2nc(N)nc(N)n2)cc(OC)c1O",
    "CC(C)c1ccc(-n2c(N)nc3ccccc32)cc1",
    "Nc1nc(-c2ccc(Br)cc2)nc2ccccc12",
    "c1ccc(-c2nc(Nc3ccccn3)nc(N)n2)cc1",
    "O=C(Nc1ccccc1)c1cnc2ccccc2n1",
    "Cc1noc(-c2ccc(F)cc2)c1NC(=O)c1ccccc1",
]

DEMO_TRAINING = [
    "c1ccc(-c2nncn2-c2ccccc2)cc1",
    "Nc1nc(-c2ccccc2)nc2ccccc12",
    "CC(=O)Nc1ccc(-c2ccccc2)cc1",
    "c1ccc(-c2ccnc(-c3ccccc3)n2)cc1",
    "COc1ccc(-c2nc(N)nc(N)n2)cc1",
]


# ========================= CLI Entry Point =========================

def main():
    parser = argparse.ArgumentParser(
        description="Scaffold & fingerprint novelty analysis (Reviewer R2-Q8)."
    )
    parser.add_argument("--generated", type=str, default="",
                        help="Path to generated molecules (CSV/SDF/TXT)")
    parser.add_argument("--training", type=str, default="",
                        help="Path to training molecules (CSV/SDF/TXT)")
    parser.add_argument("--output_dir", type=str,
                        default="results/scaffold_analysis",
                        help="Output directory for results and figures")
    parser.add_argument("--demo", action="store_true",
                        help="Run with built-in demo SMILES")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        print("\nRunning with demo SMILES...")
        gen_smiles = DEMO_GENERATED
        train_smiles = DEMO_TRAINING
    elif args.generated and args.training:
        print(f"\nLoading generated molecules: {args.generated}")
        gen_smiles = load_smiles_from_file(args.generated)
        print(f"Loading training molecules: {args.training}")
        train_smiles = load_smiles_from_file(args.training)
    else:
        print("ERROR: Provide --generated and --training, or use --demo")
        print("Example:")
        print("  python scripts/scaffold_novelty.py --demo")
        print("  python scripts/scaffold_novelty.py \\")
        print("      --generated results/run_xxx/ligands/generated.csv \\")
        print("      --training data/ligands.sdf")
        sys.exit(1)

    if not gen_smiles:
        print("ERROR: No valid generated SMILES found.")
        sys.exit(1)

    run_full_scaffold_analysis(gen_smiles, train_smiles, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
