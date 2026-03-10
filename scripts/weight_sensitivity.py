#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Objective Weight Sensitivity Analysis (Reviewer R2-Q5)

Analyzes how candidate ranking stability changes under different
multi-objective weight configurations:
  1. Dirichlet sampling of weight vectors
  2. Re-ranking candidates under each weight vector
  3. Kendall's tau ranking correlation for stability
  4. Heatmap and line plots of ranking sensitivity

Usage:
  python scripts/weight_sensitivity.py --results results/run_xxx/docking/docking_results.csv
  python scripts/weight_sensitivity.py --demo  # run with demo data
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ========================= Default Objective Names =========================
# These correspond to the scoring dimensions in MolFoundry's multi-task scorer
DEFAULT_OBJECTIVES = [
    "binding_affinity",   # AutoDock Vina score (lower = better)
    "sa_score",           # Synthetic accessibility (lower = easier)
    "qed",               # Quantitative Estimate of Drug-likeness (higher = better)
    "lipinski_score",     # Lipinski compliance (higher = better)
]

# Sign convention: +1 means higher is better, -1 means lower is better
OBJECTIVE_SIGNS = {
    "binding_affinity": -1,  # lower (more negative) is better
    "sa_score": -1,          # lower is easier to synthesize
    "qed": +1,               # higher is more drug-like
    "lipinski_score": +1,    # higher compliance is better
    "tpsa": -1,              # context-dependent, usually want moderate
    "logp": -1,              # context-dependent
}


# ========================= Core Functions =========================

def normalize_scores(scores: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Min-max normalize each objective column, respecting sign convention.

    After normalization, higher values are always better.
    """
    n_samples, n_obj = scores.shape
    normalized = np.zeros_like(scores, dtype=float)

    for j in range(n_obj):
        col = scores[:, j] * signs[j]  # flip sign so higher = better
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-12:
            normalized[:, j] = (col - col_min) / (col_max - col_min)
        else:
            normalized[:, j] = 0.5  # constant column

    return normalized


def weighted_rank(normalized_scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted scalarized score and return ranking (0 = best)."""
    composite = normalized_scores @ weights
    return np.argsort(-composite)  # descending order (higher = better)


def kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Compute Kendall's tau-b correlation between two rankings."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(rank_a, rank_b)
    return float(tau) if not np.isnan(tau) else 1.0


def sample_dirichlet_weights(
    n_objectives: int,
    n_samples: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Sample weight vectors from Dirichlet distribution.

    Args:
        n_objectives: number of objectives
        n_samples: number of weight vectors to sample
        alpha: concentration parameter (1.0 = uniform, >1 = concentrated)
        seed: random seed

    Returns:
        (n_samples, n_objectives) array of weight vectors
    """
    rng = np.random.RandomState(seed)
    alphas = np.ones(n_objectives) * alpha
    return rng.dirichlet(alphas, size=n_samples)


def run_sensitivity_analysis(
    scores: np.ndarray,
    objective_names: List[str],
    molecule_ids: Optional[List[str]] = None,
    n_weight_samples: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
    output_dir: str = "results/weight_sensitivity",
) -> Dict:
    """Run full multi-objective weight sensitivity analysis.

    Args:
        scores: (n_molecules, n_objectives) raw score matrix
        objective_names: list of objective names
        molecule_ids: optional molecule identifiers
        n_weight_samples: number of Dirichlet weight samples
        alpha: Dirichlet concentration parameter
        seed: random seed
        output_dir: output directory

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    n_mol, n_obj = scores.shape
    assert n_obj == len(objective_names), \
        f"Score columns ({n_obj}) != objectives ({len(objective_names)})"

    if molecule_ids is None:
        molecule_ids = [f"mol_{i}" for i in range(n_mol)]

    signs = np.array([OBJECTIVE_SIGNS.get(name, +1) for name in objective_names])

    print("=" * 60)
    print("  Multi-Objective Weight Sensitivity Analysis (R2-Q5)")
    print("=" * 60)
    print(f"\n  Molecules:  {n_mol}")
    print(f"  Objectives: {objective_names}")
    print(f"  Signs:      {signs.tolist()}")
    print(f"  Dirichlet:  alpha={alpha}, n_samples={n_weight_samples}")


    # Step 1: Normalize scores
    norm_scores = normalize_scores(scores, signs)

    # Step 2: Default (equal) weight ranking
    default_weights = np.ones(n_obj) / n_obj
    default_ranking = weighted_rank(norm_scores, default_weights)
    default_rank_map = np.argsort(default_ranking)  # molecule -> rank position

    # Step 3: Sample Dirichlet weight vectors
    weight_samples = sample_dirichlet_weights(n_obj, n_weight_samples, alpha, seed)

    # Step 4: Compute rankings under each weight vector
    all_rankings = []
    tau_values = []

    for w in weight_samples:
        ranking = weighted_rank(norm_scores, w)
        rank_map = np.argsort(ranking)
        all_rankings.append(rank_map)
        tau = kendall_tau(default_rank_map, rank_map)
        tau_values.append(tau)

    tau_arr = np.array(tau_values)

    # Step 5: Per-molecule rank statistics
    rank_matrix = np.array(all_rankings)  # (n_samples, n_mol)
    mean_ranks = rank_matrix.mean(axis=0)
    std_ranks = rank_matrix.std(axis=0)
    min_ranks = rank_matrix.min(axis=0)
    max_ranks = rank_matrix.max(axis=0)

    # Step 6: Print results
    print(f"\n--- Kendall's Tau Stability ---")
    print(f"  Mean tau:   {tau_arr.mean():.4f}")
    print(f"  Std tau:    {tau_arr.std():.4f}")
    print(f"  Min tau:    {tau_arr.min():.4f}")
    print(f"  Max tau:    {tau_arr.max():.4f}")
    print(f"  Median tau: {np.median(tau_arr):.4f}")

    print(f"\n--- Top-10 Most Stable Molecules (lowest rank std) ---")
    stable_idx = np.argsort(std_ranks)[:10]
    print(f"  {'ID':<12s} {'Mean Rank':>10s} {'Std':>8s} {'Range':>12s}")
    for idx in stable_idx:
        print(f"  {molecule_ids[idx]:<12s} {mean_ranks[idx]:>10.1f} "
              f"{std_ranks[idx]:>8.2f} [{min_ranks[idx]}-{max_ranks[idx]}]")

    print(f"\n--- Top-10 Most Sensitive Molecules (highest rank std) ---")
    sensitive_idx = np.argsort(-std_ranks)[:10]
    for idx in sensitive_idx:
        print(f"  {molecule_ids[idx]:<12s} {mean_ranks[idx]:>10.1f} "
              f"{std_ranks[idx]:>8.2f} [{min_ranks[idx]}-{max_ranks[idx]}]")

    # Build results
    results = {
        "n_molecules": n_mol,
        "n_objectives": n_obj,
        "objective_names": objective_names,
        "n_weight_samples": n_weight_samples,
        "dirichlet_alpha": alpha,
        "kendall_tau": {
            "mean": round(float(tau_arr.mean()), 4),
            "std": round(float(tau_arr.std()), 4),
            "min": round(float(tau_arr.min()), 4),
            "max": round(float(tau_arr.max()), 4),
            "median": round(float(np.median(tau_arr)), 4),
        },
        "per_molecule_stats": [
            {
                "id": molecule_ids[i],
                "mean_rank": round(float(mean_ranks[i]), 2),
                "std_rank": round(float(std_ranks[i]), 2),
                "min_rank": int(min_ranks[i]),
                "max_rank": int(max_ranks[i]),
            }
            for i in range(n_mol)
        ],
    }

    # Save JSON
    json_path = os.path.join(output_dir, "weight_sensitivity_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results: {json_path}")

    # Save Markdown
    _write_sensitivity_markdown(results, output_dir)

    # Generate figures
    try:
        _generate_sensitivity_figures(
            tau_arr, mean_ranks, std_ranks, molecule_ids,
            weight_samples, objective_names, output_dir
        )
    except ImportError as e:
        print(f"[WARNING] Figures skipped (missing dependency): {e}")

    return results


# ========================= Output Helpers =========================

def _write_sensitivity_markdown(results: Dict, output_dir: str):
    """Write Markdown summary."""
    md_path = os.path.join(output_dir, "weight_sensitivity_summary.md")
    tau = results["kendall_tau"]
    lines = [
        "# Multi-Objective Weight Sensitivity Analysis\n",
        f"- **Molecules**: {results['n_molecules']}",
        f"- **Objectives**: {', '.join(results['objective_names'])}",
        f"- **Weight samples**: {results['n_weight_samples']} "
        f"(Dirichlet alpha={results['dirichlet_alpha']})\n",
        "## Kendall's Tau Ranking Stability\n",
        "| Statistic | Value |",
        "|:----------|------:|",
        f"| Mean | {tau['mean']:.4f} |",
        f"| Std | {tau['std']:.4f} |",
        f"| Min | {tau['min']:.4f} |",
        f"| Max | {tau['max']:.4f} |",
        f"| Median | {tau['median']:.4f} |",
        "",
        "## Per-Molecule Rank Statistics (sorted by std)\n",
        "| Molecule | Mean Rank | Std | Range |",
        "|:---------|----------:|----:|------:|",
    ]
    sorted_mols = sorted(results["per_molecule_stats"], key=lambda x: x["std_rank"])
    for m in sorted_mols[:20]:
        lines.append(
            f"| {m['id']} | {m['mean_rank']:.1f} | {m['std_rank']:.2f} "
            f"| [{m['min_rank']}-{m['max_rank']}] |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown summary: {md_path}")


def _generate_sensitivity_figures(
    tau_arr: np.ndarray,
    mean_ranks: np.ndarray,
    std_ranks: np.ndarray,
    molecule_ids: List[str],
    weight_samples: np.ndarray,
    objective_names: List[str],
    output_dir: str,
):
    """Generate weight sensitivity analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})

    # Figure 1: Kendall's tau distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tau_arr, bins=50, color="#3498DB", edgecolor="black", alpha=0.8)
    ax.axvline(tau_arr.mean(), color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Mean τ = {tau_arr.mean():.3f}")
    ax.axvline(np.median(tau_arr), color="#2ECC71", linestyle="--", linewidth=2,
               label=f"Median τ = {np.median(tau_arr):.3f}")
    ax.set_xlabel("Kendall's τ (vs. equal-weight ranking)")
    ax.set_ylabel("Frequency")
    ax.set_title("Ranking Stability Under Weight Perturbation", fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "kendall_tau_distribution.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # Figure 2: Mean rank vs rank std scatter (stability plot)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(mean_ranks, std_ranks, c=std_ranks, cmap="RdYlGn_r",
                         s=50, edgecolors="black", linewidth=0.5, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Rank Std (instability)")
    ax.set_xlabel("Mean Rank (lower = better)")
    ax.set_ylabel("Rank Std (higher = more sensitive)")
    ax.set_title("Molecule Ranking Stability", fontweight="bold")

    # Annotate top-5 most sensitive
    sensitive_idx = np.argsort(-std_ranks)[:5]
    for idx in sensitive_idx:
        ax.annotate(molecule_ids[idx],
                     (mean_ranks[idx], std_ranks[idx]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color="#E74C3C", fontweight="bold")

    # Annotate top-3 most stable (low std, low mean rank)
    stable_idx = np.argsort(std_ranks)[:3]
    for idx in stable_idx:
        ax.annotate(molecule_ids[idx],
                     (mean_ranks[idx], std_ranks[idx]),
                     textcoords="offset points", xytext=(5, -10),
                     fontsize=7, color="#2ECC71", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "rank_stability_scatter.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # Figure 3: Weight distribution heatmap (sampled weights)
    n_show = min(50, len(weight_samples))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(weight_samples[:n_show], ax=ax,
                xticklabels=objective_names, cmap="YlOrRd",
                cbar_kws={"label": "Weight"},
                linewidths=0.1, linecolor="white")
    ax.set_ylabel(f"Sample index (showing {n_show}/{len(weight_samples)})")
    ax.set_title("Sampled Dirichlet Weight Vectors", fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "weight_heatmap.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


# ========================= Demo Data =========================

def generate_demo_data(n_molecules: int = 30, seed: int = 42):
    """Generate synthetic multi-objective scores for demo/testing."""
    rng = np.random.RandomState(seed)

    # Simulate realistic score ranges
    binding_affinity = rng.uniform(-10.0, -3.0, n_molecules)   # kcal/mol
    sa_score = rng.uniform(1.5, 6.0, n_molecules)              # 1-10 scale
    qed = rng.uniform(0.2, 0.9, n_molecules)                   # 0-1 scale
    lipinski = rng.choice([0, 1, 2, 3, 4, 5], n_molecules,
                          p=[0.02, 0.03, 0.05, 0.15, 0.35, 0.40])

    scores = np.column_stack([binding_affinity, sa_score, qed, lipinski])
    mol_ids = [f"mol_{i:03d}" for i in range(n_molecules)]
    obj_names = ["binding_affinity", "sa_score", "qed", "lipinski_score"]

    return scores, obj_names, mol_ids


def load_scores_from_csv(filepath: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load multi-objective scores from CSV file.

    Expected format: first column = molecule ID, remaining columns = objectives.
    """
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Empty or invalid CSV: {filepath}")

        # First column is molecule ID, rest are objectives
        id_col = fieldnames[0]
        obj_names = list(fieldnames[1:])

        mol_ids = []
        rows = []
        for row in reader:
            mol_ids.append(row[id_col])
            rows.append([float(row[col]) for col in obj_names])

    scores = np.array(rows)
    print(f"  Loaded {len(mol_ids)} molecules × {len(obj_names)} objectives from {filepath}")
    return scores, obj_names, mol_ids


# ========================= CLI Entry Point =========================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective weight sensitivity analysis (Reviewer R2-Q5)."
    )
    parser.add_argument("--results", type=str, default="",
                        help="Path to CSV with multi-objective scores")
    parser.add_argument("--output_dir", type=str,
                        default="results/weight_sensitivity",
                        help="Output directory for results and figures")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo data")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of Dirichlet weight samples (default: 1000)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet concentration parameter (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_demo_mols", type=int, default=30,
                        help="Number of demo molecules (default: 30)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.demo:
        print("\nGenerating synthetic demo data...")
        scores, obj_names, mol_ids = generate_demo_data(
            n_molecules=args.n_demo_mols, seed=args.seed
        )
    elif args.results:
        print(f"\nLoading scores from: {args.results}")
        scores, obj_names, mol_ids = load_scores_from_csv(args.results)
    else:
        print("ERROR: Provide --results <csv_path> or use --demo")
        print("Example:")
        print("  python scripts/weight_sensitivity.py --demo")
        print("  python scripts/weight_sensitivity.py \\")
        print("      --results results/run_xxx/docking/docking_results.csv \\")
        print("      --n_samples 2000 --alpha 1.0")
        sys.exit(1)

    run_sensitivity_analysis(
        scores=scores,
        objective_names=obj_names,
        molecule_ids=mol_ids,
        n_weight_samples=args.n_samples,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
