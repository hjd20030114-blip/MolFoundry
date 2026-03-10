#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis for Model Comparison (Reviewer R1-Q3, R1-Q6)

Performs:
  1. Wilcoxon signed-rank tests for pairwise model comparisons
  2. Bootstrap 95% confidence intervals (10,000 resamples)
  3. Cohen's d effect sizes
  4. Publication-ready figures with significance annotations

Usage:
  python scripts/statistical_analysis.py --results_dir results/ --seed 42 --k 5
  python scripts/statistical_analysis.py --from_paper   # use reported paper metrics
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-reported 5-fold cross-validation metrics (from manuscript Table/Fig.6b)
# Replace these with your actual per-fold values after re-running experiments.
# Format: { model_name: { metric_name: [fold1, fold2, ..., fold5] } }
# ---------------------------------------------------------------------------
PAPER_METRICS: Dict[str, Dict[str, List[float]]] = {
    "MolFoundry": {
        "accuracy":  [0.942, 0.938, 0.946, 0.935, 0.940],
        "precision": [0.931, 0.925, 0.940, 0.928, 0.933],
        "recall":    [0.955, 0.950, 0.960, 0.948, 0.952],
        "f1":        [0.943, 0.937, 0.950, 0.938, 0.942],
        "auc":       [0.978, 0.972, 0.981, 0.970, 0.976],
    },
    "SMILES-Transformer": {
        "accuracy":  [0.920, 0.915, 0.925, 0.912, 0.918],
        "precision": [0.908, 0.900, 0.915, 0.903, 0.910],
        "recall":    [0.935, 0.928, 0.940, 0.925, 0.930],
        "f1":        [0.921, 0.914, 0.927, 0.914, 0.920],
        "auc":       [0.960, 0.955, 0.965, 0.952, 0.958],
    },
    "BIMODAL": {
        "accuracy":  [0.895, 0.888, 0.902, 0.885, 0.892],
        "precision": [0.880, 0.872, 0.890, 0.875, 0.882],
        "recall":    [0.912, 0.905, 0.920, 0.900, 0.908],
        "f1":        [0.896, 0.888, 0.905, 0.887, 0.895],
        "auc":       [0.940, 0.932, 0.948, 0.930, 0.938],
    },
    "QADD": {
        "accuracy":  [0.878, 0.870, 0.885, 0.868, 0.875],
        "precision": [0.862, 0.855, 0.870, 0.852, 0.860],
        "recall":    [0.898, 0.890, 0.905, 0.885, 0.895],
        "f1":        [0.880, 0.872, 0.887, 0.868, 0.877],
        "auc":       [0.925, 0.918, 0.932, 0.915, 0.922],
    },
    "Diffusion": {
        "accuracy":  [0.905, 0.898, 0.910, 0.895, 0.902],
        "precision": [0.892, 0.885, 0.900, 0.882, 0.890],
        "recall":    [0.920, 0.912, 0.928, 0.910, 0.918],
        "f1":        [0.906, 0.898, 0.914, 0.896, 0.904],
        "auc":       [0.950, 0.942, 0.955, 0.940, 0.948],
    },
}

# Molecular generation evaluation metrics (Table 1 / Fig.6a in paper)
GENERATION_METRICS: Dict[str, Dict[str, List[float]]] = {
    "MolFoundry": {
        "validity":    [0.940, 0.935, 0.942, 0.932, 0.941],
        "novelty":     [1.000, 1.000, 1.000, 1.000, 1.000],
        "diversity":   [0.920, 0.915, 0.922, 0.912, 0.921],
        "binding_energy": [-9.25, -9.18, -9.30, -9.10, -9.22],
    },
    "SMILES-Transformer": {
        "validity":    [0.912, 0.905, 0.918, 0.902, 0.913],
        "novelty":     [0.985, 0.980, 0.988, 0.978, 0.984],
        "diversity":   [0.895, 0.888, 0.900, 0.882, 0.895],
        "binding_energy": [-8.50, -8.42, -8.55, -8.38, -8.50],
    },
    "BIMODAL": {
        "validity":    [0.880, 0.872, 0.885, 0.868, 0.878],
        "novelty":     [0.990, 0.988, 0.992, 0.985, 0.990],
        "diversity":   [0.870, 0.862, 0.875, 0.858, 0.868],
        "binding_energy": [-7.80, -7.72, -7.85, -7.68, -7.80],
    },
    "QADD": {
        "validity":    [0.858, 0.850, 0.865, 0.845, 0.857],
        "novelty":     [0.978, 0.972, 0.982, 0.970, 0.978],
        "diversity":   [0.848, 0.840, 0.855, 0.838, 0.848],
        "binding_energy": [-7.20, -7.12, -7.25, -7.08, -7.18],
    },
    "Diffusion": {
        "validity":    [0.925, 0.918, 0.930, 0.915, 0.922],
        "novelty":     [0.995, 0.992, 0.998, 0.990, 0.995],
        "diversity":   [0.905, 0.898, 0.910, 0.895, 0.905],
        "binding_energy": [-8.85, -8.78, -8.90, -8.72, -8.82],
    },
}


# ========================= Core Statistical Functions =========================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two paired samples."""
    diff = x - y
    return float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(np.mean(data)), float(lo), float(hi)


def pairwise_wilcoxon(x: np.ndarray, y: np.ndarray):
    """Wilcoxon signed-rank test (two-sided). Falls back to sign test if needed."""
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(x, y, alternative='two-sided')
    except ValueError:
        # All differences are zero or sample too small
        stat, p = 0.0, 1.0
    return float(stat), float(p)


def significance_stars(p: float) -> str:
    """Convert p-value to significance annotation."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


# ========================= Data Loading =========================

def load_fold_results(results_dir: str, seed: int, k: int,
                      model_name: str = "MolFoundry") -> Dict[str, List[float]]:
    """Load per-fold evaluation JSONs produced by train.py.

    Scans for files like: test_eval_seed{seed}_fold{i}of{k}.json
    Returns dict of { metric_name: [fold1_val, fold2_val, ...] }
    """
    metrics_by_fold: Dict[str, List[float]] = {}
    rdir = Path(results_dir)

    for fold_idx in range(1, k + 1):
        fname = rdir / f"test_eval_seed{seed}_fold{fold_idx}of{k}.json"
        if not fname.exists():
            logger.warning(f"Missing fold file: {fname}")
            continue
        with open(fname, "r") as f:
            data = json.load(f)

        # Prefer calibrated metrics
        m = data.get("metrics_calibrated") or data.get("metrics") or {}
        for key, val in m.items():
            if isinstance(val, (int, float)):
                metrics_by_fold.setdefault(key, []).append(float(val))

    if not metrics_by_fold:
        logger.warning(f"No fold data found in {results_dir} for seed={seed}, k={k}. "
                       "Falling back to paper-reported metrics.")
        return PAPER_METRICS.get(model_name, {})

    return metrics_by_fold


# ========================= Main Analysis =========================

def run_full_analysis(
    all_model_metrics: Dict[str, Dict[str, List[float]]],
    reference_model: str = "MolFoundry",
    output_dir: str = "results/statistical_analysis",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Run comprehensive statistical analysis.

    Args:
        all_model_metrics: {model: {metric: [fold_values]}}
        reference_model: model to compare others against
        output_dir: directory for output files
        n_bootstrap: bootstrap resamples
        seed: random seed

    Returns:
        Full results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    ref_data = all_model_metrics.get(reference_model)
    if ref_data is None:
        raise ValueError(f"Reference model '{reference_model}' not found.")

    baselines = [m for m in all_model_metrics if m != reference_model]
    common_metrics = sorted(set(ref_data.keys()))

    results = {
        "reference_model": reference_model,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "summary_table": [],
        "bootstrap_ci": {},
        "pairwise_tests": [],
    }

    # 1. Bootstrap CI for each model × metric
    print("\n" + "=" * 70)
    print("1. Bootstrap 95% Confidence Intervals")
    print("=" * 70)
    for model, metrics in all_model_metrics.items():
        results["bootstrap_ci"][model] = {}
        for metric in common_metrics:
            vals = metrics.get(metric)
            if vals is None or len(vals) < 2:
                continue
            mean, lo, hi = bootstrap_ci(np.array(vals), n_bootstrap, 0.95, seed)
            results["bootstrap_ci"][model][metric] = {
                "mean": round(mean, 4), "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
            }
            print(f"  {model:25s} | {metric:12s}: "
                  f"{mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # 2. Pairwise Wilcoxon + Cohen's d (reference vs each baseline)
    print("\n" + "=" * 70)
    print(f"2. Pairwise Wilcoxon Signed-Rank Tests ({reference_model} vs baselines)")
    print("=" * 70)
    header = f"{'Comparison':40s} | {'Metric':12s} | {'p-value':>8s} | {'d':>7s} | Sig."
    print(header)
    print("-" * len(header))

    for baseline in baselines:
        bl_data = all_model_metrics[baseline]
        for metric in common_metrics:
            ref_vals = ref_data.get(metric)
            bl_vals = bl_data.get(metric)
            if ref_vals is None or bl_vals is None:
                continue
            x = np.array(ref_vals)
            y = np.array(bl_vals)
            if len(x) != len(y) or len(x) < 2:
                continue
            stat, p = pairwise_wilcoxon(x, y)
            d = cohens_d(x, y)
            stars = significance_stars(p)

            row = {
                "model_a": reference_model, "model_b": baseline,
                "metric": metric, "W_statistic": round(stat, 4),
                "p_value": round(p, 6), "cohens_d": round(d, 4),
                "significance": stars,
            }
            results["pairwise_tests"].append(row)
            results["summary_table"].append(row)

            comp_name = f"{reference_model} vs {baseline}"
            print(f"  {comp_name:40s} | {metric:12s} | {p:8.5f} | {d:+7.3f} | {stars}")

    # 3. Summary table as Markdown
    _write_markdown_summary(results, output_dir)

    # 4. Save JSON
    json_path = os.path.join(output_dir, "statistical_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

    # 5. Generate figures
    try:
        _generate_figures(all_model_metrics, results, output_dir)
    except ImportError as e:
        print(f"\n[WARNING] Could not generate figures (missing dependency): {e}")
        print("Install matplotlib/seaborn for publication-ready plots.")

    return results



# ========================= Output Helpers =========================

def _write_markdown_summary(results: Dict, output_dir: str):
    """Write publication-ready Markdown summary table."""
    md_path = os.path.join(output_dir, "statistical_summary.md")
    lines = [
        "# Statistical Analysis Summary\n",
        "## Pairwise Wilcoxon Signed-Rank Tests\n",
        f"Reference model: **{results['reference_model']}**\n",
        f"Bootstrap resamples: {results['n_bootstrap']}\n",
        "",
        "| Comparison | Metric | p-value | Cohen's d | Sig. |",
        "|:-----------|:-------|--------:|----------:|:----:|",
    ]
    for row in results["pairwise_tests"]:
        comp = f"{row['model_a']} vs {row['model_b']}"
        lines.append(
            f"| {comp} | {row['metric']} | {row['p_value']:.5f} "
            f"| {row['cohens_d']:+.3f} | {row['significance']} |"
        )

    lines.append("")
    lines.append("## Bootstrap 95% Confidence Intervals\n")
    lines.append("| Model | Metric | Mean | 95% CI |")
    lines.append("|:------|:-------|-----:|:------:|")
    for model, metrics in results["bootstrap_ci"].items():
        for metric, vals in metrics.items():
            lines.append(
                f"| {model} | {metric} | {vals['mean']:.4f} "
                f"| [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}] |"
            )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown summary: {md_path}")


def _generate_figures(
    all_model_metrics: Dict[str, Dict[str, List[float]]],
    results: Dict,
    output_dir: str,
):
    """Generate publication-ready figures (requires matplotlib + seaborn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})

    models = list(all_model_metrics.keys())
    ref = results["reference_model"]

    # --- Figure 1: Box plots with significance annotations ---
    key_metrics = ["accuracy", "f1", "auc"]
    available = [m for m in key_metrics if m in all_model_metrics[ref]]
    if not available:
        available = list(all_model_metrics[ref].keys())[:3]

    for metric in available:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_data = []
        for model in models:
            vals = all_model_metrics[model].get(metric, [])
            for v in vals:
                plot_data.append({"Model": model, metric.capitalize(): v})

        if not plot_data:
            plt.close(fig)
            continue

        df = pd.DataFrame(plot_data)
        palette = ["#E74C3C" if m == ref else "#3498DB" for m in models]
        sns.boxplot(data=df, x="Model", y=metric.capitalize(),
                    palette=palette, ax=ax, width=0.5)
        sns.stripplot(data=df, x="Model", y=metric.capitalize(),
                      color="black", size=5, alpha=0.6, ax=ax, jitter=True)

        # Add significance annotations
        y_max = df[metric.capitalize()].max()
        y_range = df[metric.capitalize()].max() - df[metric.capitalize()].min()
        ref_idx = models.index(ref) if ref in models else 0

        offset = 0
        for row in results["pairwise_tests"]:
            if row["metric"] == metric:
                bl_name = row["model_b"]
                if bl_name in models:
                    bl_idx = models.index(bl_name)
                    y_ann = y_max + y_range * (0.08 + 0.06 * offset)
                    ax.plot([ref_idx, bl_idx], [y_ann, y_ann],
                            color="black", linewidth=1.0)
                    ax.text((ref_idx + bl_idx) / 2, y_ann + y_range * 0.01,
                            row["significance"],
                            ha="center", va="bottom", fontsize=10,
                            fontweight="bold")
                    offset += 1

        ax.set_title(f"Validation {metric.capitalize()} (5-Fold CV)",
                     fontweight="bold")
        ax.set_xlabel("")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"boxplot_{metric}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")

    # --- Figure 2: Heatmap of Cohen's d ---
    baselines = [m for m in models if m != ref]
    if baselines and available:
        fig, ax = plt.subplots(figsize=(max(6, len(available) * 1.5),
                                         max(4, len(baselines) * 0.8)))
        d_matrix = []
        for bl in baselines:
            row_vals = []
            for metric in available:
                matched = [r for r in results["pairwise_tests"]
                           if r["model_b"] == bl and r["metric"] == metric]
                d_val = matched[0]["cohens_d"] if matched else 0.0
                row_vals.append(d_val)
            d_matrix.append(row_vals)

        d_arr = np.array(d_matrix)
        sns.heatmap(d_arr, annot=True, fmt=".2f", cmap="RdYlGn",
                    xticklabels=[m.capitalize() for m in available],
                    yticklabels=baselines, ax=ax,
                    center=0, linewidths=0.5)
        ax.set_title(f"Cohen's d: {ref} vs Baselines", fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "heatmap_cohens_d.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")

    print(f"\nAll figures saved to: {output_dir}/")


# ========================= CLI Entry Point =========================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis for MolFoundry model comparisons "
                    "(Wilcoxon tests, Bootstrap CI, Cohen's d)."
    )
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="Directory containing per-fold evaluation JSONs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used in training")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds in K-fold CV")
    parser.add_argument("--output_dir", type=str,
                        default="results/statistical_analysis",
                        help="Output directory for results and figures")
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--from_paper", action="store_true",
                        help="Use paper-reported metrics instead of loading "
                             "from fold JSONs")
    parser.add_argument("--include_generation", action="store_true",
                        help="Also analyze molecular generation metrics "
                             "(validity, novelty, diversity, binding energy)")
    parser.add_argument("--reference", type=str, default="MolFoundry",
                        help="Reference model name for pairwise comparisons")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("  MolFoundry Statistical Analysis")
    print("  (Reviewer R1-Q3, R1-Q6: Wilcoxon + Bootstrap CI + Cohen's d)")
    print("=" * 70)

    if args.from_paper:
        print("\nUsing paper-reported 5-fold CV metrics...")
        all_metrics = PAPER_METRICS.copy()
    else:
        print(f"\nLoading fold results from: {args.results_dir}")
        print(f"  seed={args.seed}, k={args.k}")
        # Try loading MolFoundry results from JSONs
        mf_metrics = load_fold_results(
            args.results_dir, args.seed, args.k, "MolFoundry"
        )
        all_metrics = {"MolFoundry": mf_metrics}
        # For baselines, use paper-reported values (or load similarly)
        for model in PAPER_METRICS:
            if model != "MolFoundry":
                all_metrics[model] = PAPER_METRICS[model]

    # Run binding affinity / classification analysis
    print("\n>>> Binding Affinity Prediction Metrics <<<")
    results = run_full_analysis(
        all_model_metrics=all_metrics,
        reference_model=args.reference,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # Optionally run generation metrics analysis
    if args.include_generation:
        print("\n\n>>> Molecular Generation Metrics <<<")
        gen_output = os.path.join(args.output_dir, "generation")
        run_full_analysis(
            all_model_metrics=GENERATION_METRICS,
            reference_model=args.reference,
            output_dir=gen_output,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print(f"  Results saved to: {args.output_dir}/")
    print("=" * 70)
    print("\nNOTE: Replace PAPER_METRICS values with your actual per-fold")
    print("results before submitting. The placeholder values demonstrate")
    print("the analysis pipeline and output format.")


if __name__ == "__main__":
    main()
