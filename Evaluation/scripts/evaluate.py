import os
import csv
import json
from typing import Dict, List, Set

from metrics import read_smiles_from_csv, score_all, canonical_set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_DIR = os.path.join(ROOT, "Evaluation")
IN_DIR = os.path.join(EVAL_DIR, "inputs")
REF_PATH = os.path.join(EVAL_DIR, "reference", "reference_smiles.csv")
OUT_DIR = os.path.join(EVAL_DIR, "outputs")
OUR_MODEL_DEFAULT = os.path.join(ROOT, "deep_learning_results", "phase2_generated_molecules.csv")

MODELS = [
    ("our_model", OUR_MODEL_DEFAULT),
    ("diffusion", os.path.join(IN_DIR, "diffusion.csv")),
    ("egnn", os.path.join(IN_DIR, "egnn.csv")),
    ("transformer", os.path.join(IN_DIR, "transformer.csv")),
    ("bimodal", os.path.join(IN_DIR, "bimodal.csv")),
    ("organ", os.path.join(IN_DIR, "organ.csv")),
    ("qadd", os.path.join(IN_DIR, "qadd.csv")),
    ("mars", os.path.join(IN_DIR, "mars.csv")),
]

# Optional docking score readers
from pathlib import Path
from glob import glob

def _canonical(s: str) -> str:
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else ""

def read_docking_scores_csv(path: str, smiles_col: str = "smiles", affinity_col: str = "binding_affinity") -> Dict[str, float]:
    mp: Dict[str, float] = {}
    if not os.path.exists(path):
        return mp
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            smi = (row.get(smiles_col, "") or "").strip()
            aff = row.get(affinity_col, None)
            if not smi or aff is None:
                continue
            try:
                val = float(aff)
            except Exception:
                continue
            c = _canonical(smi)
            if c:
                # Keep the best (most negative) if multiple entries exist
                if c in mp:
                    mp[c] = min(mp[c], val)
                else:
                    mp[c] = val
    return mp

def find_latest_our_model_docking_csv(root: str) -> str:
    # prefer explicit docking_run_* under results
    cand = sorted(glob(os.path.join(root, "HJD", "results", "docking_run_*", "docking_results.csv")), key=os.path.getmtime, reverse=True)
    if cand:
        return cand[0]
    # fallback: results/run_*/docking/docking_results.csv
    cand = sorted(glob(os.path.join(root, "HJD", "results", "run_*", "docking", "docking_results.csv")), key=os.path.getmtime, reverse=True)
    return cand[0] if cand else ""

def load_model_docking_map(model_name: str) -> Dict[str, float]:
    # Priority 1: HJD/Evaluation/inputs/docking/<model_name>.csv
    in1 = os.path.join(IN_DIR, "docking", f"{model_name}.csv")
    if os.path.exists(in1):
        return read_docking_scores_csv(in1)
    # Priority 2: HJD/Evaluation/inputs/<model_name>_docking.csv
    in2 = os.path.join(IN_DIR, f"{model_name}_docking.csv")
    if os.path.exists(in2):
        return read_docking_scores_csv(in2)
    # Priority 3: for our_model, try latest results docking csv
    if model_name == "our_model":
        p = find_latest_our_model_docking_csv(ROOT)
        if p:
            return read_docking_scores_csv(p)
    return {}



def load_reference() -> Set[str]:
    with open(REF_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return canonical_set(row["smiles"].strip() for row in r if row["smiles"].strip())


def model_smiles(path: str) -> List[str]:
    # attempt to detect smiles column
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(',')
    header_lower = [h.lower() for h in header]
    if "smiles" in header_lower:
        return read_smiles_from_csv(path, smiles_col=header[header_lower.index("smiles")])
    # fallback: try first column
    return read_smiles_from_csv(path, smiles_col=header[0])


def rank_models(metrics: Dict[str, Dict[str, float]]) -> List[str]:
    # higher is better for all listed metrics
    keys = list(metrics.keys())
    indicators = ["Validity", "Novelty", "Uniq", "IntDiv", "QED"]
    # compute ranks per indicator (equal-importance, by average rank)
    ranks = {k: [] for k in keys}
    for ind in indicators:
        sorted_models = sorted(keys, key=lambda k: metrics[k][ind], reverse=True)
        for rank, name in enumerate(sorted_models, start=1):
            ranks[name].append(rank)
    # average rank
    avg_rank = {k: sum(v)/len(v) for k, v in ranks.items()}
    return [k for k, _ in sorted(avg_rank.items(), key=lambda x: x[1])]


def rank_models_weighted(metrics: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> List[str]:
    # higher is better for all listed metrics; use weighted sum of raw scores
    # normalize weights to sum to 1.0
    inds = ["Validity", "Novelty", "Uniq", "IntDiv", "QED"]
    s = sum(weights.get(k, 0.0) for k in inds)
    norm_w = {k: (weights.get(k, 0.0) / s if s > 0 else 0.0) for k in inds}
    def score(name: str) -> float:
        m = metrics[name]
        return sum(m[k] * norm_w.get(k, 0.0) for k in inds)
    return sorted(metrics.keys(), key=score, reverse=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = load_reference()

    # compute metrics
    metrics_per_model: Dict[str, Dict[str, float]] = {}
    affinity_available_for_all = True
    affinity_values_for_norm = []
    docking_maps: Dict[str, Dict[str, float]] = {}

    for name, path in MODELS:
        smiles = model_smiles(path)
        scores = score_all(smiles, ref)
        # optional: best binding affinity (magnitude; higher is better)
        dock_map = load_model_docking_map(name)
        docking_maps[name] = dock_map
        best_aff_mag: float = float('nan')
        if dock_map:
            # match by canonical smiles
            can_smiles = [c for c in (_canonical(s) for s in smiles) if c]
            matched = [dock_map[c] for c in can_smiles if c in dock_map]
            if matched:
                best = min(matched)  # most negative (best)
                best_aff_mag = -best  # magnitude; higher is better
                scores["BestAffinity"] = best_aff_mag
                affinity_values_for_norm.append(best_aff_mag)
            else:
                scores["BestAffinity"] = None
                affinity_available_for_all = False
        else:
            scores["BestAffinity"] = None
            affinity_available_for_all = False

        metrics_per_model[name] = scores
        print(name, scores)

    # rankings
    order_equal = rank_models(metrics_per_model)
    # Discovery-oriented weights (sum to 1.0)
    weights = {"Novelty": 0.55, "IntDiv": 0.20, "QED": 0.15, "Uniq": 0.07, "Validity": 0.03}
    order_weighted = rank_models_weighted(metrics_per_model, weights)

    # write json
    with open(os.path.join(OUT_DIR, "metrics_per_model.json"), 'w', encoding='utf-8') as wf:
        json.dump(metrics_per_model, wf, indent=2, ensure_ascii=False)

    # write summary csv (equal-rank order)
    fields = ["model", "Validity", "Novelty", "Uniq", "IntDiv", "QED"]
    with open(os.path.join(OUT_DIR, "metrics_summary.csv"), 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(fields)
        for name in order_equal:
            m = metrics_per_model[name]
            w.writerow([name, m["Validity"], m["Novelty"], m["Uniq"], m["IntDiv"], m["QED"]])

    # write rankings
    with open(os.path.join(OUT_DIR, "ranking.csv"), 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(["rank", "model"])
        for i, name in enumerate(order_equal, start=1):
            w.writerow([i, name])
    with open(os.path.join(OUT_DIR, "ranking_weighted.csv"), 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(["rank", "model"])
        for i, name in enumerate(order_weighted, start=1):
            w.writerow([i, name])

    # markdown report
    with open(os.path.join(OUT_DIR, "report.md"), 'w', encoding='utf-8') as wf:
        wf.write("# Evaluation Summary\n\n")
        wf.write("Models compared: our_model, diffusion, egnn, transformer, bimodal, organ, qadd, mars\n\n")
        wf.write("## Ranking (overall - equal importance)\n\n")
        for i, name in enumerate(order_equal, start=1):
            wf.write(f"{i}. {name}\n")
        wf.write("\n## Ranking (discovery-oriented weighted)\n\n")
        wf.write("Weights: Novelty=0.55, IntDiv=0.20, QED=0.15, Uniq=0.07, Validity=0.03\n\n")
        for i, name in enumerate(order_weighted, start=1):
            wf.write(f"{i}. {name}\n")
        wf.write("\n## Metrics per model (equal-rank order)\n\n")
        for name in order_equal:
            m = metrics_per_model[name]
            wf.write(f"- {name}: Validity={m['Validity']:.3f}, Novelty={m['Novelty']:.3f}, Uniq={m['Uniq']:.3f}, IntDiv={m['IntDiv']:.3f}, QED={m['QED']:.3f}\n")

    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

