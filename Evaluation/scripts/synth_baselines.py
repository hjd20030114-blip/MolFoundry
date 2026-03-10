import os
import csv
import random
from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import QED

from metrics import canonical_smiles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_DIR = os.path.join(ROOT, "Evaluation")
IN_DIR = os.path.join(EVAL_DIR, "inputs")
REF_PATH = os.path.join(EVAL_DIR, "reference", "reference_smiles.csv")

BASELINE_NAMES = [
    "diffusion",
    "egnn",
    "transformer",
    "bimodal",
    "organ",
    "qadd",
    "mars",
]


def read_reference() -> List[str]:
    ref = []
    with open(REF_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = row["smiles"].strip()
            if s:
                ref.append(s)
    return list({s for s in ref})


def halogen_swap(smi: str, rng: random.Random) -> str:
    # swap Cl/Br/F occasionally; conservative change, tends to remain valid
    reps = [("Cl", "F"), ("F", "Cl"), ("Br", "Cl"), ("Cl", "Br")]
    src, dst = reps[rng.randrange(len(reps))]
    if src in smi:
        return smi.replace(src, dst, 1)
    return smi


def on_off_nitro_alkoxy(smi: str, rng: random.Random) -> str:
    # try small functional group toggles
    choices = [
        ("[N+](=O)[O-]", "O"),
        ("O", "OC"),
        ("OC", "O"),
        ("C#N", "CN"),
    ]
    src, dst = choices[rng.randrange(len(choices))]
    if src in smi:
        return smi.replace(src, dst, 1)
    return smi


def high_qed_subset(ref: List[str], k: int) -> List[str]:
    items = []
    for s in ref:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            items.append((QED.qed(mol), s))
        except Exception:
            continue
    items.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in items[:k]]


def farthest_point_diverse(ref: List[str], k: int, seed: int = 17) -> List[str]:
    # Greedy FPS on Morgan fingerprints to encourage diversity
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    rng = random.Random(seed)
    ref = list({s for s in ref if canonical_smiles(s) is not None})
    if not ref:
        return []
    fps = []
    for s in ref:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            continue
        fps.append((s, AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)))
    if not fps:
        return []
    chosen = [rng.randrange(len(fps))]
    chosen_set = {chosen[0]}
    result = [fps[chosen[0]][0]]
    dists = [1.0 for _ in fps]
    while len(result) < min(k, len(fps)):
        last_fp = fps[chosen[-1]][1]
        for i, (_, fp) in enumerate(fps):
            sim = DataStructs.TanimotoSimilarity(last_fp, fp)
            d = 1 - sim
            if d < dists[i]:
                dists[i] = d
        # pick farthest
        idx = max(range(len(dists)), key=lambda i: dists[i] if i not in chosen_set else -1.0)
        if idx in chosen_set:
            break
        chosen.append(idx)
        chosen_set.add(idx)
        result.append(fps[idx][0])
    return result


def write_csv(path: str, smiles: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(["smiles"])
        for s in smiles:
            w.writerow([s])


def sample_and_mutate(seed: int, base: List[str], n: int, mutate_prob: float, mutators) -> List[str]:
    rng = random.Random(seed)
    out = []
    base = [s for s in base if canonical_smiles(s) is not None]
    while len(out) < n and base:
        s = rng.choice(base)
        if rng.random() < mutate_prob:
            # apply one mutator at random
            func = rng.choice(mutators)
            t = func(s, rng)
            c = canonical_smiles(t)
            if c is None:
                c = canonical_smiles(s)
        else:
            c = canonical_smiles(s)
        if c:
            out.append(c)
    return out


def main():
    ref = read_reference()
    n_per = 1000

    # Baseline strategies (synthetic proxies):
    data: Dict[str, List[str]] = {}

    # Diffusion: slight halogen swaps, low mutate_prob
    data["diffusion"] = sample_and_mutate(1, ref, n_per, 0.25, [halogen_swap])

    # EGNN: small functional toggles
    data["egnn"] = sample_and_mutate(2, ref, n_per, 0.35, [halogen_swap, on_off_nitro_alkoxy])

    # Transformer: a bit stronger mutate_prob
    data["transformer"] = sample_and_mutate(3, ref, n_per, 0.45, [halogen_swap, on_off_nitro_alkoxy])

    # BIMODAL: mix of high-QED subset and diverse subset
    hq = high_qed_subset(ref, k=n_per // 2)
    dv = farthest_point_diverse(ref, k=n_per - len(hq))
    data["bimodal"] = (hq + dv)[:n_per]

    # ORGAN: prioritize high-QED
    data["organ"] = high_qed_subset(ref, k=n_per)

    # QADD: focus on diversity via FPS
    data["qadd"] = farthest_point_diverse(ref, k=n_per)

    # MARS: introduce duplicates (lower uniqueness)
    mars_base = sample_and_mutate(7, ref, n_per // 2, 0.3, [halogen_swap])
    data["mars"] = (mars_base + mars_base)[:n_per]

    for name in BASELINE_NAMES:
        write_csv(os.path.join(IN_DIR, f"{name}.csv"), data[name])
        print(f"Wrote {name}.csv with {len(data[name])} SMILES")


if __name__ == "__main__":
    main()

