import os
import csv
from typing import List, Set
from rdkit import Chem

from metrics import canonical_set

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PL = os.path.join(ROOT, "data", "P-L")
OUT_DIR = os.path.join(ROOT, "Evaluation", "reference")
OUT_PATH = os.path.join(OUT_DIR, "reference_smiles.csv")


def collect_smiles_from_file(path: str) -> List[str]:
    smiles = []
    fn = os.path.basename(path).lower()
    try:
        if fn.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(path, removeHs=False)
            for mol in suppl:
                if mol is None:
                    continue
                s = Chem.MolToSmiles(mol, canonical=True)
                smiles.append(s)
        elif fn.endswith('.mol2'):
            mol = Chem.MolFromMol2File(path, sanitize=True)
            if mol is not None:
                smiles.append(Chem.MolToSmiles(mol, canonical=True))
        elif fn.endswith('.smi') or fn.endswith('.smiles') or fn.endswith('.csv'):
            # try simple CSV/SMI read
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    if ' ' in line or ',' in line:
                        # best effort: take first token before space/comma
                        token = line.split(',')[0].split()[0]
                        smiles.append(token)
                    else:
                        smiles.append(line)
    except Exception:
        pass
    return smiles


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_smiles = []
    for root, _, files in os.walk(DATA_PL):
        for f in files:
            if f.lower().endswith(('.sdf', '.mol2', '.smi', '.smiles', '.csv')):
                path = os.path.join(root, f)
                all_smiles.extend(collect_smiles_from_file(path))
    ref = list(canonical_set(all_smiles))
    ref.sort()
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(["smiles"])
        for s in ref:
            w.writerow([s])
    print(f"Wrote {len(ref)} canonical reference SMILES to {OUT_PATH}")


if __name__ == "__main__":
    main()

