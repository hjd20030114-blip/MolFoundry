import csv
import random
from typing import List, Tuple, Dict, Optional, Set

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit import DataStructs


def canonical_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def read_smiles_from_csv(path: str, smiles_col: str = "smiles") -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if smiles_col not in reader.fieldnames:
            # fallback: try first column
            smiles_col = reader.fieldnames[0]
        for row in reader:
            s = row.get(smiles_col, "").strip()
            if s:
                out.append(s)
    return out


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048) -> Optional[ExplicitBitVect]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    except Exception:
        return None


def compute_validity(smiles_list: List[str]) -> float:
    if not smiles_list:
        return 0.0
    valid = 0
    for s in smiles_list:
        if canonical_smiles(s) is not None:
            valid += 1
    return valid / len(smiles_list)


def compute_uniqueness(smiles_list: List[str]) -> float:
    if not smiles_list:
        return 0.0
    can = [c for c in (canonical_smiles(s) for s in smiles_list) if c is not None]
    if not can:
        return 0.0
    return len(set(can)) / len(can)


def compute_qed(smiles_list: List[str]) -> float:
    vals = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            vals.append(QED.qed(mol))
        except Exception:
            continue
    return sum(vals) / len(vals) if vals else 0.0


def compute_intdiv(smiles_list: List[str], sample_pairs: int = 2000, seed: int = 13) -> float:
    # Internal diversity: 1 - mean tanimoto among random pairs
    can = list({c for c in (canonical_smiles(s) for s in smiles_list) if c is not None})
    if len(can) < 2:
        return 0.0
    fps = [morgan_fp(s) for s in can]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return 0.0
    rng = random.Random(seed)
    n = len(fps)
    pairs = min(sample_pairs, n * (n - 1) // 2)
    acc = 0.0
    cnt = 0
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        acc += sim
        cnt += 1
    mean_sim = (acc / cnt) if cnt else 1.0
    return max(0.0, 1.0 - mean_sim)


def compute_novelty(smiles_list: List[str], reference_set: Set[str]) -> float:
    # Both inputs should be canonical for a strict comparison
    can = [c for c in (canonical_smiles(s) for s in smiles_list) if c is not None]
    if not can:
        return 0.0
    novel = [s for s in can if s not in reference_set]
    return len(novel) / len(can)


def score_all(smiles_list: List[str], reference_set: Set[str]) -> Dict[str, float]:
    return {
        "Validity": compute_validity(smiles_list),
        "Uniq": compute_uniqueness(smiles_list),
        "Novelty": compute_novelty(smiles_list, reference_set),
        "IntDiv": compute_intdiv(smiles_list),
        "QED": compute_qed(smiles_list),
    }


def canonical_set(smiles_iter) -> Set[str]:
    s = set()
    for smi in smiles_iter:
        c = canonical_smiles(smi)
        if c is not None:
            s.add(c)
    return s

