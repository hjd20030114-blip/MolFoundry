# -*- coding: utf-8 -*-
"""
P-L 数据集（Protein-Ligand paired dataset）
- 遍历 data/P-L/ 下的结构：<year_range>/<pdbid>/*_{protein|pocket}.pdb, *_ligand.{sdf|mol2}
- 构建正样本 (真实 pocket-ligand 配对)
- 通过跨ID随机混配构建负样本 (错误配对)
- 特征：
  - 配体：RDKit Morgan 指纹 (nBits=2048, radius=2)
  - 口袋：PDB 解析的统计特征（原子数、C/N/O/S/P计数，xyz均值/标准差）

依赖：torch, numpy, rdkit
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


# 静默 RDKit 控制台输出（若可用）
if 'HAS_RDKIT' in globals() and HAS_RDKIT:
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')  # 如需仅保留错误，可改为 'rdApp.warning'
    except Exception:
        pass


def _find_pl_items(root: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for year_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for pdb_dir in sorted([p for p in year_dir.iterdir() if p.is_dir()]):
            pdbid = pdb_dir.name
            protein = None
            pocket = None
            ligand = None
            # 搜索文件
            for f in pdb_dir.iterdir():
                name = f.name.lower()
                if name.endswith('_protein.pdb'):
                    protein = str(f)
                elif name.endswith('_pocket.pdb'):
                    pocket = str(f)
                elif name.endswith('_ligand.sdf') or name.endswith('_ligand.mol2'):
                    ligand = str(f)
            if pocket and ligand:  # 至少需要口袋和配体
                items.append({
                    'id': pdbid,
                    'protein': protein or '',
                    'pocket': pocket,
                    'ligand': ligand,
                })
    return items


def _split_ids(ids: List[str], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> Dict[str, List[str]]:
    ids = list(sorted(set(ids)))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    # 针对小数据集的健壮划分：保证在 n>=2 时 val 至少1个，n>=3 时 test 至少1个
    if n == 0:
        return {'train': [], 'val': [], 'test': []}
    if n == 1:
        return {'train': ids, 'val': [], 'test': []}
    if n == 2:
        return {'train': ids[:1], 'val': ids[1:], 'test': []}
    # n >= 3
    n_val = max(int(n * ratios[1]), 1)
    n_test = max(int(n * ratios[2]), 1)
    # 确保至少留 1 个给训练集
    if n_val + n_test >= n:
        # 优先减少 test，再减少 val，但各至少保留 1 个
        overflow = n_val + n_test - (n - 1)
        while overflow > 0 and n_test > 1:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 1:
            n_val -= 1
            overflow -= 1
    n_train = n - n_val - n_test
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return {'train': train_ids, 'val': val_ids, 'test': test_ids}


_POCKET_CACHE: dict = {}
_LIGAND_CACHE: dict = {}


def _pocket_features_from_pdb(pdb_file: str) -> np.ndarray:
    if pdb_file in _POCKET_CACHE:
        return _POCKET_CACHE[pdb_file]
    # 解析 PDB 行，统计元素计数与空间统计
    if not pdb_file or not os.path.exists(pdb_file):
        return np.zeros(12, dtype=np.float32)
    xs, ys, zs = [], [], []
    cnt = 0
    elem_counts = {e: 0 for e in ['C', 'N', 'O', 'S', 'P']}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                cnt += 1
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    xs.append(x); ys.append(y); zs.append(z)
                except Exception:
                    pass
                # 元素列（尽量从列76-78，否则取原子名首字母）
                elem = line[76:78].strip()
                if not elem:
                    name = line[12:16].strip()
                    elem = name[0]
                elem = elem.upper()
                if elem in elem_counts:
                    elem_counts[elem] += 1
    if cnt == 0:
        return np.zeros(12, dtype=np.float32)
    xs = np.array(xs, dtype=np.float32); ys = np.array(ys, dtype=np.float32); zs = np.array(zs, dtype=np.float32)
    means = np.array([xs.mean(), ys.mean(), zs.mean()], dtype=np.float32)
    stds = np.array([xs.std() + 1e-6, ys.std() + 1e-6, zs.std() + 1e-6], dtype=np.float32)
    vec = np.concatenate([
        np.array([cnt], dtype=np.float32),
        np.array([elem_counts['C'], elem_counts['N'], elem_counts['O'], elem_counts['S'], elem_counts['P']], dtype=np.float32),
        means, stds
    ], axis=0)
    _POCKET_CACHE[pdb_file] = vec
    return vec  # 1 + 5 + 3 + 3 = 12


def _ligand_fp_from_file(lig_file: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    if lig_file in _LIGAND_CACHE:
        return _LIGAND_CACHE[lig_file]
    if not HAS_RDKIT or not lig_file or not os.path.exists(lig_file):
        return np.zeros(n_bits, dtype=np.float32)
    mol = None
    try:
        if lig_file.lower().endswith('.sdf'):
            suppl = Chem.SDMolSupplier(lig_file, removeHs=False)
            if len(suppl) > 0:
                mol = suppl[0]
        elif lig_file.lower().endswith('.mol2'):
            mol = Chem.MolFromMol2File(lig_file, removeHs=False, sanitize=True)
    except Exception:
        mol = None
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)  # type: ignore
        vec = arr.astype(np.float32)
        _LIGAND_CACHE[lig_file] = vec
        return vec
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def _compute_pocket_stats(pocket_files: List[str]) -> Dict[str, List[float]]:
    feats: List[np.ndarray] = []
    for pf in pocket_files:
        feats.append(_pocket_features_from_pdb(pf))
    if not feats:
        mean = np.zeros(12, dtype=np.float32)
        std = np.ones(12, dtype=np.float32)
    else:
        arr = np.stack(feats, axis=0)
        mean = arr.mean(axis=0).astype(np.float32)
        std = (arr.std(axis=0).astype(np.float32) + 1e-6)
    return {'mean': mean.tolist(), 'std': std.tolist()}


class PLPairDataset(Dataset):
    def __init__(self,
                 root_dir: str = 'data/P-L',
                 split: str = 'train',
                 splits_file: Optional[str] = None,
                 negative_ratio: int = 1,
                 seed: int = 42,
                 normalize_pocket: bool = True,
                 feature_stats_file: Optional[str] = None,
                 min_fp_bits_on: int = 0):
        self.root = Path(root_dir)
        self.negative_ratio = max(0, int(negative_ratio))
        rng = random.Random(seed)

        # 建立索引
        items = _find_pl_items(self.root)
        arr = np.array(items)
        by_id: Dict[str, Dict[str, str]] = {it['id']: it for it in arr}
        self.ids = list(by_id.keys())
        if not self.ids:
            raise RuntimeError(f"未在 {self.root} 下找到任何 P-L 条目")
        if splits_file and os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
        else:
            splits = _split_ids(self.ids, seed=seed)
            # 默认将划分保存到 results/pl_splits.json
            out_dir = Path('results')
            out_dir.mkdir(exist_ok=True)
            with open(out_dir / 'pl_splits.json', 'w') as f:
                json.dump(splits, f, indent=2)
        split_ids = set(splits.get(split, []))
        self.items = [by_id[_id] for _id in by_id if _id in split_ids]

        # 训练集统计并保存/其他split加载：用于口袋特征标准化
        self.normalize_pocket = bool(normalize_pocket)
        self.feature_stats_file = feature_stats_file or str(Path('results') / 'pl_feature_stats.json')
        self.pocket_mean: Optional[np.ndarray] = None
        self.pocket_std: Optional[np.ndarray] = None
        if self.normalize_pocket:
            stats: Optional[Dict[str, List[float]]] = None
            if split == 'train':
                pocket_files = [it['pocket'] for it in self.items]
                stats = _compute_pocket_stats(pocket_files)
                out_path = Path(self.feature_stats_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(stats, f, indent=2)
            else:
                if self.feature_stats_file and os.path.exists(self.feature_stats_file):
                    with open(self.feature_stats_file, 'r') as f:
                        stats = json.load(f)
                else:
                    pocket_files = [it['pocket'] for it in self.items]
                    stats = _compute_pocket_stats(pocket_files)
            if stats is not None:
                self.pocket_mean = np.array(stats['mean'], dtype=np.float32)
                self.pocket_std = np.maximum(np.array(stats['std'], dtype=np.float32), 1e-6)

        # 构建正样本
        self.pos_pairs = [(it['pocket'], it['ligand']) for it in self.items]

        # 构建负样本（随机跨ID配对）
        self.neg_pairs: List[Tuple[str, str]] = []
        if self.negative_ratio > 0 and len(self.items) > 1:
            ligands = [it['ligand'] for it in self.items]
            pockets = [it['pocket'] for it in self.items]
            for i in range(len(self.items)):
                for _ in range(self.negative_ratio):
                    # 随机选择不同索引的pocket
                    j = i
                    tries = 0
                    while j == i and tries < 10:
                        j = rng.randrange(0, len(self.items))
                        tries += 1
                    if j == i:
                        continue
                    self.neg_pairs.append((pockets[j], ligands[i]))

        # 合并为样本列表 (label=1 正, label=0 负)
        self.samples: List[Tuple[str, str, int]] = []
        for p, l in self.pos_pairs:
            self.samples.append((p, l, 1))
        for p, l in self.neg_pairs:
            self.samples.append((p, l, 0))
        rng.shuffle(self.samples)

        # 可选：过滤掉配体指纹非信息样本
        self.min_fp_bits_on = int(min_fp_bits_on)
        if self.min_fp_bits_on > 0:
            filtered: List[Tuple[str, str, int]] = []
            for p, l, lab in self.samples:
                fp_vec = _ligand_fp_from_file(l)
                if fp_vec.sum() >= self.min_fp_bits_on:
                    filtered.append((p, l, lab))
            if filtered:
                self.samples = filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pocket_file, ligand_file, label = self.samples[idx]
        x_pocket = _pocket_features_from_pdb(pocket_file)
        # 标准化口袋特征
        if self.normalize_pocket and (self.pocket_mean is not None) and (self.pocket_std is not None):
            x_pocket = (x_pocket - self.pocket_mean) / self.pocket_std
        x_lig = _ligand_fp_from_file(ligand_file)
        x = np.concatenate([x_lig, x_pocket], axis=0).astype(np.float32)
        return {
            'x': torch.from_numpy(x),
            'y': torch.tensor([label], dtype=torch.float32),
            'pocket_file': pocket_file,
            'ligand_file': ligand_file
        }
