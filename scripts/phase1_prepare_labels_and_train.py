# -*- coding: utf-8 -*-
"""
Phase 1 准备标签并训练 EquivariantGNN（方案A：用 Vina 对接分数作为标签）
使用方式：
  python3 HJD/scripts/phase1_prepare_labels_and_train.py \
      --receptor data/1p65.pdbqt \
      --vina_bin "$HOME/miniforge3/envs/vina/bin/vina" \
      --max_samples 300 \
      --epochs 20

说明：
- 仅使用现有模型：deep_learning/models/equivariant_gnn.py
- 从 data/P-L/ 读取 *_ligand.sdf|mol2，提取SMILES，小样本对接打分作为 binding_affinity
- 启动 deep_learning_pipeline.py 的 Phase 1 训练
"""
from __future__ import annotations

import os
import glob
import time
import random
from pathlib import Path
import argparse
import pandas as pd

from rdkit import Chem

# 项目根加入路径
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.docking_engine import DockingEngine  # type: ignore
from deep_learning_pipeline import DeepLearningPipeline  # type: ignore


def collect_smiles_from_pl(pl_root: str, max_samples: int = 300):
    lig_files = glob.glob(os.path.join(pl_root, '**', '*_ligand.sdf'), recursive=True) + \
                glob.glob(os.path.join(pl_root, '**', '*_ligand.mol2'), recursive=True)
    random.seed(42)
    if max_samples and len(lig_files) > max_samples:
        lig_files = random.sample(lig_files, max_samples)

    ligands = []
    for lf in lig_files:
        mol = None
        try:
            if lf.lower().endswith('.sdf'):
                suppl = Chem.SDMolSupplier(lf, removeHs=False)
                mol = suppl[0] if len(suppl) > 0 else None
            else:
                mol = Chem.MolFromMol2File(lf, removeHs=False, sanitize=True)
        except Exception:
            mol = None
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            ligands.append({'smiles': smi})
        except Exception:
            continue
    return ligands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receptor', type=str, default='data/1p65.pdbqt')
    ap.add_argument('--vina_bin', type=str, required=True, help='Vina 可执行文件绝对路径')
    ap.add_argument('--pl_root', type=str, default='data/P-L')
    ap.add_argument('--max_samples', type=int, default=300)
    ap.add_argument('--epochs', type=int, default=10000)
    ap.add_argument('--protein_pdb', type=str, default='data/1p65.pdb', help='Phase1 训练中使用的 PDB 坐标文件')
    args = ap.parse_args()

    # 1) 收集SMILES
    ligands = collect_smiles_from_pl(args.pl_root, args.max_samples)
    print(f'[Phase1] Collected ligands with SMILES: {len(ligands)}')
    if not ligands:
        raise SystemExit('No ligands collected from data/P-L')

    # 2) 批量对接打分
    engine = DockingEngine()
    engine.vina_exe = os.path.expanduser(args.vina_bin)
    run_dir = Path('results')/f'docking_run_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    res_df = engine.batch_docking(ligands, receptor_file=args.receptor, output_dir=str(run_dir))
    if res_df is None or res_df.empty:
        raise SystemExit('Docking produced no results. Check vina/meeko installation and receptor path.')
    res_file = run_dir/'docking_results.csv'
    print(f'[Phase1] Docking results saved to: {res_file}')

    # 3) 构建 Phase 1 训练数据
    init_df = res_df[['smiles','binding_affinity']].copy()
    init_df = init_df.dropna().reset_index(drop=True)
    init_data = init_df.to_dict(orient='records')
    print(f'[Phase1] Training samples prepared: {len(init_data)}')

    # 4) 启动 Phase 1 训练
    pl = DeepLearningPipeline(None)
    pl.training_config.num_epochs = int(args.epochs)
    pl.training_config.batch_size = 32
    pl.training_config.num_workers = 4
    model_path = pl.phase_1_equivariant_gnn(args.protein_pdb, init_data)
    print('[Phase1] Done. Model saved at:', model_path)


if __name__ == '__main__':
    main()
