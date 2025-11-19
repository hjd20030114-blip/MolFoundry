#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDock结果查看器
读取最近一次运行的对接结果，生成若干3D可视化并打印路径
"""

import sys
from pathlib import Path
import pandas as pd

CURR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURR))
sys.path.insert(0, str(CURR / 'scripts'))

from scripts.result_manager import result_manager
from scripts.config import DATA_DIR
from scripts.visualization_3d import Visualizer3D


def main():
    dock_csv = result_manager.get_latest_docking_file()
    if not dock_csv or not dock_csv.exists():
        # 尝试默认结果位置
        fallback = CURR / 'results' / 'docking_results.csv'
        if fallback.exists():
            dock_csv = fallback
        else:
            print('❌ 未找到任何对接结果。请先运行完整流程或对接步骤。')
            sys.exit(1)

    df = pd.read_csv(dock_csv)
    if df.empty:
        print('❌ 对接结果为空')
        sys.exit(1)

    # 选择前5个最佳分子
    if 'binding_affinity' in df.columns:
        df = df.sort_values('binding_affinity')
    top = df.head(5)

    viz = Visualizer3D()
    html_files = []
    for i, row in top.iterrows():
        smiles = row.get('smiles', '')
        cid = row.get('compound_id', f"ligand_{i+1}")
        if not smiles:
            continue
        html = viz.visualize_molecule_3d(str(smiles), title=f"{cid}")
        if html:
            html_files.append(html)

    # 生成复合物可视化（使用最佳分子）
    complex_file = ''
    protein_pdb = str(Path(DATA_DIR) / '1p65.pdb')
    if not top.empty:
        best_smiles = str(top.iloc[0].get('smiles', ''))
        if best_smiles:
            complex_file = viz.visualize_protein_ligand_complex(protein_pdb, best_smiles)

    print('✅ 生成的3D文件:')
    for f in html_files:
        print(f" - {f}")
    if complex_file:
        print(f" - {complex_file}  (蛋白质-配体复合物)")


if __name__ == '__main__':
    main()
