#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小分子3D查看器
从最新运行或示例SMILES生成3D分子HTML并打印路径
"""

import sys
from pathlib import Path

# 路径设置
CURR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURR))
sys.path.insert(0, str(CURR / 'scripts'))

from scripts.result_manager import result_manager
from scripts.visualization_3d import Visualizer3D


DEFAULT_SMILES = "COc1ccc(CCN)cc1"  # 示例分子


def main():
    # 优先从最新配体文件读取一个SMILES
    smiles = None
    try:
        lig_csv = result_manager.get_latest_ligands_file()
        if lig_csv and lig_csv.exists():
            import pandas as pd
            df = pd.read_csv(lig_csv)
            if not df.empty and 'smiles' in df.columns:
                smiles = str(df.iloc[0]['smiles'])
    except Exception:
        pass

    if not smiles:
        smiles = DEFAULT_SMILES

    viz = Visualizer3D()
    html = viz.visualize_molecule_3d(smiles, title="Molecule")

    if html:
        print(f"✅ 分子3D结构文件: {html}")
    else:
        print("⚠️ 分子3D结构生成失败（请检查py3Dmol/RDKit依赖）")


if __name__ == "__main__":
    main()
