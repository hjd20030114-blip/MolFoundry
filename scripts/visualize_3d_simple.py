#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成综合3D可视化报告（简单入口）
从最新一次运行合并数据，生成/更新3D可视化HTML
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
    run_dir = result_manager.get_current_run_dir()
    if not run_dir:
        # 如果当前没有运行，尝试使用最近一次运行目录
        runs = result_manager.list_all_runs()
        if runs:
            last = sorted(runs, key=lambda r: r.get('start_time', ''))[-1]
            run_dir = Path(last['directory'])
        else:
            print('❌ 没有找到运行目录，请先执行完整流程')
            sys.exit(1)

    # 聚合数据
    lig_csv = result_manager.get_latest_ligands_file()
    dock_csv = result_manager.get_latest_docking_file()
    admet_csv = result_manager.get_latest_admet_file()

    lig_df = pd.read_csv(lig_csv) if lig_csv and lig_csv.exists() else pd.DataFrame()
    dock_df = pd.read_csv(dock_csv) if dock_csv and dock_csv.exists() else pd.DataFrame()
    admet_df = pd.read_csv(admet_csv) if admet_csv and admet_csv.exists() else pd.DataFrame()

    # 合并
    results_data = []
    dock_map = {}
    if not dock_df.empty and 'smiles' in dock_df.columns:
        for _, row in dock_df.iterrows():
            dock_map[row['smiles']] = row.get('binding_affinity')

    for _, l in lig_df.iterrows():
        smiles = l.get('smiles')
        if not smiles:
            continue
        entry = {
            'compound_id': l.get('compound_id', str(smiles)[:12]),
            'smiles': smiles,
            'binding_affinity': float(dock_map.get(smiles)) if smiles in dock_map and pd.notna(dock_map.get(smiles)) else None
        }
        if not admet_df.empty:
            match = admet_df[admet_df['smiles'] == smiles]
            if not match.empty:
                m = match.iloc[0].to_dict()
                for k in ['molecular_weight', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'tpsa']:
                    if k in m:
                        entry[k] = m[k]
        results_data.append(entry)

    viz = Visualizer3D(output_dir=str(result_manager.get_3d_viz_dir()))
    report_path = viz.generate_comprehensive_report(
        results_data=results_data,
        pdb_file=str(Path(DATA_DIR) / '1p65.pdb')
    )

    if report_path:
        print(f"✅ 综合3D可视化报告: {report_path}")
    else:
        print('⚠️ 生成报告失败（py3Dmol/plotly/RDKit 可能缺失）')


if __name__ == '__main__':
    main()
