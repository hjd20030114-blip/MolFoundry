#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRRSV抑制剂设计平台 - 一键完整工作流程
步骤：配体生成 -> 分子对接 -> ADMET分析 -> 3D可视化报告
输出保存在 ResultManager 管理的当次运行目录下
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

# 保证脚本从HJD目录运行
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
    sys.path.insert(0, str(CURRENT_DIR / 'scripts'))

from scripts.result_manager import result_manager
from scripts.ligand_generator import LigandGenerator
from scripts.molecular_docking import MolecularDocking
from scripts.admet_analyzer import ADMETAnalyzer
from scripts.visualization_3d import Visualizer3D
from scripts.config import DATA_DIR


def main():
    print("🧬 启动完整工作流程…")

    # 1) 创建运行目录
    run_dir = result_manager.create_new_run_directory()
    print(f"📁 运行目录: {run_dir}")

    # 2) 生成配体
    print("🧪 生成配体…")
    generator = LigandGenerator()
    ligands = generator.generate_optimized_ligands()
    if not ligands:
        print("❌ 配体生成失败，无可用配体")
        sys.exit(1)

    # 保存配体
    lig_dir = result_manager.get_ligands_dir()
    lig_dir.mkdir(parents=True, exist_ok=True)
    ligands_df = pd.DataFrame(ligands)
    ligands_csv = lig_dir / "generated_ligands.csv"
    ligands_df.to_csv(ligands_csv, index=False)
    # 生成SMILES列表
    smiles_file = lig_dir / "ligands.smi"
    with open(smiles_file, "w", encoding="utf-8") as f:
        for l in ligands:
            smi = l.get("smiles", "")
            if smi:
                f.write(f"{smi}\t{smi}\n")
    result_manager.update_step_completed("ligand_generation", [str(ligands_csv), str(smiles_file)])
    print(f"✅ 已生成 {len(ligands)} 个配体 -> {ligands_csv}")

    # 3) 分子对接
    print("🔬 分子对接…")
    protein_pdb = str(Path(DATA_DIR) / "1p65.pdb")
    docker = MolecularDocking()
    smiles_list = [l["smiles"] for l in ligands if l.get("smiles")]
    docking_df = docker.dock_multiple_ligands(protein_pdb, smiles_list)

    dock_dir = result_manager.get_docking_dir()
    dock_dir.mkdir(parents=True, exist_ok=True)
    docking_csv = dock_dir / "docking_results.csv"
    docking_df.to_csv(docking_csv, index=False)
    result_manager.update_step_completed("docking", [str(docking_csv)])
    print(f"✅ 对接完成: {len(docking_df)} 个结果 -> {docking_csv}")

    # 4) ADMET分析
    print("📈 ADMET分析…")
    analyzer = ADMETAnalyzer()
    admet_df = analyzer.batch_admet_analysis([{**l} for l in ligands if l.get("smiles")])

    admet_dir = result_manager.get_admet_dir()
    admet_dir.mkdir(parents=True, exist_ok=True)
    admet_csv = admet_dir / "admet_results.csv"
    if not admet_df.empty:
        admet_df.to_csv(admet_csv, index=False)
        result_manager.update_step_completed("admet", [str(admet_csv)])
        print(f"✅ ADMET完成: {len(admet_df)} 条记录 -> {admet_csv}")
    else:
        print("⚠️ 无ADMET结果（RDKit未安装或无有效分子）")

    # 5) 3D综合可视化报告
    print("🌐 生成3D综合可视化报告…")
    # 合并结果数据（若有对接分数则带上）
    results_data = []
    dock_map = {}
    if not docking_df.empty and "smiles" in docking_df.columns:
        for _, row in docking_df.iterrows():
            dock_map[row["smiles"]] = row.get("best_affinity")

    for l in ligands:
        smiles = l.get("smiles")
        if not smiles:
            continue
        entry = {
            "compound_id": l.get("compound_id", smiles[:12]),
            "smiles": smiles,
        }
        if smiles in dock_map and dock_map[smiles] is not None:
            entry["binding_affinity"] = float(dock_map[smiles])
        # 合并部分ADMET（若存在）
        if not admet_df.empty:
            match = admet_df[admet_df["smiles"] == smiles]
            if not match.empty:
                m = match.iloc[0].to_dict()
                for k in [
                    "molecular_weight", "logp", "hbd", "hba",
                    "rotatable_bonds", "tpsa"
                ]:
                    if k in m:
                        entry[k] = m[k]
        results_data.append(entry)

    viz = Visualizer3D(output_dir=str(result_manager.get_3d_viz_dir()))
    report_path = viz.generate_comprehensive_report(
        results_data=results_data,
        pdb_file=str(Path(DATA_DIR) / "1p65.pdb")
    )

    reports_dir = result_manager.get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    # 将主报告复制到reports目录以便查找
    if report_path:
        rp = Path(report_path)
        target = reports_dir / rp.name
        try:
            import shutil
            shutil.copy2(rp, target)
        except Exception:
            pass
        result_manager.update_step_completed("visualization_3d", [str(rp)])
        print(f"✅ 3D报告: {rp}")
    else:
        print("⚠️ 3D报告生成失败（缺少py3Dmol/plotly或RDKit）")

    # 汇总并结束
    summary = result_manager.get_run_summary()
    print("\n📋 运行摘要:")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    result_manager.finalize_run()
    print("\n🎉 完整工作流程完成！")


if __name__ == "__main__":
    main()
