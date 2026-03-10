#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：基于 docking 结果筛选分子，执行 DFT 与 MD，统一输出到
results/docking_run_20250920_120357/eval_run_时间戳/ 目录下。

默认方案：
- 分子筛选：binding_affinity ≤ -6.0（若不足10个，则按分数升序取Top-10）
- DFT：优先 Psi4（wB97X-D/def2-SVP 的默认设置由 dft_utils 内部管理），不可用则降级 xTB
- MD：尝试 TIP3P 水盒（若 openmmforcefields 可用并能为小分子注册SMIRNOFF模板）；否则降级为真空MD

依赖：仅使用标准库 + dft_utils.py + md_utils.py。
不会强制引入 pandas，以避免 numpy/pandas 二进制不兼容问题。
"""
from __future__ import annotations
import os
import sys
import csv
import json
import time
import argparse
from typing import List, Dict, Any, Tuple

# 与当前脚本同目录的工具
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dft_utils import compute_dft_for_smiles  # type: ignore
from md_utils import run_md_for_smiles  # type: ignore

# 模型名称（从 deep_learning 读取，若失败则回退为 MolFoundry）
try:
    import deep_learning as _dl  # type: ignore
    MODEL_NAME = getattr(_dl, "__model_name__", "MolFoundry")
except Exception:
    MODEL_NAME = "MolFoundry"

# 项目内默认的 docking 结果位置（可通过 --docking-csv 覆盖）
DEFAULT_DOCKING_CSV = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",  # outputs/
        "..",  # Evaluation/
        "results",
        "docking_run_20250920_120357",
        "docking_results.csv",
    )
)
DEFAULT_RESULTS_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "results",
        "docking_run_20250920_120357",
    )
)


def read_docking_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["binding_affinity"] = float(r.get("binding_affinity", "nan"))
            except Exception:
                continue
            rows.append(r)
    return rows


def select_molecules(rows: List[Dict[str, Any]], threshold: float, top_n: int) -> List[Dict[str, Any]]:
    valid = [r for r in rows if isinstance(r.get("binding_affinity"), float)]
    valid.sort(key=lambda x: x["binding_affinity"])  # 越小越优
    sel = [r for r in valid if r["binding_affinity"] <= threshold]
    if len(sel) < min(top_n, 10):
        sel = valid[:top_n]
    return sel


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        # 写空文件头
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFT + MD on selected ligands")
    parser.add_argument("--docking-csv", default=DEFAULT_DOCKING_CSV, help="path to docking_results.csv")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT, help="root dir to write eval_run_*")
    parser.add_argument("--threshold", type=float, default=-6.0, help="affinity threshold (kcal/mol)")
    parser.add_argument("--top-n", type=int, default=10, help="top-N if threshold not enough")
    parser.add_argument("--md-steps", type=int, default=2500000, help="MD steps (2 fs -> ~5 ns for 2.5M)")
    parser.add_argument("--md-platform", default=None, help="preferred OpenMM platform: CUDA/OpenCL/CPU")
    args = parser.parse_args()

    docking_csv = os.path.abspath(args.docking_csv)
    results_root = os.path.abspath(args.results_root)
    ensure_dir(results_root)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(results_root, f"eval_run_{ts}")
    ensure_dir(run_dir)

    # 记录配置
    config = {
        "docking_csv": docking_csv,
        "results_root": results_root,
        "run_dir": run_dir,
        "threshold": args.threshold,
        "top_n": args.top_n,
        "md_steps": args.md_steps,
        "md_platform": args.md_platform,
    }
    ensure_dir(os.path.join(run_dir, "logs"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 读取 docking 结果并筛选分子
    rows = read_docking_csv(docking_csv)
    if not rows:
        raise SystemExit(f"No rows read from {docking_csv}")

    selected = select_molecules(rows, threshold=args.threshold, top_n=args.top_n)
    # 保存选择清单
    sel_rows = [
        {
            "compound_id": r.get("compound_id"),
            "smiles": r.get("smiles"),
            "binding_affinity": r.get("binding_affinity"),
        }
        for r in selected
    ]
    ensure_dir(os.path.join(run_dir, "selection"))
    save_csv(os.path.join(run_dir, "selection", "selected_molecules.csv"), sel_rows)

    # 统计最佳 docking
    best_row = min(rows, key=lambda x: x["binding_affinity"]) if rows else None

    # DFT 计算
    dft_dir = os.path.join(run_dir, "dft")
    ensure_dir(dft_dir)
    dft_results: List[Dict[str, Any]] = []
    for r in selected:
        lig_id = str(r.get("compound_id", ""))
        smiles = str(r.get("smiles", ""))
        lig_dir = os.path.join(dft_dir, lig_id)
        ensure_dir(lig_dir)
        dft_res = compute_dft_for_smiles(smiles, workdir=lig_dir, prefer="psi4")
        out_row = {
            "compound_id": lig_id,
            "smiles": smiles,
            "engine": dft_res.get("engine"),
            "success": dft_res.get("success"),
            "energy_hartree": dft_res.get("energy_hartree"),
            "homo_ev": dft_res.get("homo_ev"),
            "lumo_ev": dft_res.get("lumo_ev"),
            "gap_ev": dft_res.get("gap_ev"),
            "eta_ev": dft_res.get("eta_ev"),
            "dipole_debye": dft_res.get("dipole_debye"),
            "message": dft_res.get("message"),
            "binding_affinity": r.get("binding_affinity"),
        }
        dft_results.append(out_row)
    save_csv(os.path.join(dft_dir, "dft_results.csv"), dft_results)
    with open(os.path.join(dft_dir, "dft_results.json"), "w", encoding="utf-8") as f:
        json.dump(dft_results, f, ensure_ascii=False, indent=2)

    # 选出“电子稳定性最高”（gap 最大）
    best_dft = None
    try:
        best_dft = max(
            [r for r in dft_results if r.get("success") and isinstance(r.get("gap_ev"), (int, float))],
            key=lambda x: x.get("gap_ev", float("-inf")),
        )
    except Exception:
        best_dft = None

    # MD 运行（默认真空；若工具链支持水盒，将在 md_utils 内部或未来版本扩展）
    md_dir = os.path.join(run_dir, "md")
    ensure_dir(md_dir)
    md_summary_rows: List[Dict[str, Any]] = []
    for r in selected:
        lig_id = str(r.get("compound_id", ""))
        smiles = str(r.get("smiles", ""))
        lig_dir = os.path.join(md_dir, lig_id)
        ensure_dir(lig_dir)
        md_res = run_md_for_smiles(
            smiles=smiles,
            out_dir=lig_dir,
            steps=int(args.md_steps),
            temperature_k=300.0,
            timestep_fs=2.0,
            friction_per_ps=1.0,
            platform_preference=args.md_platform,
        )
        md_summary_rows.append({
            "compound_id": lig_id,
            "smiles": smiles,
            "success": md_res.get("success"),
            "engine": md_res.get("engine"),
            "platform": md_res.get("platform"),
            "message": md_res.get("message"),
            "trajectory_dcd": (md_res.get("outputs") or {}).get("trajectory_dcd"),
            "md_log_csv": (md_res.get("outputs") or {}).get("md_log_csv"),
        })
    save_csv(os.path.join(md_dir, "md_summary.csv"), md_summary_rows)
    with open(os.path.join(md_dir, "md_summary.json"), "w", encoding="utf-8") as f:
        json.dump(md_summary_rows, f, ensure_ascii=False, indent=2)

    # 汇总报告
    summary: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "run_dir": run_dir,
        "selected_count": len(selected),
        "dft_success": sum(1 for r in dft_results if r.get("success")),
        "md_success": sum(1 for r in md_summary_rows if r.get("success")),
        "best_docking": {
            "compound_id": best_row.get("compound_id") if best_row else None,
            "smiles": best_row.get("smiles") if best_row else None,
            "binding_affinity": best_row.get("binding_affinity") if best_row else None,
        },
        "best_electronic_stability": {
            "compound_id": best_dft.get("compound_id") if best_dft else None,
            "smiles": best_dft.get("smiles") if best_dft else None,
            "gap_ev": best_dft.get("gap_ev") if best_dft else None,
            "eta_ev": best_dft.get("eta_ev") if best_dft else None,
            "engine": best_dft.get("engine") if best_dft else None,
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 简要 Markdown 报告
    md_lines: List[str] = []
    md_lines.append(f"# {MODEL_NAME}：DFT + MD 运行报告\n")
    md_lines.append(f"- 运行目录: `{run_dir}`\n")
    md_lines.append(f"- 模型: {MODEL_NAME}\n")
    md_lines.append(f"- 分子筛选: 阈值 {args.threshold} kcal/mol，Top-N {args.top_n}\n")
    md_lines.append(f"- DFT 成功: {summary['dft_success']} / {len(selected)}\n")
    md_lines.append(f"- MD 成功: {summary['md_success']} / {len(selected)}\n")
    if summary["best_docking"]["compound_id"] is not None:
        md_lines.append("\n## 最高（最负）对接评分\n")
        md_lines.append(f"- 分子: {summary['best_docking']['compound_id']}\n")
        md_lines.append(f"- SMILES: {summary['best_docking']['smiles']}\n")
        md_lines.append(f"- 对接评分: {summary['best_docking']['binding_affinity']} kcal/mol\n")
    if summary["best_electronic_stability"]["compound_id"] is not None:
        md_lines.append("\n## 最高电子稳定性（按带隙）\n")
        md_lines.append(f"- 分子: {summary['best_electronic_stability']['compound_id']}\n")
        md_lines.append(f"- SMILES: {summary['best_electronic_stability']['smiles']}\n")
        md_lines.append(f"- 带隙: {summary['best_electronic_stability']['gap_ev']} eV\n")
        md_lines.append(f"- 全局硬度 η ≈ {summary['best_electronic_stability']['eta_ev']} eV\n")
        md_lines.append(f"- 计算引擎: {summary['best_electronic_stability']['engine']}\n")

    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
