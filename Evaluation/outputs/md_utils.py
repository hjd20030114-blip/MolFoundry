#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD 工具函数：
- 使用 OpenFF + OpenMM 对单个小分子（SMILES）进行真空 Langevin 动力学模拟
- 若缺少依赖（openmm/openff/rdkit）则优雅降级并返回失败信息
- 输出：
  - topology.pdb（若RDKit可用）
  - trajectory.dcd（若OpenMM可用）
  - md_log.csv（能量/温度/步数）
  - md_summary.json（简要统计）

注意：
- 该实现默认在真空中模拟。若需显式溶剂（TIP3P水盒），建议安装 openmmforcefields
  并使用 SMIRNOFFTemplateGenerator 将 OpenFF 小分子参数与 TIP3P 水结合。
"""
from __future__ import annotations
import os
import io
import json
import csv
import math
import time
import shutil
import tempfile
from typing import Optional, Dict, Any

# 依赖探测

def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False

HAS_RDKIT = _has_module("rdkit")
if HAS_RDKIT:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        HAS_RDKIT = False

HAS_OPENFF = _has_module("openff.toolkit")
if HAS_OPENFF:
    try:
        from openff.toolkit.topology import Molecule as OFFMolecule
        from openff.toolkit.topology import Topology as OFFTopology
        from openff.toolkit.typing.engines.smirnoff import ForceField as OFFForceField
        from openff.units import unit as offunit
    except Exception:
        HAS_OPENFF = False

# OpenMM 兼容导入（prefer openmm>=7.7 的新命名空间）
OPENMM_IMPL = None
try:
    import openmm
    from openmm import app, unit
    OPENMM_IMPL = "openmm"
except Exception:
    try:
        from simtk import openmm
        from simtk.openmm import app
        from simtk import unit
        OPENMM_IMPL = "simtk"
    except Exception:
        OPENMM_IMPL = None

HAS_OPENMM = OPENMM_IMPL is not None


def smiles_to_offmol(smiles: str) -> Optional["OFFMolecule"]:
    if not HAS_OPENFF:
        return None
    try:
        offmol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
        if HAS_RDKIT:
            rdmol = Chem.MolFromSmiles(smiles)
            rdmol = Chem.AddHs(rdmol)
            AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(rdmol)
            offmol2 = OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)
            if offmol2 is not None:
                offmol = offmol2
        return offmol
    except Exception:
        return None


def write_rdkit_pdb(smiles: str, out_pdb: str) -> bool:
    if not HAS_RDKIT:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        pdb_block = Chem.MolToPDBBlock(mol)
        with open(out_pdb, "w", encoding="utf-8") as f:
            f.write(pdb_block)
        return True
    except Exception:
        return False


def run_md_for_smiles(
    smiles: str,
    out_dir: str,
    steps: int = 2500000,
    temperature_k: float = 300.0,
    timestep_fs: float = 2.0,
    friction_per_ps: float = 1.0,
    platform_preference: Optional[str] = None,
) -> Dict[str, Any]:
    """对一个SMILES在真空中运行OpenMM Langevin MD（若可用），返回运行结果摘要。

    参数：
    - steps：默认 2,500,000（约 5 ns @ 2 fs）；可根据机器性能调整
    - platform_preference：可选 'CUDA' | 'OpenCL' | 'CPU'
    """
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "success": False,
        "engine": None,
        "smiles": smiles,
        "steps": steps,
        "temperature_k": temperature_k,
        "timestep_fs": timestep_fs,
        "friction_per_ps": friction_per_ps,
        "platform": None,
        "message": None,
        "outputs": {
            "topology_pdb": None,
            "trajectory_dcd": None,
            "md_log_csv": None,
        },
    }

    # 尝试导出拓扑PDB
    pdb_path = os.path.join(out_dir, "topology.pdb")
    if write_rdkit_pdb(smiles, pdb_path):
        summary["outputs"]["topology_pdb"] = pdb_path

    if not (HAS_OPENFF and HAS_OPENMM):
        summary["message"] = "openff/openmm not installed; skip MD"
        return summary

    try:
        offmol = smiles_to_offmol(smiles)
        if offmol is None:
            summary["message"] = "failed to build OpenFF molecule"
            return summary

        off_top = OFFTopology.from_molecules([offmol])
        ff = OFFForceField("openff-2.0.0.offxml", allow_cosmetic_attributes=True)
        system = ff.create_openmm_system(off_top)

        # 初始坐标（nm）
        if offmol.n_conformers == 0:
            # 生成一个构象（以避免无坐标）
            if HAS_RDKIT:
                rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(rdmol)
                offmol = OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)
                off_top = OFFTopology.from_molecules([offmol])
            else:
                # 没有坐标，无法继续
                summary["message"] = "no conformer coordinates available"
                return summary

        positions = offmol.conformers[0]  # OpenFF 带单位（Å）
        # 转为 OpenMM 位置（nm）
        positions_nm = positions.to(offunit.nanometer)
        positions_nm = positions_nm.m

        # 构造 OpenMM 组件
        omm_top = off_top.to_openmm()
        integrator = openmm.LangevinIntegrator(
            temperature_k * unit.kelvin,
            friction_per_ps / unit.picosecond,
            timestep_fs * unit.femtosecond,
        )

        # 平台选择
        platform = None
        if platform_preference:
            try:
                platform = openmm.Platform.getPlatformByName(platform_preference)
            except Exception:
                platform = None
        if platform is None:
            # 自动选择
            try:
                platform = openmm.Platform.getPlatformByName("CUDA")
            except Exception:
                try:
                    platform = openmm.Platform.getPlatformByName("OpenCL")
                except Exception:
                    platform = openmm.Platform.getPlatformByName("CPU")

        simulation = app.Simulation(omm_top, system, integrator, platform)
        simulation.context.setPositions(positions_nm)

        # 能量最小化
        try:
            simulation.minimizeEnergy(maxIterations=200)
        except Exception:
            pass

        # 报告器
        dcd_path = os.path.join(out_dir, "trajectory.dcd")
        log_path = os.path.join(out_dir, "md_log.csv")
        simulation.reporters.append(app.DCDReporter(dcd_path, int(max(1, steps // 1000))))
        simulation.reporters.append(
            app.StateDataReporter(
                log_path,
                int(max(1, steps // 1000)),
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                speed=True,
            )
        )

        # 平衡与采样
        simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin)
        simulation.step(int(steps))

        summary["success"] = True
        summary["engine"] = "openmm"
        summary["platform"] = platform.getName() if platform else None
        summary["outputs"]["trajectory_dcd"] = dcd_path
        summary["outputs"]["md_log_csv"] = log_path
        return summary
    except Exception as ex:
        summary["message"] = str(ex)
        return summary


if __name__ == "__main__":
    # 简单自测（不会在导入时执行）
    out = run_md_for_smiles("CCO", out_dir=tempfile.mkdtemp(prefix="md_test_"), steps=2000)
    print(json.dumps(out, ensure_ascii=False, indent=2))
