#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 PLIP 的蛋白-配体 2D 相互作用图批量生成器（方案A）
- 支持输入：蛋白 PDB 与一批配体（PDB/PDBQT/SDF/MOL2）
- 自动将 PDBQT 转为 PDB，并与蛋白合并为复合物 PDB
- 调用 PLIP CLI 对复合物进行相互作用分析，导出 2D 示意图
- 生成 HTML 画册并可选关联现有 2D/3D 页面

依赖：
- plip  (pip install plip)
- biopython (已在 requirements)
- rdkit-pypi (已在 requirements)

示例：
python HJD/scripts/visualization_plip_interactions.py \
  --protein HJD/data/1p65.pdb \
  --ligand-dir HJD/experiment_report/docking/docking_results \
  --out-dir HJD/results/run_20251004_001/visualization_plip \
  --csv HJD/results/run_20251004_001/ligands/generated_ligands.csv \
  --top-n 20
"""
from __future__ import annotations
import os
import sys
import re
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import math
import time
import zipfile
from datetime import datetime

import pandas as pd
import xml.etree.ElementTree as ET

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

# Open Babel（优先用于 PDBQT->PDB 转换）
try:
    from openbabel import pybel
    HAS_OBABEL = True
except Exception:
    HAS_OBABEL = False

# cairosvg（可选，用于将 PLIP 导出的 SVG 转为 PNG）
try:
    import cairosvg  # type: ignore
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False

# PyMOL（用于生成图片），若不可用则仅生成 XML/TXT
try:
    import pymol  # type: ignore
    HAS_PYMOL = True
except Exception:
    HAS_PYMOL = False

# networkx / matplotlib（用于回退生成 2D 交互示意图）
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------------------
# PyMOL 3D 渲染（卡通+表面+配体sticks+接触残基+虚线氢键）
# ---------------------------
def render_3d_docking_pymol(
    complex_pdb: Path,
    out_png: Path,
    width: int = 1600,
    height: int = 1200,
    surface_transparency: float = 0.6,
    cartoon_transparency: float = 0.18,
    contact_cutoff: float = 4.2,
    fast: bool = False,
    label_top_k: int = 6,
) -> Optional[Path]:
    # 优先使用 PyMOL Python 模块；若不可用，回退到 pymol CLI + .pml 脚本
    try:
        def _get_top_contact_residues() -> List[Tuple[str, str, str]]:
            """读取复合物PDB，计算与配体链L最接近的受体残基，返回[(chain, resi, resn)]，按距离升序，仅取前K。"""
            try:
                lig_atoms: List[Tuple[float,float,float]] = []
                residues: Dict[Tuple[str,str,str], List[Tuple[float,float,float]]] = {}
                with open(complex_pdb, 'r') as f:
                    for line in f:
                        if not line.startswith(('ATOM', 'HETATM')) or len(line) < 54:
                            continue
                        try:
                            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        except ValueError:
                            continue
                        chain = line[21]
                        resn = line[17:20].strip()
                        resi = line[22:26].strip()
                        elem = line[76:78].strip().upper() if len(line) >= 78 else ''
                        if chain == 'L':
                            if elem != 'H':
                                lig_atoms.append((x,y,z))
                        else:
                            if line.startswith('ATOM') and elem != 'H':
                                residues.setdefault((chain, resi, resn), []).append((x,y,z))
                if not lig_atoms or not residues:
                    return []
                res_dists: List[Tuple[float, Tuple[str,str,str]]] = []
                for key, coords in residues.items():
                    md = 1e9
                    for (px,py,pz) in coords:
                        for (lx,ly,lz) in lig_atoms:
                            d = math.dist((px,py,pz),(lx,ly,lz))
                            if d < md:
                                md = d
                    if md <= contact_cutoff:
                        res_dists.append((md, key))
                res_dists.sort(key=lambda t: t[0])
                selected = [k for _, k in res_dists]
                if label_top_k and label_top_k > 0:
                    selected = selected[:label_top_k]
                # 转为[(chain, resi, resn)]
                return [(ch, rs, rn) for (ch, rs, rn) in selected]
            except Exception:
                return []

        def _get_label_offsets(top_res: List[Tuple[str,str,str]], push: float = 2.5) -> Dict[Tuple[str,str], Tuple[float,float,float]]:
            """为每个要标注的残基（chain, resi）计算一个从配体质心指向该残基CA的单位向量，并沿该向量偏移 push Å，返回偏移字典。"""
            try:
                lig: List[Tuple[float,float,float]] = []
                ca: Dict[Tuple[str,str], Tuple[float,float,float]] = {}
                with open(complex_pdb, 'r') as f:
                    for line in f:
                        if not line.startswith(('ATOM','HETATM')) or len(line) < 54:
                            continue
                        try:
                            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        except ValueError:
                            continue
                        chain = line[21]
                        resi = line[22:26].strip()
                        name = line[12:16].strip()
                        elem = line[76:78].strip().upper() if len(line) >= 78 else ''
                        if chain == 'L':
                            if elem != 'H':
                                lig.append((x,y,z))
                        else:
                            if name == 'CA':
                                ca[(chain, resi)] = (x,y,z)
                if not lig:
                    return {}
                # 质心
                cx = sum(p[0] for p in lig)/len(lig)
                cy = sum(p[1] for p in lig)/len(lig)
                cz = sum(p[2] for p in lig)/len(lig)
                out: Dict[Tuple[str,str], Tuple[float,float,float]] = {}
                for ch, rs, rn in top_res:
                    p = ca.get((ch, rs))
                    if not p:
                        continue
                    vx, vy, vz = (p[0]-cx, p[1]-cy, p[2]-cz)
                    norm = math.sqrt(max(1e-12, vx*vx+vy*vy+vz*vz))
                    dx, dy, dz = (push*vx/norm, push*vy/norm, push*vz/norm)
                    out[(ch, rs)] = (dx, dy, dz)
                return out
            except Exception:
                return {}
        def _render_via_cli() -> Optional[Path]:
            import os as _os
            pymol_bin = _os.environ.get('PYMOL_BIN') or which('pymol')
            if not pymol_bin:
                return None
            # 避免在 f-string 表达式里出现带反斜杠的字面量，改为逐行拼接
            alpha = surface_transparency if surface_transparency >= cartoon_transparency else cartoon_transparency
            top_res = _get_top_contact_residues()
            lines: List[str] = [
                "reinitialize",
                "bg_color white",
                "set orthoscopic, on",
                "set two_sided_lighting, on",
                f"load {complex_pdb}, complex",
                "remove hydro",
                "hide everything, all",
                "",
                "show cartoon, polymer",
                "color gray70, polymer",
                "show surface, polymer",
                "set surface_color, gray80",
                f"set transparency, {alpha}",
                "",
                "select lig, chain L",
                "show sticks, lig",
                "set stick_radius, 0.22, lig",
                "color magenta, lig",
                "",
                f"select contact, byres (polymer within {contact_cutoff} of lig)",
                "show sticks, contact",
                "set stick_radius, 0.18, contact",
                "color wheat, contact",
                "",
                "set label_font_id, 7",
                "set label_size, -5",
                "set label_color, black",
                "set label_outline_color, white",
            ]
            # 仅标注前K个残基的CA
            for ch, rs, rn in top_res:
                lines.append(f'label chain {ch} and resi {rs} and name CA, "%s%s" % (resn, resi)')
            # CLI 分支不再设置 label_position（部分旧版 PyMOL 不支持，避免报错）
            lines += [
                "",
                "distance hbonds, lig and donor, contact and acceptor, 3.3",
                "set dash_color, yellow, hbonds",
                "set dash_width, 2.2",
                "hide labels, hbonds",
                "",
                "orient lig",
                "zoom lig, 10",
                "set ray_opaque_background, off",
                "set antialias, 2",
                "set ray_shadows, off",
            ]
            if fast:
                lines += [
                    "set defer_builds_mode, 3",
                    "set surface_quality, 0",
                ]
            lines += [
                f"png {out_png}, {width}, {height}, ray={0 if fast else 1}",
                "quit",
            ]
            pml = "\n".join(lines) + "\n"
            tmp_pml = out_png.with_suffix('.pml')
            tmp_pml.write_text(pml, encoding='utf-8')
            import subprocess as _sp
            try:
                _sp.run([pymol_bin, '-cq', str(tmp_pml)], check=True, stdout=_sp.PIPE, stderr=_sp.PIPE)
            except Exception as _e:
                print(f"[WARN] pymol CLI 渲染失败: {_e}")
                return None
            finally:
                try:
                    tmp_pml.unlink(missing_ok=True)  # type: ignore
                except Exception:
                    pass
            return out_png if out_png.exists() else None

        if HAS_PYMOL:
            # 延迟导入，避免无 PyMOL 环境报错
            import pymol  # type: ignore
            from pymol import cmd  # type: ignore
            pymol.finish_launching(['pymol', '-cq'])
            cmd.reinitialize()
            cmd.bg_color('white')
            cmd.set('orthoscopic', 1)
            cmd.set('two_sided_lighting', 1)

            cmd.load(str(complex_pdb), 'complex')
            cmd.remove('hydro')
            cmd.hide('everything', 'all')

            # 受体：卡通 + 半透明表面
            cmd.show('cartoon', 'polymer')
            cmd.color('gray70', 'polymer')
            # cartoon 透明度：旧版 PyMOL 可能不支持 cartoon_transparency
            try:
                cmd.set('cartoon_transparency', cartoon_transparency)
            except Exception:
                try:
                    cmd.set('transparency', cartoon_transparency)
                except Exception:
                    pass
            cmd.show('surface', 'polymer')
            cmd.set('surface_color', 'gray80')
            # surface 透明度：旧版 PyMOL 可能不支持 surface_transparency
            try:
                cmd.set('surface_transparency', surface_transparency)
            except Exception:
                try:
                    # 回退到全局 transparency（可能影响其他表示，但保证可用）
                    cmd.set('transparency', max(surface_transparency, cartoon_transparency))
                except Exception:
                    pass

            # 配体：链 L
            cmd.select('lig', 'chain L')
            cmd.show('sticks', 'lig')
            cmd.set('stick_radius', 0.22, 'lig')
            cmd.color('magenta', 'lig')

            # 接触残基：按距离 byres 选取
            cmd.select('contact', f'byres (polymer within {contact_cutoff} of lig)')
            cmd.show('sticks', 'contact')
            cmd.set('stick_radius', 0.18, 'contact')
            cmd.color('wheat', 'contact')

            # 残基标签（仅前K个，CA 原子）
            cmd.set('label_font_id', 7)
            cmd.set('label_size', -5)
            cmd.set('label_color', 'black')
            cmd.set('label_outline_color', 'white')
            _top = _get_top_contact_residues()
            if _top:
                # 计算偏移，轻微将标签推离表面
                _offsets = _get_label_offsets(_top, push=2.5)
                for ch, rs, rn in _top:
                    try:
                        cmd.label(f'chain {ch} and resi {rs} and name CA', '"%s%s" % (resn, resi)')
                        off = _offsets.get((ch, rs))
                        if off:
                            try:
                                cmd.set('label_position', list(off), f'chain {ch} and resi {rs} and name CA')
                            except Exception:
                                pass
                    except Exception:
                        pass

            # 氢键（虚线）
            cmd.distance('hbonds', 'lig and donor', 'contact and acceptor', 3.3)
            cmd.set('dash_color', 'yellow', 'hbonds')
            cmd.set('dash_width', 2.2)
            cmd.hide('labels', 'hbonds')

            # 视角与导出
            try:
                cmd.orient('lig')
                cmd.zoom('lig', 10)
            except Exception:
                cmd.orient('all')
                cmd.zoom('all', 1.0)
            cmd.set('ray_opaque_background', 0)
            cmd.set('antialias', 2)
            cmd.set('ray_shadows', 0)
            if fast:
                try:
                    cmd.set('defer_builds_mode', 3)
                    cmd.set('surface_quality', 0)
                except Exception:
                    pass
            try:
                cmd.png(str(out_png), width, height, ray=(0 if fast else 1))
            except Exception as _e:
                print(f"[WARN] PyMOL cmd.png 失败：{_e}")
                return None
            cmd.delete('all')
            # 若未生成则回退到 CLI
            if out_png.exists():
                return out_png
            else:
                print("[WARN] PyMOL 模块渲染后未生成 PNG，尝试 CLI 回退…")
                return _render_via_cli()
        else:
            # 直接走 CLI 回退
            return _render_via_cli()
    except Exception as e:
        print(f"[WARN] render_3d_docking_pymol 异常（尝试 CLI 回退）：{e}")
        try:
            return _render_via_cli()
        except Exception as _e:
            print(f"[WARN] CLI 回退也失败：{_e}")
            return None

HTML_STYLE = """
<style>
 body { font-family: Arial, sans-serif; margin: 20px; }
 .container { max-width: 1200px; margin: 0 auto; }
 .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(360px,1fr)); gap: 18px; }
 .card { background:#fff; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.08); padding:14px; }
 .card h3 { margin: 6px 0 8px 0; color:#2c3e50; }
 .meta { font-size: 13px; color:#555; line-height:1.6; }
 .thumb { width:100%; height:auto; border:1px solid #eee; border-radius:6px; background:#fafafa; }
 .row { margin: 8px 0; }
 .btn { display:inline-block; margin-right:8px; padding:6px 10px; background:#1976d2; color:#fff; border-radius:4px; text-decoration:none; font-size:13px; }
 .btn.green { background:#2e7d32; }
 .btn.gray { background:#607d8b; }
 .info { background:#f5f5f5; padding:10px 14px; border-radius:6px; margin: 10px 0; }
</style>
"""


def which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        fp = Path(p) / cmd
        if fp.exists() and os.access(fp, os.X_OK):
            return str(fp)
    return None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_figsize(s: str) -> Tuple[float, float]:
    """解析 --figsize 参数字符串，例如 '8,8' 或 '8x8' 或 '8 8'。单位：英寸。"""
    s = (s or '').lower().replace('x', ',').replace(' ', ',')
    parts = [p for p in s.split(',') if p]
    try:
        def angle_deg(p1, p2, p3) -> float:
            # 角 p1-p2-p3，单位度
            import math as _m
            v1 = (p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])
            v2 = (p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2])
            def _dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
            def _norm(a): return _m.sqrt(_dot(a,a)) + 1e-8
            cosv = max(-1.0, min(1.0, _dot(v1,v2)/(_norm(v1)*_norm(v2))))
            return _m.degrees(_m.acos(cosv))

        def nearest_neighbor(anchor: Tuple[float,float,float,str], group: List[Tuple[float,float,float,str]], maxbond: float = 1.9) -> Optional[Tuple[float,float,float,str]]:
            ax, ay, az, _ = anchor
            best = None; md = 1e9
            for (x,y,z,e) in group:
                d = math.dist((ax,ay,az),(x,y,z))
                if d < 1e-3:
                    continue
                if d < md and d <= maxbond:
                    md = d; best = (x,y,z,e)
            return best
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return (8.0, 8.0)


def overlay_header_on_png(png_path: Path, smiles: str, affinity: str) -> Optional[Path]:
    """在 PNG 左上角叠加 SMILES 与结合能文本（半透明白底）。
    - 不新增依赖：使用 matplotlib 读写图像；保持原像素尺寸
    - 若 matplotlib 不可用或 png 不存在则跳过
    """
    if not HAS_MPL or not png_path.exists():
        return None
    try:
        img = plt.imread(str(png_path))
        h, w = img.shape[0], img.shape[1]
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.set_axis_off()
        lines = []
        if smiles:
            lines.append(f"SMILES: {smiles}")
        if affinity and affinity != 'N/A':
            lines.append(f"Affinity: {affinity}")
        if lines:
            txt = "\n".join(lines)
            try:
                import matplotlib.patheffects as _pe  # noqa: F401
                path_effects = [pe.Stroke(linewidth=2.0, foreground='white'), pe.Normal()]
            except Exception:
                path_effects = None
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes,
                va='top', ha='left', fontsize=10, color='black',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75),
                path_effects=path_effects,
                zorder=10,
            )
        fig.savefig(str(png_path), dpi=dpi, transparent=False)
        plt.close(fig)
        return png_path
    except Exception:
        return None


def parse_affinity_value(s: Optional[str]) -> Optional[float]:
    """解析结合能文本为数值（float）。
    - 兼容中文/英文单位与全角/异体负号：例如 "-8.6", "−7.5 kcal/mol", "8.2"
    - 返回值越小越好（假定更负=更优）。若解析失败返回 None
    """
    if not s:
        return None
    try:
        txt = str(s).strip().lower()
        txt = txt.replace('kcal/mol', '').replace('kcal', '').replace(' ', '')
        txt = txt.replace('−', '-')  # 统一负号
        # 提取首个可解析数字（含负号/小数）
        m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None

def guess_compound_id(file: Path) -> str:
    name = file.stem
    # 去掉常见后缀
    name = re.sub(r"_docked$", "", name)
    name = re.sub(r"_pose\d+$", "", name)
    return name


def pdbqt_to_pdb(pdbqt_path: Path, pdb_out: Path) -> Path:
    """最小可用的 PDBQT→PDB 转换：复制 ATOM/HETATM 行并规范元素列。
    注意：用于 PLIP 相互作用识别已足够，若遇到异常可考虑安装 openbabel 进行转换。
    """
    with open(pdbqt_path, 'r') as fin, open(pdb_out, 'w') as fout:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM')):
                # PDBQT 基本与 PDB 对齐，直接截取前54列坐标等，元素尝试放在 77-78 列
                rec = line[:54]
                # 粗略推断元素（列 13-14 原子名的首字母）
                atom_name = line[12:16].strip()
                elem = (atom_name[0] if atom_name else 'C').upper().rjust(2)
                line_out = f"{rec}{line[54:76]}{elem}\n"
                fout.write(line_out)
    return pdb_out


def normalize_ligand_pdb(pdb_path: Path) -> None:
    """标准化配体 PDB：
    - 记录类型统一为 HETATM
    - 残基名 RESNAME= LIG（列 18-20）
    - 链 ID = 'L'（列 22）
    - 残基序号 RESSEQ= 1（列 23-26）
    - 如元素列为空，基于原子名首字母填充（列 77-78）
    """
    lines: List[str] = []
    with open(pdb_path, 'r') as fin:
        for line in fin:
            if line.startswith(('ATOM', 'HETATM')):
                # 确保长度足够
                buf = list(line.rstrip('\n'))
                if len(buf) < 80:
                    buf += [' '] * (80 - len(buf))

                # 记录类型 HETATM（列 1-6）
                rec = list("HETATM")
                for i in range(6):
                    buf[i] = rec[i] if i < len(rec) else ' '

                # RESNAME（列 18-20, 索引 17-20）
                res = list("LIG")
                for i, ch in enumerate(res):
                    buf[17 + i] = ch

                # CHAIN（列 22, 索引 21）
                buf[21] = 'L'

                # RESSEQ（列 23-26, 索引 22-26），右对齐 4 宽度
                resseq = f"{1:>4}"
                for i, ch in enumerate(resseq):
                    buf[22 + i] = ch

                # 元素（列 77-78, 索引 76-78）
                atom_name = ''.join(buf[12:16]).strip()
                elem = (atom_name[0] if atom_name else 'C').upper().rjust(2)
                buf[76], buf[77] = elem[0], elem[1]

                lines.append(''.join(buf) + '\n')
            else:
                lines.append(line)
    with open(pdb_path, 'w') as fout:
        fout.writelines(lines)


def convert_with_openbabel(pdbqt_path: Path, pdb_out: Path) -> Path:
    """使用 Open Babel 将 PDBQT 转为 PDB，并做标准化。"""
    try:
        mols = pybel.readfile('pdbqt', str(pdbqt_path))
        mol = next(mols, None)
        if mol is None:
            raise ValueError("Open Babel 读取 PDBQT 失败")
        mol.write('pdb', str(pdb_out), overwrite=True)
        normalize_ligand_pdb(pdb_out)
        return pdb_out
    except Exception:
        # 回退到最小可用转换
        out = pdbqt_to_pdb(pdbqt_path, pdb_out)
        normalize_ligand_pdb(out)
        return out

def draw_interaction_map_from_xml(xml_path: Path, out_png: Path, title: str, dpi: int = 300, figsize: Tuple[float, float] = (6.0, 6.0)) -> Optional[Path]:
    """从 PLIP 的 report.xml 解析交互，绘制 2D 网络图（回退方案）。"""
    if not HAS_NX or not HAS_MPL or not xml_path.exists():
        return None
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        # 支持多个 binding site，这里只取第一个
        bs = root.find('.//bindingsite')
        if bs is None:
            return None
        G = nx.Graph()
        ligand_node = 'LIG'
        G.add_node(ligand_node, kind='ligand')

        edge_styles: Dict[Tuple[str, str], List[str]] = {}

        def add_edge(resname: str, restype: str, resnr: str, chain: str, itype: str):
            node = f"{restype}{resnr}:{chain}"
            G.add_node(node, kind='residue')
            key = (ligand_node, node)
            edge_styles.setdefault(key, []).append(itype)

        # 解析常见交互
        def parse_res_fields(elem) -> Tuple[str, str, str]:
            resnr = (elem.findtext('resnr') or '').strip()
            restype = (elem.findtext('restype') or '').strip()
            reschain = (elem.findtext('reschain') or '').strip()
            return restype, resnr, reschain

        # hydrophobic
        for e in bs.findall('.//hydrophobic_interactions/hydrophobic_interaction'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'hydrophobic')
        # hydrogen bonds
        for e in bs.findall('.//hydrogen_bonds/hydrogen_bond'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'hbond')
        # pi stacking
        for e in bs.findall('.//pi_stacking/pi_stack'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'pi')
        # salt bridges
        for e in bs.findall('.//salt_bridges/salt_bridge'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'salt')
        # water bridges
        for e in bs.findall('.//water_bridges/water_bridge'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'water')
        # halogen bonds
        for e in bs.findall('.//halogen_bonds/halogen_bond'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'halogen')
        # metal complexes
        for e in bs.findall('.//metal_complexes/metal_complex'):
            restype, resnr, reschain = parse_res_fields(e)
            add_edge('LIG', restype, resnr, reschain, 'metal')

        # 布局：LIG 在中心，其余节点环形
        residues = [n for n, d in G.nodes(data=True) if d.get('kind') == 'residue']
        pos = nx.circular_layout(residues, scale=2.5)
        pos[ligand_node] = (0.0, 0.0)

        # 绘制
        plt.figure(figsize=figsize)
        # 节点
        nx.draw_networkx_nodes(G, pos, nodelist=[ligand_node], node_color='#1976d2', node_size=740)
        nx.draw_networkx_nodes(G, pos, nodelist=residues, node_color='#bbdefb', node_size=540, edgecolors='#78909c', linewidths=1.0)
        # 标签
        labels = {ligand_node: ligand_node}
        labels.update({r: r for r in residues})
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

        # 与距离回退统一的线型/线宽映射
        style_map = {
            'hbond': ('#2e7d32', (12,6)),
            'salt': ('#c62828', None),
            'pi': ('#8e24aa', (8,4,2,4)),
            'pi_T': ('#f06292', (10,3,3,3)),
            'pi_cation': ('#00acc1', (10,5,2,5)),
            'pi_anion': ('#42a5f5', (10,5)),
            'pi_alkyl': ('#ba68c8', (14,6)),
            'halogen': ('#ef6c00', (14,4)),
            'chb': ('#b0bec5', (6,8)),
            'water': ('#0277bd', (6,6)),
            'metal': ('#455a64', None),
            'hydrophobic': ('#6d4c41', None),
            'vdw': ('#90a4ae', None),
        }
        width_map = {
            'hbond': 3.2, 'salt': 3.4,
            'pi': 3.0, 'pi_T': 3.0, 'pi_cation': 3.0, 'pi_anion': 3.0, 'pi_alkyl': 3.0,
            'halogen': 3.2, 'chb': 2.6, 'water': 2.6, 'metal': 3.4,
            'hydrophobic': 2.8, 'vdw': 2.4,
        }

        # 按边的首要类型绘制，并统计出现的类型
        present_types = set()
        for (u, v), types in edge_styles.items():
            t = types[0] if types else 'vdw'
            present_types.add(t)
            color, dashes = style_map.get(t, ('#616161', None))
            lw = width_map.get(t, 2.6)
            ls = (0, dashes) if dashes else '-'
            line = plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color, linewidth=lw, linestyle=ls, solid_capstyle='round')[0]
            try:
                import matplotlib.patheffects as pe
                line.set_path_effects([pe.Stroke(linewidth=lw+1.2, foreground='white'), pe.Normal()])
            except Exception:
                pass
            # 对氢键/盐桥添加箭头（v->u，假定 u 为 LIG）
            if t in ('hbond', 'salt'):
                if u == ligand_node:
                    x0,y0 = pos[u]; x1,y1 = pos[v]
                else:
                    x0,y0 = pos[v]; x1,y1 = pos[u]
                plt.annotate('', xy=(x0,y0), xytext=(x1,y1),
                             arrowprops=dict(arrowstyle='-|>', color=color, lw=0, shrinkA=0, shrinkB=0, mutation_scale=16))

        # 图例：仅展示本图出现的类型
        import matplotlib.lines as mlines
        legend_spec = [
            ('Conventional Hydrogen Bond','hbond'),
            ('Salt Bridge','salt'),
            ('Pi-Pi Stacking','pi'),
            ('Pi-T shaped','pi_T'),
            ('Pi-Cation','pi_cation'),
            ('Pi-Anion','pi_anion'),
            ('Pi-Alkyl','pi_alkyl'),
            ('Halogen Bond','halogen'),
            ('Carbon Hydrogen Bond','chb'),
            ('Water Bridge','water'),
            ('Metal Coordination','metal'),
            ('Hydrophobic/Alkyl','hydrophobic'),
            ('van der Waals','vdw'),
        ]
        used = [(k,d) for (k,d) in legend_spec if d in present_types]
        handles = []
        labels_ = []
        if used:
            for k, d in used:
                col, dash = style_map[d]
                ln = mlines.Line2D([], [], color=col, linewidth=width_map.get(d, 2.6))
                if dash:
                    ln.set_dashes(dash)
                handles.append(ln)
                labels_.append(k)
            plt.legend(handles, labels_, loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#e0e0e0', fontsize=9, fancybox=True, borderpad=0.6, handlelength=2.8, handletextpad=0.8, labelspacing=0.5)
        else:
            # 无相互作用时也给出提示图例
            dummy = mlines.Line2D([], [], color='#9e9e9e', linewidth=2.2)
            dummy.set_dashes((6,6))
            plt.legend([dummy], ['No interactions'], loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#e0e0e0', fontsize=9, fancybox=True, borderpad=0.6, handlelength=2.8, handletextpad=0.8, labelspacing=0.5)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(out_png), dpi=dpi, bbox_inches='tight', pad_inches=0.15)
        plt.close()
        return out_png
    except Exception:
        return None

def draw_interaction_map_by_distance(
    complex_pdb: Path,
    out_png: Path,
    title: str,
    cutoff: float = 4.5,
    smiles: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (8.0, 8.0),
    strict_angles: bool = True,
    hb_angle_min: float = 140.0,
    hal_angle_min: float = 150.0,
) -> Optional[Path]:
    """当 PLIP 未给出 bindingsite 时，基于几何邻近关系画简化示意图。
    - 从复合物 PDB 中读取：配体链 L 的原子，与受体（ATOM 记录，非 L 链）的原子
    - 以 cutoff 阈值筛选邻近残基，生成 LIG 与残基节点的网络图
    """
    if not HAS_NX or not HAS_MPL or not complex_pdb.exists():
        return None
    try:
        prot_atoms: Dict[str, list] = {}  # key: residue label, value: list[(x,y,z,elem)]
        lig_atoms: list = []  # list[(x,y,z,elem)]
        prot_resname: Dict[str, str] = {}
        water_oxygens: List[Tuple[float,float,float]] = []
        metals: List[Tuple[str, Tuple[float,float,float]]] = []
        METAL_ELEMS = {"ZN","MG","MN","FE","CU","NI","CA","NA","K","CO","CD"}
        with open(complex_pdb, 'r') as f:
            for line in f:
                if not line.startswith(('ATOM', 'HETATM')) or len(line) < 54:
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                chain = line[21]
                resname = line[17:20].strip()
                resseq = line[22:26].strip()
                # 元素过滤：排除氢（若元素列可用）
                elem = line[76:78].strip().upper() if len(line) >= 78 else ''
                if chain == 'L':
                    if elem != 'H':
                        lig_atoms.append((x, y, z, elem or 'C'))
                else:
                    if line.startswith('ATOM') and elem != 'H':
                        key = f"{resname}{resseq}:{chain}"
                        prot_atoms.setdefault(key, []).append((x, y, z, elem or 'C'))
                        prot_resname[key] = resname
                    # 水与金属（HETATM），纳入额外列表
                    if line.startswith('HETATM'):
                        if resname == 'HOH' and elem == 'O':
                            water_oxygens.append((x,y,z))
                        if elem in METAL_ELEMS:
                            metals.append((elem, (x,y,z)))

        if not lig_atoms or not prot_atoms:
            return None

        # 计算最短距离与类型判定
        hydro_res = {"ALA","VAL","LEU","ILE","MET","PHE","PRO","TRP"}
        aromatic_res = {"PHE","TYR","TRP","HIS"}
        pos_res = {"LYS","ARG"}
        neg_res = {"ASP","GLU"}

        # 配体电荷与是否芳香
        lig_charge = 0
        lig_has_aromatic = False
        lig_aromatic_indices = set()
        if HAS_RDKIT and smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                lig_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                lig_has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
                lig_aromatic_indices = {a.GetIdx() for a in mol.GetAromaticAtoms()}

        neighbors: List[Tuple[str, float, str]] = []  # (res, dist, type)
        # 预分组原子：用于卤键/氢键等快速判断
        lig_halogen = [(x,y,z,le) for (x,y,z,le) in lig_atoms if le in {"F","CL","BR","I"}]
        lig_heavy_NO = [(x,y,z,le) for (x,y,z,le) in lig_atoms if le in {"N","O"}]
        lig_carbons  = [(x,y,z,le) for (x,y,z,le) in lig_atoms if le == 'C']
        for res, coords in prot_atoms.items():
            # 最短距离
            min_d = 1e9
            min_pair = None
            min_p_atom = None
            min_l_atom = None
            for (px, py, pz, pe) in coords:
                for (lx, ly, lz, le) in lig_atoms:
                    d = math.dist((px,py,pz),(lx,ly,lz))
                    if d < min_d:
                        min_d = d; min_pair = (pe, le); min_p_atom=(px,py,pz,pe); min_l_atom=(lx,ly,lz,le)

            if min_d > cutoff:
                continue

            rname = prot_resname.get(res, "RES")
            itype = 'vdw'

            # 判定 H-bond（N/O…N/O 且距离较小）
            if min_pair and ((min_pair[0] in {'N','O'}) and (min_pair[1] in {'N','O'}) and min_d <= 3.2):
                # 角度约束：近似 Donor-Heavy – Donor – Acceptor 或 Donor – Acceptor – Acceptor-Heavy
                pass_hb_angle = True
                if strict_angles and (min_p_atom and min_l_atom):
                    # donor/acceptor未知，双向尝试角度
                    nn_p = nearest_neighbor(min_p_atom, coords)
                    nn_l = nearest_neighbor(min_l_atom, lig_atoms)
                    ang_ok = False
                    if nn_p:
                        ang = angle_deg((nn_p[0],nn_p[1],nn_p[2]), (min_p_atom[0],min_p_atom[1],min_p_atom[2]), (min_l_atom[0],min_l_atom[1],min_l_atom[2]))
                        if ang >= hb_angle_min:
                            ang_ok = True
                    if not ang_ok and nn_l:
                        ang = angle_deg((min_p_atom[0],min_p_atom[1],min_p_atom[2]), (min_l_atom[0],min_l_atom[1],min_l_atom[2]), (nn_l[0],nn_l[1],nn_l[2]))
                        if ang >= hb_angle_min:
                            ang_ok = True
                    pass_hb_angle = ang_ok
                itype = 'hbond' if pass_hb_angle else 'vdw'
            else:
                # 判定卤键：配体卤素…蛋白 N/O（<=3.8 Å）
                halogen_hit = False
                halogen_pass_angle = True
                for (lx,ly,lz,le) in lig_halogen:
                    for (px,py,pz,pe) in coords:
                        if pe in {'N','O'}:
                            d_ = math.dist((lx,ly,lz),(px,py,pz))
                            if d_ <= 3.8:
                                halogen_hit = True
                                if strict_angles:
                                    # 近似找 C–X 键：X 附近最近 C（<=2.1 Å）
                                    cx = nearest_neighbor((lx,ly,lz,le), lig_carbons, maxbond=2.1)
                                    if cx:
                                        ang = angle_deg((cx[0],cx[1],cx[2]), (lx,ly,lz), (px,py,pz))
                                        halogen_pass_angle = ang >= hal_angle_min
                                break
                    if halogen_hit:
                        break
                if halogen_hit and halogen_pass_angle:
                    itype = 'halogen'
                else:
                # 判定盐桥
                    if lig_charge != 0 and ((lig_charge > 0 and rname in neg_res) or (lig_charge < 0 and rname in pos_res)) and min_d <= 4.0:
                        itype = 'salt'
                    else:
                        # 判定 π-阳离子（pi-cation）：
                        # 1) 蛋白正电残基（LYS/ARG） + 配体含芳香；或 2) 蛋白芳香残基 + 配体净正电
                        if ((rname in pos_res and lig_has_aromatic) or (rname in aromatic_res and lig_charge > 0)) and min_d <= 6.0:
                            itype = 'pi_cation'
                        else:
                    # 判定疏水：C…C 距离
                            if rname in hydro_res:
                                for (px,py,pz,pe) in coords:
                                    if pe != 'C':
                                        continue
                                    for (lx,ly,lz,le) in lig_atoms:
                                        if le != 'C':
                                            continue
                                        if math.dist((px,py,pz),(lx,ly,lz)) <= 4.5:
                                            itype = 'hydrophobic'; break
                                    if itype=='hydrophobic':
                                        break
                    # 判定 π：芳香残基 + 配体芳香
                            if itype=='vdw' and rname in aromatic_res and lig_has_aromatic and min_d <= 5.0:
                                itype = 'pi'

            neighbors.append((res, min_d, itype))

        # 水桥：HOH O 同时接近配体(N/O)与蛋白(N/O)
        if water_oxygens:
            for res, coords in prot_atoms.items():
                # 最接近的蛋白 N/O 与 HOH O
                best = None; best_metric = 1e9
                for (wx,wy,wz) in water_oxygens:
                    # 蛋白侧 N/O
                    min_p = min((math.dist((wx,wy,wz),(px,py,pz)) for (px,py,pz,pe) in coords if pe in {'N','O'}), default=1e9)
                    # 配体 N/O
                    min_l = min((math.dist((wx,wy,wz),(lx,ly,lz)) for (lx,ly,lz,le) in lig_heavy_NO), default=1e9)
                    if min_p <= 3.2 and min_l <= 3.2:
                        metric = min_p + min_l
                        if metric < best_metric:
                            best_metric = metric; best = (min_p, min_l)
                if best:
                    neighbors.append((res, sum(best)/2.0, 'water'))

        # 金属配位：金属–配体重原子（N/O/S）靠近
        if metals:
            added_metal_nodes = set()
            for elem, (mx,my,mz) in metals:
                dmin = min((math.dist((mx,my,mz),(lx,ly,lz)) for (lx,ly,lz,le) in lig_atoms if le in {'N','O','S'}), default=1e9)
                if dmin <= 2.6:
                    label = f"METAL:{elem}"
                    if label not in added_metal_nodes:
                        neighbors.append((label, dmin, 'metal'))
                        added_metal_nodes.add(label)

        if not neighbors:
            return None

        G = nx.Graph()
        ligand_node = 'LIG'
        G.add_node(ligand_node, kind='ligand')
        for res, _, _ in neighbors:
            G.add_node(res, kind='residue')
            G.add_edge(ligand_node, res)

        # 布局：环形 + 中心
        residues = [n for n, d in G.nodes(data=True) if d.get('kind') == 'residue']
        pos = nx.circular_layout(residues, scale=2.5)
        pos[ligand_node] = (0.0, 0.0)

        plt.figure(figsize=figsize)
        ax = plt.gca()
        # 在中心叠加 RDKit 2D 分子图（先画分子，再画连线，避免分子被遮挡）
        if HAS_RDKIT and smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    AllChem.Compute2DCoords(mol)
                    # 尝试高亮芳香环原子，并根据 figsize 适配像素尺寸
                    w = max(320, int(100 * figsize[0]))
                    h = max(320, int(100 * figsize[1]))
                    hl = list(lig_aromatic_indices) if 'lig_aromatic_indices' in locals() else []
                    img = Draw.MolToImage(mol, size=(w, h), highlightAtoms=hl)
                    imagebox = OffsetImage(img, zoom=0.62)
                    ab = AnnotationBbox(imagebox, (0.0, 0.0), frameon=False)
                    ab.set_zorder(0)
                    ax.add_artist(ab)
                except Exception:
                    pass

        # 类型样式定义
        # 线型与颜色（加长虚线/点划使高分辨率下更明显）
        style_map = {
            'hbond': ('#2e7d32', (12,6)),          # 绿色长虚线
            'salt': ('#c62828', None),              # 红色实线
            'pi': ('#8e24aa', (8,4,2,4)),          # 紫色点划
            'pi_T': ('#f06292', (10,3,3,3)),       # 粉色点划（T形）
            'pi_cation': ('#00acc1', (10,5,2,5)),  # 青色点划
            'pi_anion': ('#42a5f5', (10,5)),       # 蓝色长虚线
            'pi_alkyl': ('#ba68c8', (14,6)),       # 紫色长虚线
            'halogen': ('#ef6c00', (14,4)),        # 橙色长虚线
            'chb': ('#b0bec5', (6,8)),             # 碳氢键：稀疏虚线
            'water': ('#0277bd', (6,6)),           # 水桥：等长虚线
            'metal': ('#455a64', None),            # 金属：深灰实线
            'hydrophobic': ('#6d4c41', None),      # 棕色实线
            'vdw': ('#90a4ae', None),              # 灰蓝实线
        }
        # 线宽（增强区分度）
        width_map = {
            'hbond': 3.2, 'salt': 3.4,
            'pi': 3.0, 'pi_T': 3.0, 'pi_cation': 3.0, 'pi_anion': 3.0, 'pi_alkyl': 3.0,
            'halogen': 3.2, 'chb': 2.6, 'water': 2.6, 'metal': 3.4,
            'hydrophobic': 2.8, 'vdw': 2.4,
        }
        node_fc_map = {
            'hbond': '#c8e6c9',       # 绿浅
            'salt': '#ffccbc',        # 橙浅
            'pi': '#e1bee7',          # 紫浅
            'pi_T': '#f8bbd0',        # 粉浅
            'pi_cation': '#b2ebf2',   # 青浅
            'pi_anion': '#bbdefb',    # 蓝浅
            'pi_alkyl': '#e1bee7',    # 紫浅
            'halogen': '#ffe0b2',     # 橙浅
            'chb': '#cfd8dc',         # 蓝灰浅
            'water': '#b3e5fc',       # 天蓝浅
            'metal': '#cfd8dc',       # 金属灰浅
            'hydrophobic': '#ffe0b2', # 棕浅
            'vdw': '#aed581',         # 绿
        }

        # 边与距离标注（H-bond/盐桥：虚/实线 + 叠加箭头；其他：按线型）
        type_code_map = {
            'hbond': 'HB', 'salt': 'SALT', 'pi': 'PI', 'pi_T': 'PI-T', 'pi_cation': 'PI-CAT',
            'pi_anion': 'PI-ANION', 'pi_alkyl': 'PI-ALK', 'halogen': 'HAL', 'chb': 'CHB',
            'hydrophobic': 'HYD', 'vdw': 'VDW'
        }
        # 统计各类型数量用于图例
        type_counts: Dict[str, int] = {}
        for res, dist, itype in neighbors:
            color, dashes = style_map.get(itype, ('#616161', None))
            x1, y1 = pos[res]
            x0, y0 = pos[ligand_node]
            # 基线（确保与图例一致的线型）
            if dashes:
                ls = (0, dashes)
            else:
                ls = '-'
            lw = width_map.get(itype, 2.6)
            line_handle = ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, linestyle=ls, zorder=2, solid_capstyle='round')[0]
            # 外描边，避免在缩略图中被背景/节点淹没
            line_handle.set_path_effects([pe.Stroke(linewidth=lw+1.2, foreground='white'), pe.Normal()])
            # 箭头（仅 hbond/salt 叠加箭头头部）
            if itype in ('hbond', 'salt'):
                ax.annotate('', xy=(x0, y0), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=0, shrinkA=0, shrinkB=0, mutation_scale=16),
                            zorder=3)
            # 距离标注
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            code = type_code_map.get(itype, itype)
            ax.text(mx, my, f"{dist:.2f}Å\n{code}", fontsize=8, color=color, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7), zorder=4)
            type_counts[itype] = type_counts.get(itype, 0) + 1

        # 先画节点：配体与不同类型的残基用不同底色（置于边之上）
        nx.draw_networkx_nodes(G, pos, nodelist=[ligand_node], node_color='#1976d2', node_size=740, ax=ax, zorder=3, alpha=0.95)
        res_by_type: Dict[str, List[str]] = {'hbond': [], 'salt': [], 'pi': [], 'pi_T': [], 'pi_cation': [], 'pi_anion': [], 'pi_alkyl': [], 'halogen': [], 'chb': [], 'water': [], 'metal': [], 'hydrophobic': [], 'vdw': []}
        for r, _, t in neighbors:
            res_by_type.setdefault(t, []).append(r)
        for t, lst in res_by_type.items():
            if not lst:
                continue
            nx.draw_networkx_nodes(G, pos, nodelist=lst, node_color=node_fc_map.get(t, '#bbdefb'), node_size=540,
                                   edgecolors='#78909c', linewidths=1.0, ax=ax, zorder=3, alpha=0.93)

        # 标签
        labels = {ligand_node: ligand_node}
        labels.update({r: r for r in residues})
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax, zorder=4)

        # 图例
        import matplotlib.lines as mlines
        # 仅展示实际出现的类型，避免“图例有而图中无”的困惑
        legend_spec = [
            ('Conventional Hydrogen Bond','hbond'),
            ('Salt Bridge','salt'),
            ('Pi-Pi Stacking','pi'),
            ('Pi-T shaped','pi_T'),
            ('Pi-Cation','pi_cation'),
            ('Pi-Anion','pi_anion'),
            ('Pi-Alkyl','pi_alkyl'),
            ('Halogen Bond','halogen'),
            ('Carbon Hydrogen Bond','chb'),
            ('Water Bridge','water'),
            ('Metal Coordination','metal'),
            ('Hydrophobic/Alkyl','hydrophobic'),
            ('van der Waals','vdw'),
        ]
        present = {t for _,_,t in neighbors}
        used = [(k,d) for (k,d) in legend_spec if d in present]
        if not used:
            used = [('van der Waals','vdw')]
        handles = []
        labels_ = []
        for k, d in used:
            col, dash = style_map[d]
            ln = mlines.Line2D([], [], color=col, linewidth=width_map.get(d, 2.6))
            if dash:
                ln.set_dashes(dash)
            handles.append(ln)
            cnt = type_counts.get(d)
            labels_.append(f"{k}{' ('+str(cnt)+')' if cnt else ''}")
        ax.legend(
            handles,
            labels_,
            loc='upper right',
            frameon=True,
            framealpha=0.9,
            facecolor='white',
            edgecolor='#e0e0e0',
            fontsize=9,
            fancybox=True,
            borderpad=0.6,
            handlelength=2.8,
            handletextpad=0.8,
            labelspacing=0.5,
        )

        ax.set_title(title + f"  (<= {cutoff} Å)")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(str(out_png), dpi=dpi, bbox_inches='tight', pad_inches=0.15)
        plt.close()
        return out_png
    except Exception:
        return None
def collect_receptor_chains(complex_pdb: Path) -> List[str]:
    """从复合物 PDB 中收集作为受体的链 ID：排除配体链 'L'。"""
    chains: List[str] = []
    seen = set()
    with open(complex_pdb, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and len(line) >= 22:
                ch = line[21].strip()
                if ch and ch != 'L' and ch not in seen:
                    seen.add(ch)
                    chains.append(ch)
    return chains
def merge_protein_ligand(protein_pdb: Path, ligand_pdb: Path, complex_out: Path) -> Path:
    """合并蛋白与配体坐标为一个复合物 PDB。
    - 去除蛋白中的 END/MASTER/MODEL/ENDMDL 等结束记录
    - 仅写入配体的 ATOM/HETATM 记录，并标准化为 LIG/L/1
    - 文件最后仅写入一次 TER 与 END
    """
    protein_lines: List[str] = []
    with open(protein_pdb, 'r') as fp:
        for line in fp:
            rec = line[:6]
            if rec.startswith('END') or rec.startswith('MAST') or rec.startswith('MODEL') or rec.startswith('ENDMDL'):
                continue
            protein_lines.append(line)

    ligand_lines: List[str] = []
    with open(ligand_pdb, 'r') as fl:
        for line in fl:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            buf = list(line.rstrip('\n'))
            if len(buf) < 80:
                buf += [' '] * (80 - len(buf))
            # 记录类型 HETATM
            het = list('HETATM')
            for i in range(6):
                buf[i] = het[i]
            # RESNAME = LIG（列 18-20）
            res = list('LIG')
            for i, ch in enumerate(res):
                buf[17 + i] = ch
            # CHAIN = L（列 22）
            buf[21] = 'L'
            # RESSEQ = 1（列 23-26）
            resseq = f"{1:>4}"
            for i, ch in enumerate(resseq):
                buf[22 + i] = ch
            # 元素列（列 77-78）
            atom_name = ''.join(buf[12:16]).strip()
            elem = (atom_name[0] if atom_name else 'C').upper().rjust(2)
            buf[76], buf[77] = elem[0], elem[1]
            ligand_lines.append(''.join(buf) + '\n')

    with open(complex_out, 'w') as fo:
        for line in protein_lines:
            fo.write(line)
        fo.write('TER\n')
        for line in ligand_lines:
            fo.write(line)
        fo.write('TER\nEND\n')
    return complex_out


def run_plip(complex_pdb: Path, out_dir: Path) -> Tuple[bool, Optional[Path]]:
    plip_exec = which('plip')
    if not plip_exec:
        print("[ERR] 未找到 plip 可执行文件。请先安装: pip install plip （建议装入项目虚拟环境）")
        return False, None
    ensure_dir(out_dir)
    cmd = [plip_exec, '-f', str(complex_pdb), '-o', str(out_dir), '-x', '-t']
    # 只有在 PyMOL 模块与可执行程序均可用时才生成图片
    has_pymol_bin = which('pymol') is not None
    if HAS_PYMOL and has_pymol_bin:
        cmd.append('-p')
    # 明确指定受体/配体链，提高识别稳定性
    rec_chains = collect_receptor_chains(complex_pdb)
    if rec_chains:
        chains_repr = "[[" + ", ".join([f"'{c}'" for c in rec_chains]) + "], ['L']]"
        cmd.extend(['--chains', chains_repr])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[ERR] PLIP 运行失败: {complex_pdb}: {e}")
        return False, None

    # 寻找输出的 2D 图（PLIP 生成 png/svg，目录名通常为 report_* 或 name_ligand_*）
    pngs = list(out_dir.glob('**/*.png'))
    svgs = list(out_dir.glob('**/*.svg'))
    figure = pngs[0] if pngs else (svgs[0] if svgs else None)
    return True, figure


def load_csv_props(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not csv_path or not csv_path.exists():
        return mapping
    df = pd.read_csv(csv_path)
    if 'compound_id' not in df.columns or 'smiles' not in df.columns:
        return mapping
    for _, r in df.iterrows():
        cid = str(r.get('compound_id', '')).strip()
        if not cid:
            continue
        mapping[cid] = {
            'smiles': str(r.get('smiles', '')),
            'binding_affinity': str(r.get('binding_affinity', 'N/A')) if 'binding_affinity' in df.columns else 'N/A',
            'molecular_weight': str(r.get('molecular_weight', 'N/A')) if 'molecular_weight' in df.columns else 'N/A',
            'logp': str(r.get('logp', 'N/A')) if 'logp' in df.columns else 'N/A',
        }
    return mapping


def build_index_html(cards: List[str], out_dir: Path, title: str = '蛋白-配体 2D 相互作用图（PLIP）') -> Path:
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  {HTML_STYLE}
</head>
<body>
  <div class=\"container\">
    <h1>{title}</h1>
    <div class=\"info\">共 {len(cards)} 个条目</div>
    <div class=\"grid\">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""
    out = out_dir / 'index.html'
    out.write_text(html, encoding='utf-8')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protein', required=True, help='蛋白 PDB 路径')
    ap.add_argument('--ligand-dir', required=True, help='配体文件目录（支持 pdb/pdbqt/sdf/mol2）')
    ap.add_argument('--out-dir', required=True, help='输出目录，如 results/.../visualization_plip')
    ap.add_argument('--csv', default=None, help='可选：包含 compound_id, smiles 等列的 CSV，用于属性补全与链接')
    ap.add_argument('--top-n', type=int, default=None, help='只处理前 N 个配体（按文件名排序）')
    ap.add_argument('--render-3d', action='store_true', help='使用 PyMOL 渲染 3D 对接图（卡通+表面+sticks+虚线氢键）')
    ap.add_argument('--pymol-size', default='1600,1200', help='3D PNG 尺寸 像素W,H，默认 1600,1200')
    ap.add_argument('--fast-3d', action='store_true', help='快速预览：ray=0，lower surface quality，defer builds')
    ap.add_argument('--label-top-k', type=int, default=6, help='3D 图中标注的残基数量上限，默认 6（仅标注最近的若干残基）')
    ap.add_argument('--top-by-affinity', type=int, default=None, help='按 CSV 中 binding_affinity 升序选择前K（更负更优）；需提供 --csv')
    ap.add_argument('--run-dir', default=None, help='可选：run 目录（用于自动关联 visualization_2d / visualization_3d 页面）')
    ap.add_argument('--dist-cutoff', type=float, default=4.5, help='距离回退示意图的邻近阈值(Å)，默认4.5')
    ap.add_argument('--dpi', type=int, default=300, help='导出 PNG 的分辨率 DPI，默认 300')
    ap.add_argument('--figsize', default='8,8', help='图像尺寸(英寸)，如 8,8 或 8x8，默认 8,8')
    ap.add_argument('--strict-angles', action='store_true', default=True, help='启用角度约束（氢键/卤键等），默认开启')
    ap.add_argument('--no-strict-angles', action='store_false', dest='strict_angles', help='关闭角度约束')
    ap.add_argument('--hb-angle-min', type=float, default=140.0, help='氢键最小夹角阈值(度)，默认140')
    ap.add_argument('--hal-angle-min', type=float, default=150.0, help='卤键最小夹角阈值(度)，默认150')
    ap.add_argument('--export-dir', default=None, help='可选：导出所有生成 PNG 的目录（将拷贝 figures/*.png 到此目录）')
    ap.add_argument('--zip-figs', action='store_true', help='可选：将导出目录内的 PNG 打包为 zip')
    args = ap.parse_args()

    protein = Path(args.protein).expanduser().resolve()
    # 解析 --ligand-dir 参数
    lig_dir = Path(args.ligand_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else (out_dir.parent if out_dir.parent.exists() else None)

    if not protein.exists():
        raise SystemExit(f"蛋白 PDB 不存在: {protein}")
    if not lig_dir.exists():
        raise SystemExit(f"配体目录不存在: {lig_dir}")
    ensure_dir(out_dir)
    complex_dir = ensure_dir(out_dir / 'complexes')
    figures_dir = ensure_dir(out_dir / 'figures')

    props_map = load_csv_props(csv_path)
    dist_cutoff = args.dist_cutoff
    dpi = int(args.dpi)
    figsize = parse_figsize(args.figsize)
    strict_angles = bool(args.strict_angles)
    hb_angle_min = float(args.hb_angle_min)
    hal_angle_min = float(args.hal_angle_min)

    # 收集配体文件
    lig_files: List[Path] = []
    for ext in ('*.pdb', '*.pdbqt', '*.sdf', '*.mol2'):
        lig_files.extend(sorted(lig_dir.glob(ext)))
    if not lig_files:
        raise SystemExit(f"在 {lig_dir} 未找到配体文件（pdb/pdbqt/sdf/mol2）")
    # 若指定按结合能筛选，则基于 CSV 的 binding_affinity 值排序后取前K
    if args.top_by_affinity:
        if not props_map:
            print('[WARN] --top-by-affinity 需要 --csv 且包含 binding_affinity 列，已忽略此选项。')
        else:
            scored: List[Tuple[float, Path]] = []
            tail: List[Path] = []
            for lf in lig_files:
                cid0 = guess_compound_id(lf)
                val = parse_affinity_value(props_map.get(cid0, {}).get('binding_affinity'))
                if val is None:
                    tail.append(lf)
                else:
                    scored.append((val, lf))
            scored.sort(key=lambda t: t[0])  # 越小越优（更负更好）
            selected = [lf for _, lf in scored][: int(args.top_by_affinity)]
            # 若不足K，可用无分值者补齐
            if len(selected) < int(args.top_by_affinity):
                need = int(args.top_by_affinity) - len(selected)
                selected += tail[:need]
            lig_files = selected
    elif args.top_n:
        lig_files = lig_files[: args.top_n]

    # 准备页面卡片
    cards: List[str] = []

    for lf in lig_files:
        cid = guess_compound_id(lf)
        # 预取属性（供回退出图叠加 RDKit 2D 分子等使用）
        smiles = props_map.get(cid, {}).get('smiles', '')
        aff_txt = props_map.get(cid, {}).get('binding_affinity', 'N/A')
        lig_pdb = figures_dir / f"{cid}.pdb"
        # 1) 统一得到 ligand PDB
        if lf.suffix.lower() == '.pdb':
            # 若源与目标为同一路径，避免 SameFileError，直接原地规范化
            try:
                same_path = False
                try:
                    same_path = lf.resolve() == lig_pdb.resolve()
                except Exception:
                    same_path = str(lf) == str(lig_pdb)
                if not same_path:
                    shutil.copyfile(lf, lig_pdb)
                else:
                    lig_pdb = lf
            except Exception:
                lig_pdb = lf
            normalize_ligand_pdb(lig_pdb)
        elif lf.suffix.lower() == '.pdbqt':
            if HAS_OBABEL:
                lig_pdb = convert_with_openbabel(lf, lig_pdb)
            else:
                lig_pdb = pdbqt_to_pdb(lf, lig_pdb)
                normalize_ligand_pdb(lig_pdb)
        elif lf.suffix.lower() in ('.sdf', '.mol2') and HAS_RDKIT:
            # RDKit 转为 PDB
            mols = Chem.SDMolSupplier(str(lf), removeHs=False) if lf.suffix.lower()=='.sdf' else [Chem.MolFromMol2File(str(lf), sanitize=False)]
            mol = None
            for m in mols:
                if m is not None:
                    mol = m; break
            if mol is None:
                print(f"[WARN] 无法读取配体: {lf}");
                continue
            Chem.MolToPDBFile(mol, str(lig_pdb))
            normalize_ligand_pdb(lig_pdb)
        else:
            print(f"[WARN] 不支持的配体格式或缺少 RDKit: {lf}")
            continue

        # 2) 合并为复合物
        complex_pdb = complex_dir / f"{cid}_complex.pdb"
        merge_protein_ligand(protein, lig_pdb, complex_pdb)

        # 3) 运行 PLIP（生成 2D 交互图备用）
        lig_out_dir = ensure_dir(out_dir / f"plip_{cid}")
        ok, fig = run_plip(complex_pdb, lig_out_dir)
        # 复制一份缩略图（即使 PLIP 失败也保留占位图）
        thumb_rel = None
        if fig and fig.exists():
            # 统一导出命名：*_plip.png
            plip_png = figures_dir / f"{cid}_plip.png"
            try:
                if str(fig.suffix).lower() == '.png':
                    shutil.copyfile(fig, plip_png)
                elif str(fig.suffix).lower() == '.svg':
                    conv_ok = False
                    try:
                        if HAS_CAIROSVG:
                            cairosvg.svg2png(url=str(fig), write_to=str(plip_png))  # type: ignore
                            conv_ok = True
                    except Exception:
                        conv_ok = False
                    if not conv_ok:
                        # 回退：若无法转换，则仍复制为 .png 名称（部分查看器兼容），失败则忽略
                        try:
                            shutil.copyfile(fig, plip_png)
                        except Exception:
                            pass
                else:
                    shutil.copyfile(fig, plip_png)
                if plip_png.exists():
                    thumb_rel = f"figures/{plip_png.name}"
            except Exception:
                thumb_rel = None
        else:
            # 回退一：使用 report.xml 生成 2D 交互网络图
            xml_report = lig_out_dir / 'report.xml'
            fallback_png = figures_dir / f"{cid}_plip.png"
            gen = None
            if xml_report.exists():
                gen = draw_interaction_map_from_xml(xml_report, fallback_png, f"{cid} interactions", dpi=dpi, figsize=figsize)
            # 回退二：基于几何邻近（若 XML 为空或不存在）
            if (not gen or not gen.exists()):
                gen2 = draw_interaction_map_by_distance(
                    complex_pdb,
                    fallback_png,
                    f"{cid} interactions",
                    cutoff=dist_cutoff,
                    smiles=smiles,
                    dpi=dpi,
                    figsize=figsize,
                    strict_angles=strict_angles,
                    hb_angle_min=hb_angle_min,
                    hal_angle_min=hal_angle_min,
                )
                if gen2 and gen2.exists():
                    thumb_rel = f"figures/{fallback_png.name}"

        # 3D PyMOL 渲染（函数内部会在无 Python 模块时回退到 CLI）
        if getattr(args, 'render_3d', False):
            size_src = str(getattr(args, 'pymol_size', '1600,1200'))
            # fast 模式，且用户未显式改尺寸时，改用 1200x900 加速
            if getattr(args, 'fast_3d', False) and size_src.strip().lower().replace('x', 'x') in ('1600,1200', '1600x1200', '1600 1200'):
                size_src = '1200,900'
            size_txt = size_src.lower().replace('x', ',').replace(' ', ',')
            try:
                parts = [int(p) for p in size_txt.split(',') if p][:2]
                w, h = (parts + [1600, 1200])[:2]
            except Exception:
                w, h = 1600, 1200
            out3d = figures_dir / f"{cid}_3d.png"
            img3d = render_3d_docking_pymol(
                complex_pdb,
                out3d,
                width=w,
                height=h,
                surface_transparency=0.6,
                cartoon_transparency=0.18,
                contact_cutoff=4.2,
                fast=bool(getattr(args, 'fast_3d', False)),
                label_top_k=int(getattr(args, 'label_top_k', 6)),
            )
            if img3d and img3d.exists():
                # 叠加 SMILES 与结合能到左上角
                try:
                    overlay_header_on_png(out3d, smiles, aff_txt)
                except Exception:
                    pass
                thumb_rel = f"figures/{out3d.name}"

        # 属性与链接
        ba = props_map.get(cid, {}).get('binding_affinity', 'N/A')
        mw = props_map.get(cid, {}).get('molecular_weight', 'N/A')
        logp = props_map.get(cid, {}).get('logp', 'N/A')

        link_2d = None
        link_3d = None
        if run_dir and (run_dir / 'visualization_2d').exists():
            cand = run_dir / 'visualization_2d' / f"{cid}_2D.html"
            if cand.exists():
                link_2d = os.path.relpath(cand, out_dir)
        if run_dir and (run_dir / 'visualization_3d').exists():
            # 选首个包含 cid 的 3D 页面
            for f in (run_dir / 'visualization_3d').glob('*_3D.html'):
                if cid in f.name:
                    link_3d = os.path.relpath(f, out_dir)
                    break

        # 4) 生成卡片（带缓存破除参数）
        if thumb_rel:
            fig_path = (out_dir / thumb_rel) if not Path(thumb_rel).is_absolute() else Path(thumb_rel)
            ver = int(fig_path.stat().st_mtime) if fig_path.exists() else int(time.time())
            img_src = f"{thumb_rel}?v={ver}"
            img_tag = f'<img class="thumb" src="{img_src}">'
        else:
            img_tag = '<div class="thumb" style="height:220px;display:flex;align-items:center;justify-content:center;color:#888;">(无图)</div>'
        btn_2d = f'<a class="btn" href="{link_2d}">2D 分子</a>' if link_2d else ''
        btn_3d = f'<a class="btn green" href="{link_3d}">3D 结构</a>' if link_3d else ''
        btn_plip = f'<a class="btn gray" href="{os.path.relpath(lig_out_dir, out_dir)}">PLIP 输出</a>'

        card = f"""
        <div class="card">
          <h3>{cid}</h3>
          {img_tag}
          <div class="meta">
            <div class="row"><strong>SMILES:</strong> {smiles[:80]}{'...' if len(smiles)>80 else ''}</div>
            <div class="row"><strong>Binding Affinity:</strong> {ba} | <strong>MW:</strong> {mw} | <strong>LogP:</strong> {logp}</div>
          </div>
          <div class="row">{btn_2d}{btn_3d}{btn_plip}</div>
        </div>
        """
        cards.append(card)

    index = build_index_html(cards, out_dir)
    print(f"[OK] PLIP 相互作用图生成完成：{out_dir}\n - 总览：{index}")

    # 可选：导出 PNG 并打包
    export_dir_arg = getattr(args, 'export_dir', None)
    if export_dir_arg:
        exp_dir = Path(export_dir_arg).expanduser().resolve()
        ensure_dir(exp_dir)
        copied: List[Path] = []
        for png in sorted((out_dir / 'figures').glob('*.png')):
            dst = exp_dir / png.name
            try:
                shutil.copyfile(png, dst)
                copied.append(dst)
            except Exception as e:
                print(f"[WARN] 拷贝失败 {png} -> {dst}: {e}")
        print(f"[OK] 已导出 {len(copied)} 张图片到：{exp_dir}")
        if getattr(args, 'zip_figs', False) and copied:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_path = exp_dir / f"plip_figures_{ts}.zip"
            try:
                with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                    for p in copied:
                        zf.write(str(p), arcname=p.name)
                print(f"[OK] 导出压缩包：{zip_path}")
            except Exception as e:
                print(f"[WARN] 打包失败：{e}")


if __name__ == '__main__':
    main()
