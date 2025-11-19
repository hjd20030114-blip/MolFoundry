#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 可视化模块
将 CSV 中的分子生成与 3D 页面风格一致的 2D HTML 页面与总览 index。

用法示例：
python HJD/scripts/visualization_2d.py \
  --csv HJD/results/run_20251004_001/ligands/dl_phase3_optimized_molecules.csv \
  --out-dir HJD/results/run_20251004_001/visualization_2d \
  --top-n 20

依赖：rdkit-pypi
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem import rdMolDraw2D, Draw
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

HTML_STYLE = """
<style>
 body { font-family: Arial, sans-serif; margin: 20px; }
 .container { max-width: 1000px; margin: 0 auto; }
 .info { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
 .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
 .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
 .card h3 { margin: 0 0 8px 0; color: #2c3e50; }
 .props { color: #555; font-size: 14px; line-height: 1.6; }
 .link { margin-top: 10px; }
 .btn { display:inline-block; padding:8px 12px; background:#1976d2; color:#fff; border-radius:4px; text-decoration:none; }
 .btn:visited { color:#fff; }
 .img-box { text-align:center; }
 img.mol { width: 100%; height: auto; border-radius: 4px; border: 1px solid #eee; }
</style>
"""


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_basic_props(smiles: str) -> Dict[str, str]:
    if not HAS_RDKIT:
        return {"formula": "N/A", "mw": "N/A", "logp": "N/A", "tpsa": "N/A"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"formula": "N/A", "mw": "N/A", "logp": "N/A", "tpsa": "N/A"}
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = "N/A"
    try:
        mw = f"{Descriptors.MolWt(mol):.2f}"
    except Exception:
        mw = "N/A"
    try:
        logp = f"{Crippen.MolLogP(mol):.2f}"
    except Exception:
        logp = "N/A"
    try:
        tpsa = f"{rdMolDescriptors.CalcTPSA(mol):.2f}"
    except Exception:
        tpsa = "N/A"
    return {"formula": formula, "mw": mw, "logp": logp, "tpsa": tpsa}


def draw_mol_image(smiles: str, out_path: Path, size: int = 500) -> Optional[Path]:
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        if hasattr(rdMolDraw2D, 'MolDraw2DCairo'):
            drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            with open(out_path.with_suffix('.png'), 'wb') as f:
                f.write(drawer.GetDrawingText())
            return out_path.with_suffix('.png')
        else:
            drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            with open(out_path.with_suffix('.svg'), 'w', encoding='utf-8') as f:
                f.write(svg)
            return out_path.with_suffix('.svg')
    except Exception:
        try:
            img = Draw.MolToImage(mol, size=(size, size))
            img.save(out_path.with_suffix('.png'))
            return out_path.with_suffix('.png')
        except Exception:
            return None


def per_molecule_page(compound_id: str, smiles: str, props: Dict[str, str], image_rel: str, out_dir: Path, link_3d: Optional[str] = None) -> Path:
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{compound_id} - 2D分子结构</title>
  {HTML_STYLE}
</head>
<body>
  <div class="container">
    <h1>{compound_id} - 2D分子结构</h1>
    <div class="info">
      <strong>SMILES:</strong> {smiles}<br>
      <strong>分子式:</strong> {props.get('formula','N/A')}<br>
      <strong>分子量:</strong> {props.get('mw','N/A')} Da<br>
      <strong>LogP:</strong> {props.get('logp','N/A')} | <strong>TPSA:</strong> {props.get('tpsa','N/A')}
    </div>
    <div class="card img-box">
      <img class="mol" src="{image_rel}" alt="{compound_id}">
    </div>
    {f'<div class="link"><a class="btn" href="../visualization_3d/{Path(link_3d).name}">查看 3D 结构</a></div>' if link_3d else ''}
  </div>
</body>
</html>
"""
    out_file = out_dir / f"{compound_id}_2D.html"
    out_file.write_text(html, encoding='utf-8')
    return out_file


def index_page(entries: List[Dict[str, str]], out_dir: Path, title: str = "2D 分子画廊") -> Path:
    cards: List[str] = []
    for e in entries:
        link3d_html = ""
        if e.get('link_3d'):
            link3d_html = f" <a class=\"btn\" style=\"background:#43a047; margin-left:8px;\" href=\"{e['link_3d']}\">3D 页面</a>"

        # 单卡片 HTML
        card_html = (
            f"""
        <div class=\"card\">
          <h3>{e['compound_id']}</h3>
          <div class=\"img-box\"><img class=\"mol\" src=\"{e['image_file']}\" alt=\"{e['compound_id']}\"></div>
          <div class=\"props\"> 
            <div><strong>SMILES:</strong> {e['smiles'][:64]}{'...' if len(e['smiles'])>64 else ''}</div>
            <div><strong>Binding Affinity:</strong> {e.get('binding_affinity','N/A')}</div>
            <div><strong>MW:</strong> {e.get('mw','N/A')} | <strong>LogP:</strong> {e.get('logp','N/A')}</div>
          </div>
          <div class=\"link\"><a class=\"btn\" href=\"{e['page']}\">打开 2D 页面</a>{link3d_html}</div>
        </div>
        """
        )
        cards.append(card_html)

    cards_html = "\n".join(cards)
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
    <h1>🧪 {title}</h1>
    <div class=\"info\">共 {len(entries)} 个分子条目</div>
    <div class=\"grid\">
      {cards_html}
    </div>
  </div>
</body>
</html>
"""

    out_file = out_dir / "index.html"
    out_file.write_text(html, encoding='utf-8')
    return out_file


def guess_3d_html(run_dir: Path, compound_id: str) -> Optional[str]:
    viz3d_dir = run_dir / "visualization_3d"
    if not viz3d_dir.exists():
        return None
    # 寻找形如 Top_*_{compound_id}_3D.html 或包含 compound_id 的文件
    for f in viz3d_dir.glob("*_3D.html"):
        if compound_id in f.name:
            return f.name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="含有分子结果的CSV（需含 smiles、compound_id 等列）")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认与CSV同级建 visualization_2d/")
    ap.add_argument("--top-n", type=int, default=20, help="生成前N个分子的2D页面（按排序字段）")
    ap.add_argument("--sort-by", default=None, help="排序字段，默认优先 binding_affinity（升序）")
    ap.add_argument("--descending", action="store_true", help="是否降序排序（默认升序）")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV 不存在: {csv_path}")

    # 推断 run 目录与输出目录
    run_dir = csv_path.parent.parent if csv_path.parent.name == 'ligands' else csv_path.parent
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_dir / "visualization_2d")
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise SystemExit("CSV 缺少 smiles 列")
    if 'compound_id' not in df.columns:
        # 尝试用索引占位
        df['compound_id'] = [f"mol_{i+1}" for i in range(len(df))]

    # 选择排序字段
    sort_by = args.sort_by
    if sort_by is None:
        if 'binding_affinity' in df.columns:
            sort_by = 'binding_affinity'
        elif 'pred_binding_affinity' in df.columns:
            sort_by = 'pred_binding_affinity'
        else:
            sort_by = 'compound_id'
    ascending = not args.descending

    try:
        df_sorted = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    except Exception:
        df_sorted = df.reset_index(drop=True)

    if args.top_n and len(df_sorted) > args.top_n:
        df_sorted = df_sorted.head(args.top_n)

    entries = []
    for _, row in df_sorted.iterrows():
        smi = str(row.get('smiles', '')).strip()
        cid = str(row.get('compound_id', 'mol'))
        if not smi:
            continue
        # 生成图片
        img_path = draw_mol_image(smi, out_dir / cid, size=520)
        if img_path is None:
            continue
        # 计算属性（补全缺失）
        props = compute_basic_props(smi)
        mw = row.get('molecular_weight', props['mw'])
        logp = row.get('logp', props['logp'])
        # 生成 2D 页面
        link_3d_name = guess_3d_html(run_dir, cid)
        page = per_molecule_page(
            compound_id=cid,
            smiles=smi,
            props={**props, 'mw': str(mw), 'logp': str(logp)},
            image_rel=os.path.basename(img_path),
            out_dir=out_dir,
            link_3d=(str((run_dir/"visualization_3d"/link_3d_name)) if link_3d_name else None)
        )
        entries.append({
            'compound_id': cid,
            'smiles': smi,
            'image_file': os.path.basename(img_path),
            'binding_affinity': row.get('binding_affinity', 'N/A'),
            'mw': mw,
            'logp': logp,
            'page': os.path.basename(page),
            'link_3d': (f"../visualization_3d/{link_3d_name}" if link_3d_name else None)
        })

    # 生成 index
    idx = index_page(entries, out_dir, title=f"2D 分子画廊（源自 {csv_path.name}）")
    print(f"[OK] 2D 分子页面生成完成：{out_dir}\n - 总览：{idx}")


if __name__ == "__main__":
    main()
