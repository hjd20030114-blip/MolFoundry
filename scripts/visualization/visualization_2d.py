#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 可视化模块
- 支持在无 Cairo 的环境下生成 2D 分子图（优先 SVG，回退 PIL）
- 提供单分子渲染与画廊生成功能
"""

import os
import logging
from typing import Dict, List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

logger = logging.getLogger(__name__)


class Visualizer2D:
    """2D 可视化器"""

    def __init__(self, output_dir: str = "visualization_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def render_molecule(self, smiles: str, base_name: str, size: int = 300) -> Optional[str]:
        """
        渲染单个分子为图片（PNG 或 SVG）
        - 优先使用 rdMolDraw2D + Cairo(PNG)
        - 否则使用 rdMolDraw2D SVG
        - 否则回退到 PIL 的 PNG
        返回输出文件路径或 None
        """
        if not HAS_RDKIT:
            logger.warning("RDKit 不可用，无法渲染 2D 分子图")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # 优先 rdMolDraw2D
            try:
                from rdkit.Chem import rdMolDraw2D
                if hasattr(rdMolDraw2D, "MolDraw2DCairo"):
                    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
                    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                    drawer.FinishDrawing()
                    img_path = os.path.join(self.output_dir, f"{base_name}.png")
                    with open(img_path, "wb") as f:
                        f.write(drawer.GetDrawingText())
                    return img_path
                else:
                    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
                    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                    drawer.FinishDrawing()
                    svg = drawer.GetDrawingText()
                    img_path = os.path.join(self.output_dir, f"{base_name}.svg")
                    with open(img_path, "w", encoding="utf-8") as f:
                        f.write(svg)
                    return img_path
            except Exception:
                # 回退 PIL
                try:
                    pil_img = Draw.MolToImage(mol, size=(size, size))
                    img_path = os.path.join(self.output_dir, f"{base_name}.png")
                    pil_img.save(img_path)
                    return img_path
                except Exception as e:
                    logger.error(f"PIL 渲染失败: {e}")
                    return None
        except Exception as e:
            logger.error(f"渲染分子失败: {e}")
            return None

    def create_gallery(self, molecules_data: List[Dict], max_molecules: int = 20) -> Optional[str]:
        """
        生成分子 2D 画廊 HTML
        """
        if not HAS_RDKIT:
            logger.warning("RDKit 不可用，跳过 2D 画廊生成")
            return None

        molecules = molecules_data[:max_molecules]
        items = []
        for i, mol in enumerate(molecules):
            smiles = mol.get("smiles", "")
            cid = mol.get("compound_id", f"mol_{i+1}")
            if not smiles:
                continue
            img = self.render_molecule(smiles, cid)
            if img:
                items.append({
                    "compound_id": cid,
                    "smiles": smiles,
                    "image_file": os.path.basename(img),
                    "binding_affinity": mol.get("binding_affinity", "N/A"),
                    "molecular_weight": mol.get("molecular_weight", "N/A"),
                    "logp": mol.get("logp", "N/A")
                })

        cards = "\n".join([
            f"""
            <div class=\"molecule-card\">
              <img src=\"{it['image_file']}\" alt=\"{it['compound_id']}\" class=\"mol-image\"/>
              <div class=\"mol-info\">
                <h3>{it['compound_id']}</h3>
                <p><strong>SMILES:</strong> {it['smiles']}</p>
                <p><strong>结合亲和力:</strong> {it['binding_affinity']}</p>
                <p><strong>分子量:</strong> {it['molecular_weight']}</p>
                <p><strong>LogP:</strong> {it['logp']}</p>
              </div>
            </div>
            """ for it in items
        ])

        html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <title>分子 2D 画廊</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; margin: 20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .molecule-card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .mol-image {{ width: 100%; height: auto; border-radius: 6px; }}
    .mol-info h3 {{ margin: 8px 0 6px; font-size: 16px; }}
    .mol-info p {{ margin: 4px 0; color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>分子 2D 画廊</h1>
    <div class=\"gallery\">
      {cards}
    </div>
  </div>
</body>
</html>
"""
        out = os.path.join(self.output_dir, "molecular_gallery.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        return out
