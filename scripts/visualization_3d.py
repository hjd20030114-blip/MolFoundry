#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D可视化模块
用于展示PRRSV抑制剂设计结果的3D可视化
包括分子结构、蛋白质-配体复合物、结合位点等
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import tempfile
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from .config import *

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查3D可视化库的可用性
VISUALIZATION_AVAILABLE = False
RDKIT_AVAILABLE = False
PY3DMOL_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw
    RDKIT_AVAILABLE = True

    # 尝试导入2D绘图模块
    try:
        from rdkit.Chem import rdDepictor, rdMolDraw2D
        RDKIT_2D_AVAILABLE = True
    except ImportError:
        RDKIT_2D_AVAILABLE = False
        logger.warning("RDKit 2D绘图模块不可用，将跳过分子画廊功能")

    logger.info("RDKit已成功导入，将使用分子结构可视化功能")
except ImportError as e:
    RDKIT_AVAILABLE = False
    RDKIT_2D_AVAILABLE = False
    logger.warning(f"RDKit不可用: {e}")

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
    logger.info("py3Dmol已成功导入，将使用3D分子可视化功能")
except ImportError as e:
    logger.warning(f"py3Dmol不可用: {e}")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly已成功导入，将使用交互式图表功能")
except ImportError as e:
    logger.warning(f"Plotly不可用: {e}")

VISUALIZATION_AVAILABLE = RDKIT_AVAILABLE and PY3DMOL_AVAILABLE and PLOTLY_AVAILABLE

class Visualizer3D:
    """3D可视化器"""
    
    def __init__(self, output_dir: str = "visualization_output"):
        """
        初始化3D可视化器

        Args:
            output_dir: 输出目录
        """
        # 尝试使用结果管理器的当前运行目录
        try:
            from scripts.result_manager import result_manager

            # 如果有当前运行目录，使用它；否则使用默认目录
            if result_manager.current_run_dir:
                self.output_dir = str(result_manager.get_3d_viz_dir())
            else:
                self.output_dir = output_dir
        except ImportError:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.available = VISUALIZATION_AVAILABLE
        if not self.available:
            logger.warning("3D可视化功能不完全可用，请安装: pip install py3dmol plotly rdkit-pypi")

        logger.info(f"3D可视化器初始化完成，输出目录: {self.output_dir}")
    
    def visualize_molecule_3d(self, smiles: str, title: str = "Molecule") -> str:
        """
        3D分子结构可视化
        
        Args:
            smiles: SMILES字符串
            title: 分子标题
            
        Returns:
            HTML文件路径
        """
        if not RDKIT_AVAILABLE or not PY3DMOL_AVAILABLE:
            logger.error("分子3D可视化需要RDKit和py3Dmol")
            return ""
        
        try:
            # 生成3D构象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"无法解析SMILES: {smiles}")
                return ""
            
            # 添加氢原子
            mol = Chem.AddHs(mol)
            
            # 生成3D构象
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 转换为SDF格式
            sdf_block = Chem.MolToMolBlock(mol)
            
            # 创建3D可视化
            viewer = py3Dmol.view(width=800, height=600)
            viewer.addModel(sdf_block, 'sdf')
            viewer.setStyle({'stick': {'radius': 0.1}, 'sphere': {'radius': 0.3}})
            viewer.setBackgroundColor('white')
            viewer.zoomTo()
            
            # 保存HTML
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title} - 3D分子结构</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .info {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title} - 3D分子结构</h1>
        <div class="info">
            <strong>SMILES:</strong> {smiles}<br>
            <strong>分子式:</strong> {Chem.rdMolDescriptors.CalcMolFormula(mol)}<br>
            <strong>分子量:</strong> {Descriptors.MolWt(mol):.2f} Da
        </div>
        <div id="viewer" style="height: 600px; width: 100%; position: relative;"></div>
        {viewer._make_html()}
    </div>
</body>
</html>
"""
            
            output_file = os.path.join(self.output_dir, f"{title.replace(' ', '_')}_3D.html")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"3D分子结构已保存: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"3D分子可视化失败: {e}")
            return ""
    
    def visualize_protein_ligand_complex(self, pdb_file: str, ligand_smiles: str, 
                                       binding_site: Dict = None) -> str:
        """
        蛋白质-配体复合物3D可视化
        
        Args:
            pdb_file: PDB文件路径
            ligand_smiles: 配体SMILES
            binding_site: 结合位点信息
            
        Returns:
            HTML文件路径
        """
        if not PY3DMOL_AVAILABLE:
            logger.error("蛋白质-配体复合物可视化需要py3Dmol")
            return ""
        
        try:
            # 读取PDB文件
            if not os.path.exists(pdb_file):
                logger.error(f"PDB文件不存在: {pdb_file}")
                return ""
            
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
            
            # 创建3D可视化
            viewer = py3Dmol.view(width=1000, height=700)
            
            # 添加蛋白质
            viewer.addModel(pdb_content, 'pdb')
            
            # 蛋白质样式
            viewer.setStyle({'cartoon': {'color': 'lightblue', 'opacity': 0.8}})
            
            # 如果有结合位点信息，高亮显示
            if binding_site and 'center' in binding_site:
                center = binding_site['center']
                radius = binding_site.get('radius', 8.0)

                # 高亮结合位点球体
                viewer.addSphere({
                    'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                    'radius': radius,
                    'color': 'yellow',
                    'alpha': 0.3
                })

                # 高亮结合位点残基
                if 'residues' in binding_site:
                    for residue in binding_site['residues']:
                        # 解析残基信息 (格式: RES_NUM_CHAIN)
                        parts = residue.split('_')
                        if len(parts) >= 2:
                            res_num = parts[1]
                            chain = parts[2] if len(parts) > 2 else 'A'

                            # 高亮显示结合残基
                            viewer.setStyle(
                                {'resi': res_num, 'chain': chain},
                                {'stick': {'color': 'red', 'radius': 0.3},
                                 'cartoon': {'color': 'red', 'opacity': 0.8}}
                            )

                            # 添加残基标签
                            viewer.addLabel(
                                residue.replace('_', ''),
                                {'position': {'x': center[0], 'y': center[1], 'z': center[2]},
                                 'backgroundColor': 'red', 'fontColor': 'white',
                                 'fontSize': 12, 'showBackground': True}
                            )
                
                # 显示结合位点残基
                viewer.setStyle(
                    {'within': {'distance': radius, 'sel': {'x': center[0], 'y': center[1], 'z': center[2]}}},
                    {'stick': {'radius': 0.3}, 'cartoon': {'color': 'red'}}
                )
            
            # 如果有配体SMILES，生成配体结构并添加
            if ligand_smiles and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(ligand_smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    
                    # 如果有结合位点，将配体放置在结合位点附近
                    if binding_site and 'center' in binding_site:
                        conf = mol.GetConformer()
                        center = binding_site['center']
                        # 简单地将配体移动到结合位点
                        for i in range(mol.GetNumAtoms()):
                            pos = conf.GetAtomPosition(i)
                            conf.SetAtomPosition(i, (pos.x + center[0], pos.y + center[1], pos.z + center[2]))
                    
                    sdf_block = Chem.MolToMolBlock(mol)
                    viewer.addModel(sdf_block, 'sdf')
                    viewer.setStyle({'model': -1}, {'stick': {'radius': 0.2, 'color': 'green'}, 
                                                   'sphere': {'radius': 0.4, 'color': 'green'}})
            
            viewer.setBackgroundColor('white')
            viewer.zoomTo()
            
            # 生成HTML
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>蛋白质-配体复合物 - 3D可视化</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .info {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .controls {{ background: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PRRSV衣壳蛋白-配体复合物 3D可视化</h1>
        <div class="info">
            <strong>蛋白质文件:</strong> {os.path.basename(pdb_file)}<br>
            <strong>配体SMILES:</strong> {ligand_smiles}<br>
            {'<strong>结合位点名称:</strong> ' + binding_site.get('name', 'N/A') + '<br>' if binding_site else ''}
            {'<strong>结合位点中心:</strong> ' + str(binding_site.get('center', 'N/A')) + '<br>' if binding_site else ''}
            {'<strong>结合位点半径:</strong> ' + str(binding_site.get('radius', 'N/A')) + ' Å<br>' if binding_site else ''}
            {'<strong>结合残基数量:</strong> ' + str(len(binding_site.get('residues', []))) + '<br>' if binding_site else ''}
        </div>

        {f'''
        <div class="info" style="background: #fff3cd; border-left: 4px solid #ffc107;">
            <h3>🧬 结合位点残基</h3>
            <div style="font-family: monospace; background: white; padding: 10px; border-radius: 3px;">
                {', '.join(binding_site.get('residues', []))}
            </div>
            <p><strong>描述:</strong> {binding_site.get('description', '无描述')}</p>
        </div>
        ''' if binding_site and binding_site.get('residues') else ''}
        <div class="controls">
            <strong>可视化说明:</strong><br>
            • 蛋白质: 浅蓝色卡通模型<br>
            • 结合位点: 黄色透明球体<br>
            • 结合位点残基: 红色棒状模型<br>
            • 配体分子: 绿色棒球模型<br>
            • 鼠标操作: 左键旋转，右键缩放，中键平移
        </div>
        <div id="viewer" style="height: 700px; width: 100%; position: relative;"></div>
        {viewer._make_html()}
    </div>
</body>
</html>
"""
            
            output_file = os.path.join(self.output_dir, "protein_ligand_complex_3D.html")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"蛋白质-配体复合物3D可视化已保存: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"蛋白质-配体复合物可视化失败: {e}")
            return ""
    
    def create_interactive_results_dashboard(self, results_data: List[Dict]) -> str:
        """
        创建交互式结果仪表板
        
        Args:
            results_data: 结果数据列表
            
        Returns:
            HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            logger.error("交互式仪表板需要Plotly")
            return ""
        
        try:
            df = pd.DataFrame(results_data)

            # 创建3x2子图：
            # 1,1: 结合亲和力分布（hist）  1,2: 分子量 vs LogP（scatter）
            # 2,1: Top-10 排序（bar）      2,2: TPSA 分布（hist，如无则留空）
            # 3,1: 溶解度等级（bar/计数）   3,2: 毒性风险（pie，如无则Lipinski饼图）
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '结合亲和力分布', '分子性质散点图 (MW vs LogP)',
                    'Top-10 化合物排序', 'TPSA 分布',
                    '溶解度等级分布', '毒性风险 / Lipinski占比'
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "histogram"}],
                    [{"type": "bar"}, {"type": "domain"}]
                ]
            )
            
            # 1. 结合亲和力分布直方图
            if 'binding_affinity' in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df['binding_affinity'],
                        name='结合亲和力',
                        nbinsx=20,
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
            
            # 2. 分子量 vs LogP 散点图
            if 'molecular_weight' in df.columns and 'logp' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['molecular_weight'],
                        y=df['logp'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=df['binding_affinity'] if 'binding_affinity' in df.columns else None,
                            colorscale='Viridis',
                            showscale=('binding_affinity' in df.columns),
                            colorbar=dict(title="结合亲和力") if 'binding_affinity' in df.columns else None
                        ),
                        text=df.get('compound_id', []),
                        name='化合物'
                    ),
                    row=1, col=2
                )
            
            # 3. 前10个化合物排序（若无total_score则使用负binding_affinity排序）
            try:
                if 'total_score' in df.columns:
                    top_10 = df.nlargest(10, 'total_score')
                    y_data = top_10['total_score']
                    y_name = '综合评分'
                elif 'binding_affinity' in df.columns:
                    top_10 = df.nsmallest(10, 'binding_affinity')
                    y_data = -top_10['binding_affinity']  # 亲和力越低越好，取负便于直观比较
                    y_name = '-Binding Affinity'
                else:
                    top_10 = df.head(10)
                    y_data = [0]*len(top_10)
                    y_name = 'N/A'
                fig.add_trace(
                    go.Bar(
                        x=top_10.get('compound_id', []),
                        y=y_data,
                        name=y_name,
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
            except Exception:
                pass
            
            # 4. TPSA 分布（如有）
            if 'tpsa' in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df['tpsa'],
                        name='TPSA',
                        nbinsx=20,
                        marker_color='#9b59b6'
                    ),
                    row=2, col=2
                )
            
            # 5. 溶解度等级分布（如有）或 logS 分布
            if 'solubility_class' in df.columns and df['solubility_class'].notna().any():
                vc = df['solubility_class'].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=vc.index.tolist(),
                        y=vc.values.tolist(),
                        name='Solubility Class',
                        marker_color='#16a085'
                    ),
                    row=3, col=1
                )
            elif 'predicted_logS' in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df['predicted_logS'],
                        name='logS',
                        nbinsx=20,
                        marker_color='#16a085'
                    ),
                    row=3, col=1
                )

            # 6. 毒性风险占比（如无则 Lipinski 合规性）
            if 'toxicity_risk_level' in df.columns and df['toxicity_risk_level'].notna().any():
                vc_tox = df['toxicity_risk_level'].fillna('未知').value_counts()
                fig.add_trace(
                    go.Pie(labels=vc_tox.index.tolist(),
                           values=vc_tox.values.tolist(), name='Toxicity Risk', hole=0.3),
                    row=3, col=2
                )
            elif 'lipinski_compliant' in df.columns:
                labels = ['符合Lipinski','不符合Lipinski']
                values = [int(df['lipinski_compliant'].sum()), int((~df['lipinski_compliant'].astype(bool)).sum())]
                fig.add_trace(
                    go.Pie(labels=labels, values=values, name='Lipinski', hole=0.3),
                    row=3, col=2
                )

            # 更新布局
            fig.update_layout(
                title_text="PRRSV抑制剂设计结果 - 交互式仪表板",
                showlegend=True,
                height=1200
            )
            
            # 更新坐标轴标签
            fig.update_xaxes(title_text="结合亲和力 (kcal/mol)", row=1, col=1)
            fig.update_yaxes(title_text="频次", row=1, col=1)
            fig.update_xaxes(title_text="分子量 (Da)", row=1, col=2)
            fig.update_yaxes(title_text="LogP", row=1, col=2)
            fig.update_xaxes(title_text="化合物ID", row=2, col=1)
            fig.update_yaxes(title_text="指标值", row=2, col=1)
            fig.update_xaxes(title_text="TPSA", row=2, col=2)
            fig.update_yaxes(title_text="频次", row=2, col=2)
            fig.update_xaxes(title_text="溶解度等级", row=3, col=1)
            fig.update_yaxes(title_text="数量", row=3, col=1)
            
            # 保存HTML
            output_file = os.path.join(self.output_dir, "interactive_dashboard.html")
            fig.write_html(output_file)

            # 同步导出静态PNG（需要kaleido）
            try:
                png_path = os.path.join(self.output_dir, "interactive_dashboard.png")
                fig.write_image(png_path, format="png", width=1200, height=800, scale=2)
                logger.info(f"交互式仪表板PNG已导出: {png_path}")
            except Exception as e:
                logger.warning(f"交互式仪表板PNG导出失败（可能缺少kaleido或环境不支持）: {e}")
            
            logger.info(f"交互式仪表板已保存: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"创建交互式仪表板失败: {e}")
            return ""

    def visualize_binding_site_analysis(self, pdb_file: str, binding_sites: List[Dict]) -> str:
        """
        结合位点分析3D可视化

        Args:
            pdb_file: PDB文件路径
            binding_sites: 结合位点列表

        Returns:
            HTML文件路径
        """
        if not PY3DMOL_AVAILABLE:
            logger.error("结合位点分析可视化需要py3Dmol")
            return ""

        try:
            # 读取PDB文件
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()

            # 创建3D可视化
            viewer = py3Dmol.view(width=1000, height=700)
            viewer.addModel(pdb_content, 'pdb')

            # 蛋白质基本样式
            viewer.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.7}})

            # 为每个结合位点添加不同颜色的球体
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

            for i, site in enumerate(binding_sites):
                if 'center' in site:
                    color = colors[i % len(colors)]
                    center = site['center']
                    radius = site.get('radius', 5.0)

                    # 添加结合位点球体
                    viewer.addSphere({
                        'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                        'radius': radius,
                        'color': color,
                        'alpha': 0.3
                    })

                    # 高亮结合残基
                    if 'residues' in site:
                        for residue in site['residues']:
                            # 解析残基信息 (格式: RES_NUM_CHAIN)
                            parts = residue.split('_')
                            if len(parts) >= 2:
                                res_num = parts[1]
                                chain = parts[2] if len(parts) > 2 else 'A'

                                # 高亮显示结合残基
                                viewer.setStyle(
                                    {'resi': res_num, 'chain': chain},
                                    {'stick': {'color': color, 'radius': 0.3},
                                     'cartoon': {'color': color, 'opacity': 0.9}}
                                )

                    # 添加标签
                    viewer.addLabel(
                        site.get('name', f'Site {i+1}'),
                        {'position': {'x': center[0], 'y': center[1], 'z': center[2] + radius + 2}},
                        {'fontSize': 12, 'fontColor': color}
                    )

            viewer.setBackgroundColor('white')
            viewer.zoomTo()

            # 生成HTML
            sites_info = ""
            for i, site in enumerate(binding_sites):
                color = colors[i % len(colors)]
                sites_info += f"""
                <div style="margin: 5px 0; padding: 5px; border-left: 4px solid {color};">
                    <strong>{site.get('name', f'位点 {i+1}')}</strong><br>
                    中心坐标: {site.get('center', 'N/A')}<br>
                    半径: {site.get('radius', 'N/A')} Å<br>
                    重要性: {site.get('importance', 'N/A')}<br>
                    描述: {site.get('description', 'N/A')}
                </div>
                """

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>结合位点分析 - 3D可视化</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .info {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .sites {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PRRSV衣壳蛋白结合位点分析</h1>
        <div class="info">
            <strong>蛋白质文件:</strong> {os.path.basename(pdb_file)}<br>
            <strong>识别的结合位点数量:</strong> {len(binding_sites)}
        </div>
        <div class="sites">
            <h3>结合位点详情:</h3>
            {sites_info}
        </div>
        <div id="viewer" style="height: 700px; width: 100%; position: relative;"></div>
        {viewer._make_html()}
    </div>
</body>
</html>
"""

            output_file = os.path.join(self.output_dir, "binding_sites_analysis_3D.html")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"结合位点分析3D可视化已保存: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"结合位点分析可视化失败: {e}")
            return ""

    def create_molecular_gallery(self, molecules_data: List[Dict], max_molecules: int = 20) -> str:
        """
        创建分子画廊

        Args:
            molecules_data: 分子数据列表
            max_molecules: 最大显示分子数

        Returns:
            HTML文件路径
        """
        # 只要 RDKit 可用即可生成 2D 图：
        # 优先使用 SVG (无需 Cairo)，有 Cairo 则用 PNG；若 rdMolDraw2D 不可用，回退到 PIL 绘制
        if not RDKIT_AVAILABLE:
            logger.warning("分子画廊需要RDKit，跳过分子画廊生成")
            return ""

        try:
            # 限制分子数量
            molecules = molecules_data[:max_molecules]

            # 生成分子图片
            mol_images = []
            for i, mol_data in enumerate(molecules):
                smiles = mol_data.get('smiles', '')
                compound_id = mol_data.get('compound_id', f'mol_{i+1}')

                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        img_file = None
                        # 优先使用 rdMolDraw2D：若有 Cairo，用 PNG；否则用 SVG
                        try:
                            from rdkit.Chem import rdMolDraw2D  # 局部导入以便环境兼容
                            if hasattr(rdMolDraw2D, 'MolDraw2DCairo'):
                                drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
                                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                                drawer.FinishDrawing()
                                img_bytes = drawer.GetDrawingText()
                                img_file = os.path.join(self.output_dir, f"{compound_id}.png")
                                with open(img_file, 'wb') as f:
                                    f.write(img_bytes)
                            else:
                                # 使用 SVG，无需 Cairo
                                drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
                                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                                drawer.FinishDrawing()
                                svg = drawer.GetDrawingText()
                                img_file = os.path.join(self.output_dir, f"{compound_id}.svg")
                                with open(img_file, 'w', encoding='utf-8') as f:
                                    f.write(svg)
                        except Exception:
                            # 回退到 PIL 绘图
                            try:
                                pil_img = Draw.MolToImage(mol, size=(300, 300))
                                img_file = os.path.join(self.output_dir, f"{compound_id}.png")
                                pil_img.save(img_file)
                            except Exception as e:
                                logger.warning(f"绘制分子2D图失败({compound_id}): {e}")
                                img_file = None

                        if img_file:
                            mol_images.append({
                                'compound_id': compound_id,
                                'smiles': smiles,
                                'image_file': os.path.basename(img_file),
                                'binding_affinity': mol_data.get('binding_affinity', 'N/A'),
                                'molecular_weight': mol_data.get('molecular_weight', 'N/A'),
                                'logp': mol_data.get('logp', 'N/A')
                            })

            # 生成HTML画廊
            gallery_html = ""
            for mol in mol_images:
                gallery_html += f"""
                <div class="molecule-card">
                    <img src="{mol['image_file']}" alt="{mol['compound_id']}" class="mol-image">
                    <div class="mol-info">
                        <h3>{mol['compound_id']}</h3>
                        <p><strong>SMILES:</strong> {mol['smiles'][:50]}{'...' if len(mol['smiles']) > 50 else ''}</p>
                        <p><strong>结合亲和力:</strong> {mol['binding_affinity']} kcal/mol</p>
                        <p><strong>分子量:</strong> {mol['molecular_weight']} Da</p>
                        <p><strong>LogP:</strong> {mol['logp']}</p>
                    </div>
                </div>
                """

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PRRSV抑制剂分子画廊</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
        .molecule-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .molecule-card:hover {{ transform: translateY(-5px); }}
        .mol-image {{ width: 100%; height: auto; border-radius: 5px; }}
        .mol-info {{ margin-top: 15px; }}
        .mol-info h3 {{ color: #2c3e50; margin: 0 0 10px 0; }}
        .mol-info p {{ margin: 5px 0; color: #555; }}
        .stats {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PRRSV抑制剂分子画廊</h1>
            <p>展示生成的候选化合物结构和性质</p>
        </div>

        <div class="stats">
            <h3>统计信息</h3>
            <p><strong>总分子数:</strong> {len(molecules_data)} | <strong>显示数量:</strong> {len(mol_images)}</p>
        </div>

        <div class="gallery">
            {gallery_html}
        </div>
    </div>
</body>
</html>
"""

            output_file = os.path.join(self.output_dir, "molecular_gallery.html")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"分子画廊已保存: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"创建分子画廊失败: {e}")
            return ""

    def generate_comprehensive_report(self, results_data: List[Dict],
                                    pdb_file: str = None,
                                    binding_sites: List[Dict] = None) -> str:
        """
        生成综合3D可视化报告

        Args:
            results_data: 结果数据
            pdb_file: PDB文件路径
            binding_sites: 结合位点信息

        Returns:
            主报告HTML文件路径
        """
        try:
            logger.info("生成综合3D可视化报告...")

            # 生成各种可视化
            generated_files = []

            # 1. 交互式仪表板
            dashboard_file = self.create_interactive_results_dashboard(results_data)
            if dashboard_file:
                generated_files.append(("交互式仪表板", os.path.basename(dashboard_file)))

            # 2. 分子画廊
            gallery_file = self.create_molecular_gallery(results_data)
            if gallery_file:
                generated_files.append(("分子画廊", os.path.basename(gallery_file)))

            # 3. 前几个最佳分子的3D结构
            top_molecules = sorted(results_data, key=lambda x: x.get('binding_affinity', 0))[:5]
            for i, mol in enumerate(top_molecules):
                if 'smiles' in mol:
                    mol_3d_file = self.visualize_molecule_3d(
                        mol['smiles'],
                        f"Top_{i+1}_{mol.get('compound_id', f'mol_{i+1}')}"
                    )
                    if mol_3d_file:
                        generated_files.append((f"第{i+1}名分子3D结构", os.path.basename(mol_3d_file)))

            # 4. 蛋白质-配体复合物（使用最佳分子）
            if pdb_file and top_molecules:
                best_mol = top_molecules[0]
                complex_file = self.visualize_protein_ligand_complex(
                    pdb_file,
                    best_mol.get('smiles', ''),
                    binding_sites[0] if binding_sites else None
                )
                if complex_file:
                    generated_files.append(("蛋白质-配体复合物", os.path.basename(complex_file)))

            # 5. 结合位点分析
            if pdb_file and binding_sites:
                sites_file = self.visualize_binding_site_analysis(pdb_file, binding_sites)
                if sites_file:
                    generated_files.append(("结合位点分析", os.path.basename(sites_file)))

            # 生成主报告页面
            files_html = ""
            for title, filename in generated_files:
                files_html += f"""
                <div class="file-card">
                    <h3>{title}</h3>
                    <a href="{filename}" target="_blank" class="view-btn">查看 →</a>
                </div>
                """

            # 统计信息
            total_molecules = len(results_data)
            avg_binding = np.mean([mol.get('binding_affinity', 0) for mol in results_data])
            best_binding = min([mol.get('binding_affinity', 0) for mol in results_data])

            main_report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PRRSV抑制剂设计 - 3D可视化综合报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin: 10px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{ margin: 0; font-size: 2em; }}
        .stat-card p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .files-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .file-card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .file-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .file-card h3 {{
            color: #2c3e50;
            margin: 0 0 15px 0;
        }}
        .view-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #00b894, #00a085);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 25px;
            transition: all 0.3s ease;
        }}
        .view-btn:hover {{
            background: linear-gradient(135deg, #00a085, #00b894);
            transform: scale(1.05);
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 PRRSV抑制剂设计</h1>
            <p>3D可视化综合报告</p>
            <p><small>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>{total_molecules}</h3>
                <p>生成的候选化合物</p>
            </div>
            <div class="stat-card">
                <h3>{best_binding:.2f}</h3>
                <p>最佳结合亲和力 (kcal/mol)</p>
            </div>
            <div class="stat-card">
                <h3>{avg_binding:.2f}</h3>
                <p>平均结合亲和力 (kcal/mol)</p>
            </div>
            <div class="stat-card">
                <h3>{len(generated_files)}</h3>
                <p>生成的可视化文件</p>
            </div>
        </div>

        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 可视化文件</h2>
        <div class="files-grid">
            {files_html}
        </div>

        <div class="footer">
            <p>🔬 PRRSV衣壳蛋白-整合素PPI抑制剂设计平台</p>
            <p>基于CMD-GEN深度学习模型和AutoDock Vina分子对接</p>
        </div>
    </div>
</body>
</html>
"""

            main_report_file = os.path.join(self.output_dir, "comprehensive_3d_report.html")
            with open(main_report_file, 'w', encoding='utf-8') as f:
                f.write(main_report_html)

            logger.info(f"综合3D可视化报告已生成: {main_report_file}")
            logger.info(f"共生成 {len(generated_files)} 个可视化文件")

            return main_report_file

        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return ""
