#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 3D分子查看器组件
在Web界面中直接显示3D分子结构
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
from typing import Optional

# 检查依赖
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def create_3d_molecule_viewer(smiles: str, width: int = 800, height: int = 600, 
                             title: str = "分子3D结构") -> Optional[str]:
    """
    创建3D分子查看器
    
    Args:
        smiles: SMILES字符串
        width: 查看器宽度
        height: 查看器高度
        title: 标题
        
    Returns:
        HTML字符串或None
    """
    if not PY3DMOL_AVAILABLE:
        st.error("需要安装py3Dmol: pip install py3dmol")
        return None
    
    if not RDKIT_AVAILABLE:
        st.error("需要安装RDKit: pip install rdkit-pypi")
        return None
    
    try:
        # 生成3D分子结构
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error(f"无法解析SMILES: {smiles}")
            return None
        
        # 添加氢原子并生成3D构象
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # 转换为SDF格式
        sdf_block = Chem.MolToMolBlock(mol)
        
        # 创建3D查看器
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(sdf_block, 'sdf')
        viewer.setStyle({'stick': {'radius': 0.1}, 'sphere': {'radius': 0.3}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        
        # 生成HTML
        html_content = f"""
        <div style="text-align: center; margin: 10px 0;">
            <h4>{title}</h4>
            <p><strong>SMILES:</strong> {smiles}</p>
        </div>
        <div id="viewer" style="height: {height}px; width: {width}px; margin: 0 auto;"></div>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        <script>
        {viewer._make_html().split('<script>')[1].split('</script>')[0]}
        </script>
        """
        
        return html_content
        
    except Exception as e:
        st.error(f"生成3D分子结构失败: {e}")
        return None

def display_molecule_3d(smiles: str, compound_id: str = "", width: int = 800, height: int = 600):
    """
    在Streamlit中显示3D分子结构
    
    Args:
        smiles: SMILES字符串
        compound_id: 化合物ID
        width: 查看器宽度
        height: 查看器高度
    """
    title = f"{compound_id} - 3D分子结构" if compound_id else "3D分子结构"
    
    html_content = create_3d_molecule_viewer(smiles, width, height, title)
    
    if html_content:
        components.html(html_content, width=width, height=height + 100)
    else:
        # 备用显示方式
        st.info(f"**{title}**")
        st.code(f"SMILES: {smiles}")
        st.warning("3D可视化不可用，请安装必要的依赖包")

def create_protein_ligand_viewer(pdb_content: str, ligand_smiles: str = "", 
                                width: int = 1000, height: int = 700) -> Optional[str]:
    """
    创建蛋白质-配体复合物查看器
    
    Args:
        pdb_content: PDB文件内容
        ligand_smiles: 配体SMILES
        width: 查看器宽度
        height: 查看器高度
        
    Returns:
        HTML字符串或None
    """
    if not PY3DMOL_AVAILABLE:
        return None
    
    try:
        # 创建3D查看器
        viewer = py3Dmol.view(width=width, height=height)
        
        # 添加蛋白质
        viewer.addModel(pdb_content, 'pdb')
        viewer.setStyle({'cartoon': {'color': 'lightblue', 'opacity': 0.8}})
        
        # 如果有配体，添加配体
        if ligand_smiles and RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(ligand_smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    
                    sdf_block = Chem.MolToMolBlock(mol)
                    viewer.addModel(sdf_block, 'sdf')
                    viewer.setStyle({'model': -1}, {'stick': {'radius': 0.2, 'color': 'green'}, 
                                                   'sphere': {'radius': 0.4, 'color': 'green'}})
            except Exception as e:
                st.warning(f"添加配体失败: {e}")
        
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        
        # 生成HTML
        html_content = f"""
        <div style="text-align: center; margin: 10px 0;">
            <h4>蛋白质-配体复合物</h4>
            <p><strong>蛋白质:</strong> 蓝色卡通模型 | <strong>配体:</strong> 绿色棒球模型</p>
        </div>
        <div id="viewer" style="height: {height}px; width: {width}px; margin: 0 auto;"></div>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        <script>
        {viewer._make_html().split('<script>')[1].split('</script>')[0]}
        </script>
        """
        
        return html_content
        
    except Exception as e:
        st.error(f"生成蛋白质-配体复合物失败: {e}")
        return None

def display_protein_ligand_complex(pdb_file: str, ligand_smiles: str = "", 
                                  width: int = 1000, height: int = 700):
    """
    在Streamlit中显示蛋白质-配体复合物
    
    Args:
        pdb_file: PDB文件路径
        ligand_smiles: 配体SMILES
        width: 查看器宽度
        height: 查看器高度
    """
    if not os.path.exists(pdb_file):
        st.error(f"PDB文件不存在: {pdb_file}")
        return
    
    try:
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()
        
        html_content = create_protein_ligand_viewer(pdb_content, ligand_smiles, width, height)
        
        if html_content:
            components.html(html_content, width=width, height=height + 100)
        else:
            st.warning("蛋白质-配体复合物3D可视化不可用")
            
    except Exception as e:
        st.error(f"读取PDB文件失败: {e}")

def create_molecular_comparison_viewer(molecules: list, width: int = 1200, height: int = 600):
    """
    创建分子比较查看器
    
    Args:
        molecules: 分子列表，每个元素包含{'smiles': str, 'name': str, 'color': str}
        width: 查看器宽度
        height: 查看器高度
    """
    if not PY3DMOL_AVAILABLE or not RDKIT_AVAILABLE:
        st.warning("分子比较查看器需要py3Dmol和RDKit")
        return
    
    try:
        viewer = py3Dmol.view(width=width, height=height)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        for i, mol_info in enumerate(molecules[:6]):  # 最多显示6个分子
            smiles = mol_info.get('smiles', '')
            name = mol_info.get('name', f'Molecule {i+1}')
            color = mol_info.get('color', colors[i % len(colors)])
            
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                    
                    # 移动分子位置以避免重叠
                    conf = mol.GetConformer()
                    offset_x = (i % 3) * 15  # 3列布局
                    offset_y = (i // 3) * 15  # 2行布局
                    
                    for atom_idx in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(atom_idx)
                        conf.SetAtomPosition(atom_idx, (pos.x + offset_x, pos.y + offset_y, pos.z))
                    
                    sdf_block = Chem.MolToMolBlock(mol)
                    viewer.addModel(sdf_block, 'sdf')
                    viewer.setStyle({'model': i}, {'stick': {'radius': 0.1, 'color': color}, 
                                                  'sphere': {'radius': 0.3, 'color': color}})
                    
                    # 添加标签
                    viewer.addLabel(name, {'position': {'x': offset_x, 'y': offset_y + 8, 'z': 0}},
                                  {'fontSize': 12, 'fontColor': color})
        
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        
        # 生成HTML
        html_content = f"""
        <div style="text-align: center; margin: 10px 0;">
            <h4>分子结构比较</h4>
            <p>显示前{min(len(molecules), 6)}个分子的3D结构对比</p>
        </div>
        <div id="viewer" style="height: {height}px; width: {width}px; margin: 0 auto;"></div>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        <script>
        {viewer._make_html().split('<script>')[1].split('</script>')[0]}
        </script>
        """
        
        components.html(html_content, width=width, height=height + 100)
        
    except Exception as e:
        st.error(f"生成分子比较查看器失败: {e}")

# Streamlit应用示例
def main():
    """主函数 - 用于测试"""
    st.title("3D分子查看器测试")
    
    # 测试分子
    test_smiles = "CCOc1ccc(C(N)=O)cc1"
    
    st.subheader("单个分子3D结构")
    display_molecule_3d(test_smiles, "测试分子")
    
    st.subheader("分子比较")
    test_molecules = [
        {'smiles': 'CCOc1ccc(C(N)=O)cc1', 'name': '分子1', 'color': 'red'},
        {'smiles': 'Cc1ccc(C(=O)N(C)C)cc1', 'name': '分子2', 'color': 'blue'},
        {'smiles': 'COc1ccc(CCN)cc1', 'name': '分子3', 'color': 'green'}
    ]
    create_molecular_comparison_viewer(test_molecules)

if __name__ == "__main__":
    main()
