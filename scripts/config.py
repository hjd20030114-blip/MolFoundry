# -*- coding: utf-8 -*-
"""
PRRSV病毒衣壳蛋白抑制剂设计项目配置文件
"""

import os

# 项目根目录 - 指向HJD目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# 确保目录存在
for directory in [DATA_DIR, RESULTS_DIR, SCRIPTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 蛋白质文件路径
PROTEIN_FILES = {
    "virus_protein": os.path.join(DATA_DIR, "1p65.pdbqt"),  # PRRSV核衣壳蛋白（主要靶点）
    "integrin_complex_1": os.path.join(DATA_DIR, "AF-Q9GLP0-F1-model_v4_rigid.pdbqt"),  # 整合素复合物1
    "integrin_complex_2": os.path.join(DATA_DIR, "AF-F1SR53-F1-model_v4.pdbqt"),  # 整合素复合物2
    "old_virus_protein": os.path.join(DATA_DIR, "new1p65.pdbqt"),  # 旧的病毒蛋白文件（备用）
}

# 序列文件路径
SEQUENCE_FILES = {
    "capsid": os.path.join(DATA_DIR, "capsid.fasta"),
    "integrin": os.path.join(DATA_DIR, "integrin.fasta"),
}

# 智能检测不同平台的vina可执行文件路径
_default_vina = os.path.join(DATA_DIR, "vina.exe")
_project_parent = os.path.dirname(PROJECT_ROOT)
_mac_vina = os.path.join(_project_parent, "autodock_vina", "bin", "vina")
_vina_path = _default_vina
if os.path.exists(_mac_vina) and os.access(_mac_vina, os.X_OK):
    _vina_path = _mac_vina
elif os.path.exists(_default_vina) and os.access(_default_vina, os.X_OK):
    _vina_path = _default_vina
else:
    # 退回到系统路径中的vina
    _vina_path = "vina"

# AutoDock Vina配置
VINA_CONFIG = {
    # 结合位点坐标 (基于1p65.pdbqt结构)
    "center_x": -0.24,
    "center_y": 26.06, 
    "center_z": 65.48,
    
    # 搜索盒子尺寸 (Å)
    "size_x": 55.85,
    "size_y": 49.77,
    "size_z": 70.43,
    
    # 对接参数 - 优化以获得更好的结合亲和力
    "exhaustiveness": 16,  # 增加搜索详尽度（8 -> 16）
    "num_modes": 20,       # 增加构象数量（9 -> 20）
    "energy_range": 4,     # 增加能量范围以获得更多构象
    
    # Vina可执行文件路径（已智能检测）
    "vina_exe": _vina_path,
}

# 分子生成参数
LIGAND_GENERATION = {
    "num_ligands": 100,  # 生成配体数量（默认值，可被用户输入覆盖）
    "max_atoms": 50,    # 最大原子数
    "min_atoms": 10,    # 最小原子数
    "target_logp": 2.5, # 目标LogP值
    "target_mw": 400,   # 目标分子量
}

# ADMET筛选标准 - 针对抗病毒药物优化
ADMET_CRITERIA = {
    "logp_range": (1, 4.5),         # LogP范围（抗病毒药物通常需要适中的脂溶性）
    "molecular_weight": (200, 600), # 分子量范围（扩大以包含更多活性分子）
    "hbd_max": 4,                   # 最大氢键供体数（减少以提高膜透过性）
    "hba_max": 8,                   # 最大氢键受体数（适中以平衡溶解性和透过性）
    "rotatable_bonds_max": 8,       # 最大可旋转键数（略增加以允许更好的结合适应性）
    "tpsa_range": (30, 120),        # 拓扑极性表面积范围（优化口服生物利用度）
    "aromatic_rings_max": 4,        # 最大芳香环数（新增：控制分子复杂度）
    "heavy_atoms_range": (15, 45),  # 重原子数范围（新增：确保分子大小适中）
}

# 结果文件命名
OUTPUT_FILES = {
    "docking_results": "docking_results.csv",
    "admet_results": "admet_analysis.csv",
    "top_ligands": "top_ligands.csv",
    "binding_analysis": "binding_analysis.txt",
    "ligand_images": "ligand_images/",
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(RESULTS_DIR, "project.log"),
}

# 数据库配置 (可选，用于存储结果)
DATABASE_CONFIG = {
    "enabled": False,
    "type": "sqlite",
    "path": os.path.join(RESULTS_DIR, "prrsv_inhibitors.db"),
} 