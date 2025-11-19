# 数据集说明

## PDBbind-Plus数据集

本项目使用PDBbind-Plus数据库（19,037个蛋白质-配体复合物）。

### 数据结构

```
data/P-L/
├── 1981-2000/
│   ├── 10gs/
│   │   ├── 10gs_protein.pdb
│   │   ├── 10gs_pocket.pdb
│   │   ├── 10gs_ligand.sdf
│   │   └── 10gs_ligand.mol2
│   └── ...
├── 2001-2010/
└── 2011-2025/
```

### 下载数据集

由于数据集过大（~15GB），未包含在仓库中。请从以下来源下载：

1. **官方来源**: [PDBbind数据库](http://www.pdbbind.org.cn/)
2. **备用链接**: [待补充]

### 数据预处理

下载后，运行以下命令处理数据：

```bash
# 解压数据集
tar -xzvf pdbbind_plus.tar.gz -C data/

# 验证数据完整性
python tools/verify_data.py

# 查看数据统计
python tools/data_statistics.py
```

### 数据划分

训练/验证/测试集划分已保存在 `results/pl_splits_seed42_fold*.json`
- 训练集: ~15,230 复合物
- 验证集: ~1,903 复合物  
- 测试集: ~1,904 复合物

### 引用

如果使用本数据集，请引用：

```bibtex
@article{pdbbind2023,
  title={PDBbind-Plus: An Extended Database of Protein-Ligand Complexes},
  author={Liu, Z. and others},
  journal={Journal of Medicinal Chemistry},
  year={2023}
}
```
