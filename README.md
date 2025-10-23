# PRRSV深度学习抑制剂设计平台

## 项目简介

PRRSV（猪繁殖与呼吸综合征病毒）深度学习抑制剂设计平台是一个集成了传统计算化学方法和现代深度学习技术的药物设计系统。该平台专门针对PRRSV病毒进行抑制剂的智能设计和优化。

## 核心功能

### 🧬 分子生成
- **CMD-GEN集成**: 基于DiffPhar和GCPG的智能分子生成
- **深度学习生成**: SE(3)-Equivariant GNN驱动的分子设计
- **口袋条件生成**: 基于蛋白质结合位点的定向分子生成

### 🎯 分子对接
- **AutoDock Vina集成**: 高精度分子对接计算
- **批量对接**: 支持多配体同时对接分析
- **结合位点分析**: 自动识别和分析蛋白质结合位点

### 💊 ADMET分析
- **成药性评估**: Lipinski五规则等成药性指标
- **毒性预测**: 基于机器学习的毒性风险评估
- **药代动力学**: 吸收、分布、代谢、排泄性质预测

### 🤖 深度学习模块
- **SE(3)-Equivariant GNN**: 等变图神经网络
- **Diffusion Model**: 扩散模型分子生成
- **Transformer**: 交叉注意力机制
- **Multi-task Learning**: 多任务学习框架

### 📊 可视化分析
- **3D分子可视化**: 交互式分子结构展示
- **对接结果可视化**: 蛋白质-配体复合物展示
- **数据分析图表**: 实时生成分析图表

## 项目结构

```
HJD/
├── README.md
├── requirements.txt
├── start_project.py                 # 启动脚本（含依赖检查与UI入口）
├── unified_web_interface.py         # 统一 Streamlit 界面
├── run_full_workflow.py             # 一键式完整流程（生成→对接→ADMET→3D报告）
├── deep_learning_pipeline.py        # 深度学习端到端流水线（研究版）
│
├── scripts/                         # 计算化学/工程流水线
│   ├── config.py                    # 路径、Vina参数、ADMET筛选阈值
│   ├── ligand_generator.py          # 配体生成（模板/片段 + CMD-GEN 可选）
│   ├── molecular_docking.py         # AutoDock Vina 对接（含稳健回退）
│   ├── admet_analyzer.py            # RDKit ADMET 与规则评估（含简化回退）
│   ├── binding_site_analyzer.py     # 结合位点解析/几何/表面分析
│   ├── visualization_3d.py          # 3D 分子/复合物/仪表板/综合报告
│   ├── result_manager.py            # 以 run 为单位的结果目录与元信息
│   ├── cmdgen_integration.py        # 外部 CMD-GEN 集成（DiffPhar/GCPG）
│   ├── view_all_3d_results.py       # 打包查看3D结果
│   └── streamlit_3d_viewer.py       # 独立3D查看器
│
├── deep_learning/                   # 深度学习研究模块
│   ├── models/
│   │   ├── equivariant_gnn.py       # SE(3)-等变GNN评分器
│   │   ├── transformer.py           # 口袋-配体交叉注意力Transformer
│   │   ├── diffusion_model.py       # 口袋条件扩散生成器
│   │   ├── discriminator.py         # 多任务判别器（ADMET/合成难度等）
│   │   └── pl_pair_classifier.py    # 口袋-配体二分类
│   └── data/featurizers.py          # 分子/蛋白/相互作用特征化
│
├── data/                            # 示例数据与资源
│   ├── 1p65.pdb / 1p65.pdbqt / 1p65.cif
│   ├── AF-Q9GLP0-F1-model_v4.pdb(.pdbqt)  # 整合素相关结构
│   ├── AF-F1SR53-F1-model_v4.pdb(.pdbqt)
│   ├── capsid.fasta / integrin.fasta       # 序列数据
│   ├── ligands.sdf                         # 种子/示例配体
│   └── P-L/                                # 蛋白-配体对（训练/评测示例）
│
└── results/                         # 运行产出（按日期自动分目录）
    └── run_YYYYMMDD_xxx/
        ├── ligands/                 # 生成配体与 .smi
        ├── docking/                 # docking_results.csv
        ├── admet/                   # admet_results.csv
        ├── visualization_3d/        # HTML报告、仪表板、画廊
        └── reports/                 # 其他报告
```

## 快速开始

### 1. 环境要求
- Python 3.9+
- PyTorch 2.0+
- RDKit
- Streamlit
- 其他依赖见requirements.txt

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动系统
```bash
python start_project.py
```

### 4. 访问Web界面
打开浏览器访问: http://localhost:8501

## 使用指南

### Web界面操作
1. **主页**: 查看平台概览和功能介绍
2. **分子生成**: 使用CMD-GEN或深度学习生成分子
3. **分子对接**: 上传蛋白质和配体进行对接分析
4. **ADMET分析**: 评估分子的成药性和安全性
5. **数据可视化**: 查看3D分子结构和分析结果
6. **实验报告**: 生成综合分析报告

### 命令行使用
```python
# 分子对接示例
from scripts.molecular_docking import MolecularDocking

docker = MolecularDocking()
results = docker.dock_multiple_ligands("HJD/data/1p65.pdb", ["CCO", "CC(=O)O"])

# CMD-GEN 分子生成示例（如未配置CMD-GEN，将自动回退到备用生成）
from scripts.cmdgen_integration import CMDGENGenerator

generator = CMDGENGenerator()
molecules = generator.generate_pocket_based_molecules(
    pdb_file="HJD/data/1p65.pdb", num_molecules=10, ref_ligand="A:1"
)
```

## 项目原理（Principles）

本平台围绕“结构指导的口袋-配体发现”构建可复现的一体化工作流：
- 结构指导生成：以结合口袋为条件，先生成或导入候选分子，再进行筛选与优化
- 物理打分 + 规则筛选：利用 AutoDock Vina 进行结合亲和力打分，结合 RDKit 描述符与 Lipinski 等规则进行 ADMET 初筛
- 可视化闭环：输出 2D/3D 与仪表板/综合报告，辅助解释与人工复核
- 结果可追溯：按 run 组织所有中间与最终产出，便于复现实验与对比

平台包含两类“分子来源”：
1) 规则/模板/片段驱动的化学空间扩展（默认内置，开箱可用）
2) 外部生成模型（CMD-GEN：DiffPhar + GCPG；以及深度学习研究模块，便于未来挂载 SOTA 方法）

## 数据集与预处理（Datasets & Preprocessing）

- 蛋白质结构（Proteins）
  - 示例：`data/1p65.pdb(.pdbqt)`、`data/AF-Q9GLP0-F1-model_v4.pdb(.pdbqt)`、`data/AF-F1SR53-F1-model_v4.pdb(.pdbqt)`
  - 预处理：加氢、转 pdbqt（借助 Meeko/AutoDock 工具链），在 `scripts/config.py` 中设置 Vina 的网格中心与尺寸；也可用 `scripts/binding_site_analyzer.py` 进行结合位点几何/表面分析
- 配体（Ligands）
  - 种子/示例：`data/ligands.sdf`；运行时由 `scripts/ligand_generator.py` 生成更大规模候选（去重、规范化、SMILES/SD 导出）
  - 3D 构象：RDKit ETKDG 生成 + 能量最小化；对接阶段自动处理
- 蛋白-配体对（Protein–Ligand Pairs）
  - 研究用途：`data/P-L/`（用于二分类或交互建模的示例数据）
  - 特征：见 `deep_learning/data/featurizers.py`
- 结果目录（Results）
  - 每次运行自动创建 `results/run_YYYYMMDD_xxx/`，含 `ligands/`、`docking/`、`admet/`、`visualization_2d|3d/`、`reports/`

## 方法与模型（Methods & Models）

- 计算化学与规则方法
  - 对接：AutoDock Vina（`scripts/molecular_docking.py` 或 `scripts/docking_engine.py` 批量流程，含 Meeko/RDKit 预处理）
  - ADMET：RDKit 描述符（MW、LogP、HBD/HBA、RotB、TPSA、芳香性等）与 Lipinski 五规则（`scripts/admet_analyzer.py`）
  - 可视化：2D（`scripts/visualization/visualization_2d.py`）、3D（`scripts/visualization_3d.py`）
- 外部生成模型（可选）
  - CMD-GEN 集成（`scripts/cmdgen_integration.py`）：
    - DiffPhar：由蛋白结构推断药效团点
    - GCPG：在药效团条件下生成分子（支持过滤与回退）
- 深度学习研究模块（`deep_learning/models/`）
  - EquivariantGNN（SE(3)-等变 GNN）：输入蛋白/配体图结构与三维坐标，输出节点/图级表征与亲和力评分
  - PocketLigandTransformer：口袋氨基酸序列/几何与配体 token 的跨模态交叉注意力，输出 P-L 交互表征
  - Pocket-conditioned Diffusion：以口袋条件控制的分子生成扩散过程
  - MultiTaskDiscriminator：对生成/筛选分子进行合成难度、毒性等多任务打分
  - PLPairClassifier：口袋-配体是否匹配的轻量二分类器

以上深度学习模块已给出可运行的研究版实现/占位，便于后续替换和扩展。

## 核心模型流程图（PNG）

以下为核心深度学习模型的流程图（PNG），位于 `docs/diagrams/png/`：

- 口袋条件扩散模型（Pocket-Conditioned Diffusion）

  ![Pocket-Conditioned Diffusion](docs/diagrams/png/fig_pocket_diffusion.png)

- Pocket–Ligand 交叉注意力 Transformer（Pocket–Ligand Transformer）

  ![Pocket–Ligand Transformer](docs/diagrams/png/fig_pl_transformer.png)

- SE(3)-等变图神经网络（SE(3)-Equivariant GNN）

  ![SE(3)-Equivariant GNN](docs/diagrams/png/fig_equivariant_gnn.png)

- 多任务判别器（Multi-Task Discriminator）

  ![Multi-Task Discriminator](docs/diagrams/png/fig_multitask_discriminator.png)

## Architecture Diagrams (English, Landscape)

Images only (PNG, high resolution):

- End-to-End Pipeline

  ![End-to-End Pipeline](docs/diagrams/end_to_end_pipeline.png)

- Data Featurization to Model Inputs

  ![Data Featurization to Model Inputs](docs/diagrams/featurization_to_inputs.png)

- SE(3)-Equivariant GNN

  ![SE(3)-Equivariant GNN](docs/diagrams/se3_equivariant_gnn.png)

- Pocket–Ligand Cross-attention Transformer

  ![Pocket–Ligand Transformer](docs/diagrams/pocket_ligand_transformer.png)

- Pocket-conditioned Diffusion Model

  ![Pocket-conditioned Diffusion](docs/diagrams/pocket_conditioned_diffusion.png)

- Multi-task Discriminator

  ![Multi-task Discriminator](docs/diagrams/multitask_discriminator.png)

- Pocket–Ligand Pair Classifier

  ![Pocket–Ligand Pair Classifier](docs/diagrams/pl_pair_classifier.png)

## 端到端工作流程（End-to-End Pipeline）

1) 配体生成（Ligand Generation）
   - 规则/模板/片段扩展或 CMD-GEN 生成；产出 CSV/SMILES/SD 到 `results/.../ligands/`
2) 分子对接（Docking）
   - 调用 AutoDock Vina，自动准备 pdbqt 与 3D 构象；产出 `docking/docking_results.csv`
3) ADMET 分析（ADMET Filtering）
   - 计算描述符与合规性；产出 `admet/admet_results.csv`
4) 可视化与报告（Visualization & Reports）
   - 2D：Top-N 单体图/网格与 HTML 报告 → `visualization_2d/`
   - 3D：单分子/复合物/结合位点 + 交互仪表板/综合报告 → `visualization_3d/`
5) 结果管理（Result Management）
   - 全流程产物以 run 归档，包含 `run_info.json` 与关键文件清单，便于复现

命令行一键运行：
```bash
python run_full_workflow.py
```
或通过 Web 界面在“完整工作流”页交互运行。

## 实验结果与示例（Results & Examples）

- 典型运行会生成：
  - 数十至上百个候选分子（`ligands/`）
  - `docking_results.csv`（包含亲和力、构象信息、pose 路径）
  - `admet_results.csv`（描述符与合规性标签）
  - `visualization_3d/` 下的 `interactive_dashboard.html` 与 `comprehensive_3d_report.html`
- Top-N 的确定：综合对接亲和力（越低越优）与基本 ADMET 合规性筛选
- 数值会随蛋白、参数与生成策略而变化，请以对应 run 目录中的 CSV/HTML 为准

## 复现实验与扩展（Reproducibility & Extension）

- 复现实验：
  - 固定输入（蛋白/口袋/参数），运行 `run_full_workflow.py`，比较不同生成策略或对接参数的差异
- 扩展方向：
  - 挂载外部 SOTA 模型（在 `deep_learning_pipeline.py` 与 `scripts/external_*_infer.py` 提供接口）
  - 增加自定义 ADMET 任务或替换评分标准
  - 批量化多蛋白/多口袋评估

## 外部工具与致谢（Acknowledgements）

- AutoDock Vina、Meeko、RDKit、py3Dmol、Plotly
- CMD-GEN（DiffPhar/GCPG），作为可选外部生成能力

## 技术特色

### 🔬 科学严谨
- 基于最新的深度学习和计算化学方法
- 集成多种成熟的药物设计工具
- 严格的验证和评估流程

### 🚀 高效智能
- GPU加速的深度学习计算
- 并行化的分子对接和分析
- 智能的分子生成和优化

### 🌐 用户友好
- 直观的Web界面操作
- 实时的结果可视化
- 详细的分析报告生成

### 🔧 模块化设计
- 松耦合的模块架构
- 易于扩展和维护
- 支持自定义配置

## 开发团队

本项目由专业的计算化学和人工智能团队开发，致力于为PRRSV药物研发提供先进的计算工具。

## 许可证

本项目仅供学术研究使用。

## 联系我们

如有问题或建议，请通过以下方式联系：
- 项目主页: [GitHub链接]
- 邮箱: [联系邮箱]

---

## 项目总结

本项目面向 PRRSV 病毒衣壳蛋白–整合素 PPI 抑制剂的小分子发现，集成“深度学习生成 → 分子对接 → ADMET 分析 → 2D/3D 可视化 → 结果管理”的完整工作流，支持成千上万级分子的批量生成与筛选，强调可追溯、可复现、可扩展。

### 功能矩阵
- **[分子生成]** 深度学习生成（Transformer/Diffusion 占位实现 + 外部模型挂载），规则/模板库动态扩展（上千唯一分子），Top-K 优化与 Final-N 收敛。
- **[分子对接]** AutoDock Vina 批量对接；RDKit+Meeko 构象、加氢与优化；结果归档为 CSV。
- **[ADMET 分析]** RDKit 描述符与 Lipinski 合规性；无 RDKit 环境自动启用“简化模式”。
- **[2D 可视化]** Top-N（上限 1000）单体图与网格图、HTML 报告；无对接结果时自动回退到配体 CSV。
- **[3D 可视化]** 单分子 3D、蛋白–配体复合物、结合位点分析、Plotly 交互仪表板、综合报告；无对接结果时回退到配体 CSV；输出保存至对应 run 的 `visualization_3d/`。
- **[结果管理]** 以 run 为单位的目录结构，自动复制深度学习 Phase2/Phase3 产物到 `ligands/`，全流程可追溯。

### 采用的模型与算法（可挂载 SOTA 模型）
- **[生成模型]** 现已提供“占位生成 + 化学空间动态扩展”能力，并预留外部口袋条件 Transformer、Diffusion + 强化学习（RL）等模型的对接接口（`deep_learning_pipeline.py`）。
- **[对接评分]** AutoDock Vina；RDKit 与 Meeko 用于 3D 构象生成与能量最小化（`scripts/docking_engine.py`）。
- **[ADMET 指标]** MW、LogP、HBD/HBA、RotB、TPSA、芳香性与环计数、Lipinski 规则（`scripts/admet_analyzer.py`）。
- **[可视化]** 2D 基于 RDKit rdMolDraw2D（Cairo/SVG）并带 PIL 回退；3D 基于 py3Dmol 与 Plotly（`scripts/visualization_3d.py`）。

### 关键创新点
- **[化学空间动态扩展]** 引入苯并唑/五元唑/三嗪/噻二唑/联苯/三联苯/亚当烷/双环等骨架；系统生成 o/m/p 位点二取代苯以及三取代模板，显著提升去重后的唯一分子数（软上限 ~8000）。
- **[强健的回退策略]** 2D/3D 在无对接结果时自动回退到配体 CSV（`ligands/`），不中断可视化与分析；RDKit/py3Dmol 缺失则启用简化模式。
- **[高吞吐可视化]** 2D Top-N 上限提升到 1000，支持大规模候选的快速筛查。
- **[全流程可追溯]** run 级目录管理与文件复制、步骤记录。

### 生成分子效果与评估指标（示例）
- **[多样性]** 覆盖卤素化芳环、腈/硝基/酰胺、五/六元杂环、稠环、多环芳烃、脂环（亚当烷/双环）等；o/m/p 位点控制增强空间构型多样性。
- **[典型分布]** MW 常见 100–400 Da（可拓展至 ~500）；LogP 多在 −1~5；亲和力用于 Top-N 排序筛选。
- **[报告内容]**
  - 2D：单体图、网格图、HTML 摘要报告（`visualization_2d/`）
  - 3D：单分子 3D、复合物、结合位点、交互仪表板与综合报告（`visualization_3d/`）
- 注：具体数值随 run 的参数/模型/对接条件变化，以报告与仪表板为准。

### 典型工作流
1. **分子生成**：设置“生成分子数量”（UI 支持 10000 上限）；Phase2 生成 → Phase3 优化（Top-K 随请求量动态放大）。阶段产物 CSV 自动复制到当前 run 的 `ligands/`。
2. **分子对接**：AutoDock Vina 批量对接，结果写入 `docking/docking_results.csv`。
3. **ADMET 分析**：批量计算描述符与 Lipinski 合规性，生成分析 CSV/报告。
4. **可视化**：
   - 2D：按亲和力排序的 Top-N（≤1000）+ 网格 + 报告 → `visualization_2d/`
   - 3D：单分子/复合物/位点 + 仪表板 + 综合报告 → `visualization_3d/`

### 实际工程结构（要点）
```
HJD/
├── unified_web_interface.py         # 统一 Web 界面（参数、回退、运行目录选择器等）
├── deep_learning_pipeline.py        # 生成与优化流水线（库扩展 + 外部模型挂载接口）
├── scripts/
│   ├── docking_engine.py            # AutoDock Vina 批量对接 + Meeko/RDKit 预处理
│   ├── molecule_2d_generator.py     # 2D 可视化（Top-N≤1000；对接优先、配体回退）
│   ├── visualization_3d.py          # 3D 可视化（单分子/复合物/位点/仪表板/综合报告）
│   ├── result_manager.py            # run 级目录管理与文件复制、步骤记录
│   └── admet_analyzer.py            # ADMET 描述符与 Lipinski 规则（简化模式回退）
└── results/
    └── run_YYYYMMDD_xxx/
        ├── ligands/                 # 生成/优化产物 CSV（Phase2/3 自动复制到此）
        ├── docking/                 # docking_results.csv
        ├── visualization_2d/        # individual/、grids/、reports/
        ├── visualization_3d/        # 单页可视化与综合报告 HTML
        └── reports/                 # 其它分析报告
```

### 依赖与运行
- **[环境]** Python 3.9+；建议安装 RDKit、py3Dmol、Plotly 与 Streamlit。
- **[安装]** `pip install -r requirements.txt`
- **[启动]** `python start_project.py`，浏览器访问 `http://localhost:8501`
- **[注意]**
  - 2D/3D 可视化在无对接结果时，会自动回退到当前/最新 run 的 `ligands/*.csv`
  - 3D 可视化报告固定保存至当前 run 的 `visualization_3d/`

### 示例可视化（来自一次示例运行）

> 注：以下示例图为仓库内现有文件，实际运行会在你的 `results/run_*/visualization_2d|3d/` 下生成对应图表与报告。

- **2D 分子网格图（Top-N）**
  ![2D Grid](results/run_20250918_008/visualization_2d/grids/Top_165_Molecules_grid.png)

- **ADMET/评分示例图**
  结合亲和力分布：
  ![Binding Affinity Distribution](experiment_report/binding_affinity_distribution.png)

  Top-10 分子柱状：
  ![Top 10 Molecules](experiment_report/top_10_molecules.png)

  Lipinski 合规性：
  ![Lipinski Compliance](experiment_report/lipinski_compliance.png)

  ADMET 性质（示例）：
  ![ADMET Properties](experiment_report/admet_properties.png)

- **3D 交互式仪表盘/报告**
  生成 3D 报告后，将在对应 run 的 `visualization_3d/` 下生成：
  - `interactive_dashboard.html`（交互式）
  - `interactive_dashboard.png`（静态导出，需 `kaleido`）
  - `comprehensive_3d_report.html`（综合报告）

> 若需静态 PNG 导出，请确保安装 `kaleido`（已加入 `requirements.txt`）。未安装时将自动跳过 PNG 导出，但不影响 HTML 报告生成。

---

**PRRSV深度学习抑制剂设计平台 - 让药物设计更智能**
