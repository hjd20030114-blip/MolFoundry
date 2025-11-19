#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRRSV抑制剂设计统一Web界面
合并所有功能模块的完整Web平台
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import base64
import json
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime
import threading
import http.server
import socketserver
from urllib.parse import quote

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'scripts'))

# 导入结果管理器
from scripts.result_manager import result_manager

# 导入项目模块
try:
    from scripts.config import *
    from scripts.ligand_generator import LigandGenerator
    from scripts.molecular_docking import MolecularDocking
    from scripts.admet_analyzer import ADMETAnalyzer
except ImportError:
    # 如果scripts导入失败，尝试直接导入
    from config import *
    from ligand_generator import LigandGenerator
    from molecular_docking import MolecularDocking
    from admet_analyzer import ADMETAnalyzer

# 尝试导入3D查看器（可选）
try:
    from scripts.streamlit_3d_viewer import display_molecule_3d, display_protein_ligand_complex, create_molecular_comparison_viewer
    VIEWER_3D_AVAILABLE = True
except ImportError:
    VIEWER_3D_AVAILABLE = False

# 尝试导入深度学习模块
try:
    from deep_learning_pipeline import DeepLearningPipeline
    import torch
    HAS_DEEP_LEARNING = True
    DEVICE_INFO = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
except ImportError:
    HAS_DEEP_LEARNING = False
    DEVICE_INFO = "深度学习模块未安装"
except ImportError:
    VIEWER_3D_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="PRRSV抑制剂设计统一平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UnifiedPRRSVInterface:
    """统一的PRRSV Web界面"""

    def __init__(self):
        """初始化界面"""
        # 初始化组件
        self.ligand_generator = LigandGenerator()
        self.molecular_docking = MolecularDocking()
        self.admet_analyzer = ADMETAnalyzer()

        # 初始化结果管理器
        try:
            from scripts.result_manager import result_manager
            self.result_manager = result_manager
        except ImportError:
            st.warning("⚠️ 结果管理器导入失败，将使用默认目录")
            self.result_manager = None

        # 初始化会话状态
        if 'workflow_running' not in st.session_state:
            st.session_state.workflow_running = False
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0

        # 初始化结果管理相关状态
        if 'current_run_started' not in st.session_state:
            st.session_state.current_run_started = False
        if 'current_run_dir' not in st.session_state:
            st.session_state.current_run_dir = None

    def open_file_in_browser(self, file_path, description="文件"):
        """
        尝试在浏览器中打开文件的辅助函数

        Args:
            file_path: 文件路径
            description: 文件描述
        """
        try:
            # 转换为Path对象
            if isinstance(file_path, str):
                file_path = Path(file_path)

            # 确保路径存在
            if not file_path.exists():
                st.error(f"❌ {description}不存在")
                st.info(f"📁 查找路径: {file_path}")
                st.info("💡 请先生成相应的报告文件")
                return False

            # 转换为正确的file URL格式
            abs_path = file_path.absolute()
            if os.name == 'nt':  # Windows
                file_url = f"file:///{str(abs_path)}".replace("\\", "/")
            else:  # Unix/Linux/Mac
                file_url = f"file://{str(abs_path)}"

            # 显示文件信息
            st.info(f"📁 文件位置: {abs_path}")
            st.info(f"🌐 URL: {file_url}")

            # 尝试打开
            import webbrowser
            success = webbrowser.open(file_url)

            if success:
                st.success(f"✅ 已在浏览器中打开{description}")
                st.balloons()
                return True
            else:
                st.warning(f"⚠️ 无法自动打开{description}")
                st.info(f"💡 请手动复制以下路径到浏览器地址栏:")
                st.code(str(abs_path))

                # 提供备选方案
                st.markdown("### 🔧 备选打开方式:")
                st.markdown(f"1. **复制文件路径**: `{abs_path}`")
                st.markdown(f"2. **复制URL**: `{file_url}`")
                st.markdown("3. **手动导航**: 在文件管理器中找到文件并双击打开")
                return False

        except Exception as e:
            st.error(f"❌ 打开{description}时出错: {e}")
            st.exception(e)
            st.info(f"💡 请手动复制以下路径到浏览器地址栏:")
            st.code(str(file_path))
            return False

    def start_new_run(self):
        """开始新的运行"""
        try:
            # 创建新的运行目录
            run_dir = result_manager.create_new_run_directory()

            # 更新会话状态
            st.session_state.current_run_started = True
            st.session_state.current_run_dir = str(run_dir)

            st.success(f"🎯 新运行已开始！结果将保存到: {run_dir.name}")
            st.info("💡 本次运行的所有结果（配体生成、分子对接、ADMET分析、2D/3D可视化）都将保存在此目录中")

            return run_dir

        except Exception as e:
            st.error(f"创建新运行目录失败: {e}")
            return None

    def ensure_run_started(self):
        """确保运行已开始"""
        if not st.session_state.current_run_started:
            st.warning("⚠️ 请先开始新的运行！")
            if st.button("🚀 开始新运行"):
                return self.start_new_run() is not None
            return False
        return True

    def create_file_server_link(self, file_path, port=8508):
        """
        创建本地HTTP服务器链接

        Args:
            file_path: 文件路径
            port: 服务器端口
        """
        try:
            from pathlib import Path
            import threading
            import http.server
            import socketserver
            import os

            file_path = Path(file_path)
            if not file_path.exists():
                return None

            # 获取文件所在目录
            directory = file_path.parent
            filename = file_path.name

            # 创建HTTP服务器
            class CustomHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(directory), **kwargs)

            # 检查端口是否可用
            try:
                with socketserver.TCPServer(("", port), CustomHandler) as httpd:
                    # 启动服务器线程
                    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                    server_thread.start()

                    # 返回访问链接
                    return f"http://localhost:{port}/{filename}"
            except OSError:
                # 端口被占用，尝试下一个端口
                return self.create_file_server_link(file_path, port + 1)

        except Exception as e:
            st.error(f"创建文件服务器失败: {e}")
            return None

    def render_header(self):
        """渲染页面头部"""
        st.markdown("""
        <div class="main-header">
            <h1>🧬 PRRSV抑制剂设计统一平台</h1>
            <p>集成分子生成、对接分析、ADMET评估和3D可视化的完整解决方案</p>
        </div>
        """, unsafe_allow_html=True)

        # 显示当前运行状态
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.session_state.current_run_started:
                run_dir = Path(st.session_state.current_run_dir)
                st.success(f"🎯 当前运行: {run_dir.name}")
            else:
                st.info("💡 尚未开始运行")

        with col2:
            if st.button("🚀 开始新运行", use_container_width=True):
                self.start_new_run()
                st.rerun()

        with col3:
            if st.session_state.current_run_started:
                if st.button("📊 运行摘要", use_container_width=True):
                    self.show_run_summary()

        st.divider()

    def show_run_summary(self):
        """显示运行摘要"""
        if not st.session_state.current_run_started:
            st.warning("尚未开始运行")
            return

        try:
            # 设置当前运行目录
            result_manager.current_run_dir = Path(st.session_state.current_run_dir)

            # 获取运行摘要
            summary = result_manager.get_run_summary()

            if summary:
                st.subheader("📊 当前运行摘要")

                # 基本信息
                run_info = summary.get("run_info", {})
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("运行ID", run_info.get("run_id", "未知"))
                    st.metric("开始时间", run_info.get("start_time", "未知")[:19])

                with col2:
                    st.metric("状态", run_info.get("status", "未知"))
                    st.metric("完成步骤", len(run_info.get("steps_completed", [])))

                # 文件统计
                file_counts = summary.get("file_counts", {})
                st.subheader("📁 生成文件统计")

                cols = st.columns(3)
                with cols[0]:
                    st.metric("配体文件", file_counts.get("ligands", 0))
                    st.metric("对接结果", file_counts.get("docking", 0))

                with cols[1]:
                    st.metric("ADMET分析", file_counts.get("admet", 0))
                    st.metric("2D可视化", file_counts.get("visualization_2d", 0))

                with cols[2]:
                    st.metric("3D可视化", file_counts.get("visualization_3d", 0))
                    st.metric("报告文件", file_counts.get("reports", 0))

                st.metric("总文件数", summary.get("total_files", 0))

                # 完成的步骤
                if run_info.get("steps_completed"):
                    st.subheader("✅ 已完成步骤")
                    for step in run_info["steps_completed"]:
                        st.success(f"✓ {step}")

        except Exception as e:
            st.error(f"获取运行摘要失败: {e}")

    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🎛️ 功能导航")
        
        # 主要功能选择
        page_options = [
            "🏠 项目概览",
            "🧪 配体生成",
            "🔬 分子对接",
            "📈 ADMET分析",
            "📊 结果分析",
            "🖼️ 2D分子图像",
            "🌐 3D可视化",
            "🚀 完整工作流程",
            "📁 文件管理",
            "⚙️ 系统设置"
        ]

        # 如果深度学习模块可用，仅保留 AI 训练；深度学习流水线已并入“🧪 配体生成”
        if HAS_DEEP_LEARNING:
            page_options.insert(-2, "🧠 AI模型训练")

        page = st.sidebar.selectbox(
            "选择功能模块",
            page_options
        )
        
        # 快速操作
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 快速操作")
        
        if st.sidebar.button("🔄 重置所有数据"):
            self.reset_session_data()
            st.sidebar.success("数据已重置")
        
        if st.sidebar.button("💾 保存当前状态"):
            self.save_session_state()
            st.sidebar.success("状态已保存")
        
        if st.sidebar.button("📊 生成完整报告"):
            self.generate_comprehensive_report()
        
        # 系统状态
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 系统状态")
        
        # 检查各模块状态
        modules_status = {
            "配体生成器": "✅" if hasattr(self, 'ligand_generator') else "❌",
            "对接引擎": "✅" if hasattr(self, 'docking_engine') else "❌",
            "ADMET分析器": "✅" if hasattr(self, 'admet_analyzer') else "❌",
            "3D可视化": "✅" if VIEWER_3D_AVAILABLE else "❌",
            "深度学习": "✅" if HAS_DEEP_LEARNING else "❌"
        }

        # 显示计算设备信息
        if HAS_DEEP_LEARNING:
            st.sidebar.write(f"🖥️ 计算设备: {DEVICE_INFO}")
        
        for module, status in modules_status.items():
            st.sidebar.write(f"{status} {module}")
        
        return page
    
    def show_project_overview(self):
        """显示项目概览"""
        st.title("🏠 项目概览")
        
        # 项目介绍
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 项目目标
            本平台旨在设计靶向PRRSV病毒衣壳蛋白的特异性小分子抑制剂，通过干预PRRSV衣壳蛋白与整合素的相互作用来阻断病毒感染。
            
            ### 🔬 核心功能
            - **🧪 智能配体生成**: 基于模板库和CMD-GEN深度学习模型
            - **🔬 高精度分子对接**: AutoDock Vina引擎，准确预测结合模式
            - **📈 全面ADMET分析**: 药物类似性质和毒性预测
            - **🌐 3D可视化**: 交互式分子和蛋白质结构展示
            - **📊 智能结果分析**: 多维度评分和排序系统
            """)
        
        with col2:
            # 项目统计
            st.markdown("### 📊 项目统计")
            
            # 检查结果文件
            results_dir = Path("results")
            if results_dir.exists():
                result_dirs = list(results_dir.glob("*"))
                prrsv_dirs = [d for d in result_dirs if d.name.startswith("prrsv_inhibitors_")]
                ppi_dirs = [d for d in result_dirs if d.name.startswith("ppi_design_")]
                
                st.metric("PRRSV结果", len(prrsv_dirs))
                st.metric("PPI设计结果", len(ppi_dirs))
                st.metric("总结果目录", len(result_dirs))
            else:
                st.info("暂无结果数据")
        
        # 最新结果展示
        st.markdown("---")
        st.subheader("📈 最新结果")
        
        latest_results = self.get_latest_results()
        if latest_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>生成的分子数</p>
                </div>
                """.format(latest_results.get('total_molecules', 0)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>{:.2f}</h3>
                    <p>最佳结合能 (kcal/mol)</p>
                </div>
                """.format(latest_results.get('best_binding', 0)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>{:.1f}%</h3>
                    <p>Lipinski符合率</p>
                </div>
                """.format(latest_results.get('lipinski_rate', 0)), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>处理时间 (秒)</p>
                </div>
                """.format(latest_results.get('processing_time', 'N/A')), unsafe_allow_html=True)
        
        # 快速开始
        st.markdown("---")
        st.subheader("🚀 快速开始")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 开始配体生成", type="primary", use_container_width=True):
                st.session_state.page = "🧪 配体生成"
                st.rerun()
        
        with col2:
            if st.button("🚀 运行完整工作流程", type="primary", use_container_width=True):
                st.session_state.page = "🚀 完整工作流程"
                st.rerun()
        
        with col3:
            if st.button("🌐 查看3D可视化", type="primary", use_container_width=True):
                st.session_state.page = "🌐 3D可视化"
                st.rerun()
    
    def show_ligand_generation(self):
        """显示配体生成页面"""
        st.title("🧪 配体生成")

        st.markdown("""
        <div class="feature-card">
            <h4>🎯 功能说明</h4>
            <p>基于模板库和优化算法生成候选小分子化合物，支持传统方法和CMD-GEN深度学习模型。</p>
        </div>
        """, unsafe_allow_html=True)

        # 检查是否已开始运行
        if not self.ensure_run_started():
            return
        
        # 参数设置
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ 参数设置")
            
            num_ligands = st.slider("生成分子数量", 10, 10000, 100)
            generation_method = st.selectbox(
                "生成方法",
                [
                    "优化生成",
                    "随机生成",
                    "CMD-GEN生成",
                    "深度学习生成 (Transformer/Diffusion)"
                ]
            )
            # 深度学习生成所需的附加参数
            protein_file_dl = None
            target_affinity_dl = -6.0
            use_ext_tf = False
            use_ext_df = False
            tf_script = ''
            tf_ckpt = ''
            tf_module = ''
            tf_function = ''
            tf_kwargs_json = ''
            df_script = ''
            df_ckpt = ''
            df_module = ''
            df_function = ''
            df_kwargs_json = ''
            top_k = 50
            final_top_n = 100
            if generation_method == "深度学习生成 (Transformer/Diffusion)":
                st.markdown("---")
                st.markdown("#### 🤖 深度学习生成参数")
                # 受体/蛋白结构文件选择（与分子对接相同来源）
                available_receptors = []
                receptor_options = [
                    ("data/1p65.pdb", "🦠 PRRSV核衣壳蛋白 (1p65.pdb) ✅ 推荐"),
                    ("data/AF-F1SR53-F1-model_v4.pdb", "🧬 整合素复合物2 (AF-F1SR53) ✅"),
                    ("data/AF-Q9GLP0-F1-model_v4.pdb", "🧬 整合素复合物1 (AF-Q9GLP0) ✅"),
                ]
                for file_path, description in receptor_options:
                    if os.path.exists(file_path):
                        available_receptors.append((file_path, description))
                if available_receptors:
                    protein_file_dl = st.selectbox(
                        "选择蛋白质PDB文件",
                        [r[0] for r in available_receptors],
                        format_func=lambda x: next(r[1] for r in available_receptors if r[0] == x)
                    )
                else:
                    st.warning("⚠️ 未找到内置PDB，请在‘🔬 分子对接’中准备或将PDB放置在data/目录下")
                    uploaded = st.file_uploader("或上传蛋白质PDB文件", type=["pdb"], key="dl_pdb_uploader")
                    if uploaded is not None:
                        tmp_path = f"temp_dl_protein_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdb"
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded.getbuffer())
                        protein_file_dl = tmp_path
                target_affinity_dl = st.slider("目标结合亲和力 (kcal/mol)", -10.0, -3.0, -6.0, 0.1, key="dl_target_affinity")

                with st.expander("⚙️ 高级选项：外部Transformer/Diffusion推理与规模设置", expanded=False):
                    st.markdown("##### 🔌 外部推理脚本")
                    use_ext_tf = st.checkbox("使用外部Transformer推理", value=False)
                    tf_cols = st.columns(2)
                    with tf_cols[0]:
                        tf_script = st.text_input("Transformer脚本路径", value="scripts/external_transformer_infer.py")
                    with tf_cols[1]:
                        tf_ckpt = st.text_input("Transformer检查点路径(可选)", value="")
                    tf_cols2 = st.columns(3)
                    with tf_cols2[0]:
                        tf_module = st.text_input("Transformer模块(可选)", value="", placeholder="mypkg.mymodule")
                    with tf_cols2[1]:
                        tf_function = st.text_input("Transformer函数(可选)", value="", placeholder="generate")
                    with tf_cols2[2]:
                        tf_kwargs_json = st.text_input("Transformer参数JSON(可选)", value="", placeholder='{"temperature":1.0}')

                    use_ext_df = st.checkbox("使用外部Diffusion优化", value=False)
                    df_cols = st.columns(2)
                    with df_cols[0]:
                        df_script = st.text_input("Diffusion脚本路径", value="scripts/external_diffusion_infer.py")
                    with df_cols[1]:
                        df_ckpt = st.text_input("Diffusion检查点路径(可选)", value="")
                    df_cols2 = st.columns(3)
                    with df_cols2[0]:
                        df_module = st.text_input("Diffusion模块(可选)", value="", placeholder="mypkg.mydiffusion")
                    with df_cols2[1]:
                        df_function = st.text_input("Diffusion函数(可选)", value="", placeholder="optimize")
                    with df_cols2[2]:
                        df_kwargs_json = st.text_input("Diffusion参数JSON(可选)", value="", placeholder='{"steps":50}')

                    st.markdown("##### 📏 规模设置")
                    top_k = st.slider(
                        "优化Top-K（第三阶段）",
                        min_value=10,
                        max_value=max(1000, num_ligands * 5),
                        value=min(max(10, num_ligands * 2), max(1000, num_ligands * 5)),
                        step=10
                    )
                    final_top_n = st.slider("最终保留数量（第三阶段产物）", min_value=10, max_value=max(10, num_ligands), value=num_ligands, step=10)
            
            filter_molecules = st.checkbox("应用分子过滤", True)
            save_results = st.checkbox("保存结果", True)
            
            if st.button("🚀 开始生成", type="primary"):
                self.run_ligand_generation(
                    num_ligands,
                    generation_method,
                    filter_molecules,
                    save_results,
                    protein_file_dl=protein_file_dl,
                    target_affinity_dl=target_affinity_dl,
                    use_ext_tf=use_ext_tf,
                    use_ext_df=use_ext_df,
                    tf_script=tf_script,
                    tf_ckpt=tf_ckpt,
                    df_script=df_script,
                    df_ckpt=df_ckpt,
                    top_k=top_k,
                    final_top_n=final_top_n,
                    tf_module=tf_module,
                    tf_function=tf_function,
                    tf_kwargs_json=tf_kwargs_json,
                    df_module=df_module,
                    df_function=df_function,
                    df_kwargs_json=df_kwargs_json
                )
        
        with col2:
            st.subheader("📊 生成结果")
            
            if 'ligands' in st.session_state:
                ligands_df = pd.DataFrame(st.session_state.ligands)
                
                st.success(f"✅ 成功生成 {len(ligands_df)} 个配体分子")
                
                # 显示分子信息
                st.dataframe(ligands_df.head(10), use_container_width=True)
                
                # 分子性质分布
                if 'molecular_weight' in ligands_df.columns:
                    fig = px.histogram(ligands_df, x='molecular_weight', 
                                     title='分子量分布', nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请点击'开始生成'按钮生成配体分子")
    
    def show_molecular_docking(self):
        """显示分子对接页面"""
        st.title("🔬 分子对接")
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 功能说明</h4>
            <p>使用AutoDock Vina进行高精度分子对接，预测配体与PRRSV衣壳蛋白的结合模式和亲和力。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 检查是否有配体数据
        if 'ligands' not in st.session_state:
            st.warning("⚠️ 请先生成配体分子")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("前往配体生成", type="primary"):
                    st.session_state.page = "🧪 配体生成"
                    st.rerun()

            with col_b:
                with st.expander("📝 或手动输入SMILES"):
                    manual_smiles = st.text_area(
                        "输入SMILES字符串（每行一个）：",
                        placeholder="CCO\nCC(=O)O\nc1ccccc1",
                        height=100
                    )

                    if st.button("🔄 使用输入的SMILES"):
                        if manual_smiles.strip():
                            smiles_list = [s.strip() for s in manual_smiles.strip().split('\n') if s.strip()]
                            if smiles_list:
                                # 创建配体数据
                                ligands = []
                                for i, smiles in enumerate(smiles_list, 1):
                                    ligands.append({
                                        'smiles': smiles,
                                        'name': f'manual_ligand_{i}',
                                        'source': 'manual_input'
                                    })
                                st.session_state.ligands = ligands
                                st.success(f"✅ 成功添加 {len(ligands)} 个配体分子")
                                st.rerun()
                            else:
                                st.error("❌ 请输入有效的SMILES字符串")
                        else:
                            st.error("❌ 请输入SMILES字符串")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ 对接设置")
            
            # 检查可用的受体文件
            available_receptors = []
            receptor_options = [
                ("data/1p65.pdb", "🦠 PRRSV核衣壳蛋白 (1p65.pdb) ✅ 推荐"),
                ("data/1p65.pdbqt", "🦠 PRRSV核衣壳蛋白 (1p65.pdbqt) ✅ 推荐"),
                ("data/AF-F1SR53-F1-model_v4.pdb", "🧬 整合素复合物2 (AF-F1SR53) ✅"),
                ("data/AF-Q9GLP0-F1-model_v4.pdb", "🧬 整合素复合物1 (AF-Q9GLP0) ✅"),
            ]

            for file_path, description in receptor_options:
                if os.path.exists(file_path):
                    available_receptors.append((file_path, description))

            if not available_receptors:
                st.error("❌ 没有找到可用的受体文件")
                return

            receptor_file = st.selectbox(
                "选择受体文件",
                [r[0] for r in available_receptors],
                format_func=lambda x: next(r[1] for r in available_receptors if r[0] == x)
            )

            # 显示文件状态
            if os.path.exists(receptor_file):
                file_size = os.path.getsize(receptor_file)
                st.success(f"✅ 受体文件存在 ({file_size} bytes)")
            else:
                st.error(f"❌ 受体文件不存在: {receptor_file}")
            
            exhaustiveness = st.slider("搜索精度", 8, 32, 16)
            num_poses = st.slider("生成构象数", 1, 20, 9)
            
            # 显示配体信息
            ligands = st.session_state.ligands
            st.info(f"📋 当前配体数量: {len(ligands)} 个")

            # 显示前几个配体的SMILES
            if len(ligands) > 0:
                st.write("**配体预览:**")
                for i, ligand in enumerate(ligands[:3]):
                    smiles = ligand.get('smiles', 'N/A') if isinstance(ligand, dict) else str(ligand)
                    st.code(f"{i+1}. {smiles}")
                if len(ligands) > 3:
                    st.write(f"... 还有 {len(ligands)-3} 个配体")

            if st.button("🚀 开始对接", type="primary"):
                self.run_molecular_docking(receptor_file, exhaustiveness, num_poses)
        
        with col2:
            st.subheader("📊 对接结果")

            if 'docking_results' in st.session_state:
                docking_df = st.session_state.docking_results

                st.success(f"✅ 成功对接 {len(docking_df)} 个配体")

                # 结合能分布
                fig = px.histogram(docking_df, x='binding_affinity',
                                 title='结合亲和力分布 (kcal/mol)', nbins=20,
                                 labels={'binding_affinity': '结合亲和力 (kcal/mol)', 'count': '分子数量'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # 对接质量统计
                excellent = len(docking_df[docking_df['binding_affinity'] < -8.0])
                good = len(docking_df[(docking_df['binding_affinity'] >= -8.0) & (docking_df['binding_affinity'] < -6.0)])
                moderate = len(docking_df[docking_df['binding_affinity'] >= -6.0])

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("优秀 (<-8.0)", excellent, f"{excellent/len(docking_df)*100:.1f}%")
                with col_stat2:
                    st.metric("良好 (-8.0~-6.0)", good, f"{good/len(docking_df)*100:.1f}%")
                with col_stat3:
                    st.metric("一般 (>-6.0)", moderate, f"{moderate/len(docking_df)*100:.1f}%")

                # 前10个最佳结果
                st.subheader("🏆 前10个最佳结果")
                top_10 = docking_df.head(10)
                display_cols = ['compound_id', 'binding_affinity']
                if 'molecular_weight' in docking_df.columns:
                    display_cols.append('molecular_weight')
                if 'logp' in docking_df.columns:
                    display_cols.append('logp')
                st.dataframe(top_10[display_cols], use_container_width=True)

                # 下载结果
                if st.button("💾 下载对接结果"):
                    csv = docking_df.to_csv(index=False)
                    st.download_button(
                        label="下载CSV文件",
                        data=csv,
                        file_name=f"docking_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("请点击'开始对接'按钮进行分子对接")

                # 显示对接参数说明
                st.markdown("""
                **📋 参数说明:**
                - **搜索精度**: 控制对接搜索的详尽程度，值越高精度越高但耗时越长
                - **生成构象数**: 每个配体生成的结合构象数量，通常5-20个
                - **受体文件**: 选择不同的蛋白质靶点进行对接
                """)
    
    def show_admet_analysis(self):
        """显示ADMET分析页面"""
        st.title("📈 ADMET分析")
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 功能说明</h4>
            <p>全面评估候选分子的药物类似性质，包括Lipinski规则、分子性质、毒性预测与溶解度预测（ESOL近似）。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 检查是否有配体数据
        if 'ligands' not in st.session_state:
            st.warning("⚠️ 请先生成配体分子")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ 分析设置")
            
            analysis_type = st.multiselect(
                "选择分析类型",
                ["Lipinski规则", "分子性质", "毒性预测", "溶解度预测"],
                default=["Lipinski规则", "分子性质"]
            )
            
            if st.button("🚀 开始分析", type="primary"):
                self.run_admet_analysis(analysis_type)
        
        with col2:
            st.subheader("📊 ADMET结果")
            
            if 'admet_results' in st.session_state:
                admet_df = st.session_state.admet_results
                
                st.success(f"✅ 成功分析 {len(admet_df)} 个分子")
                
                # Lipinski符合性统计
                if 'lipinski_compliant' in admet_df.columns:
                    compliant_count = admet_df['lipinski_compliant'].sum()
                    compliance_rate = (compliant_count / len(admet_df)) * 100
                    
                    st.metric("Lipinski符合率", f"{compliance_rate:.1f}%")
                # Veber/Egan 符合率
                met_cols = []
                cols = st.columns(2)
                if 'veber_compliant' in admet_df.columns:
                    veber_rate = (admet_df['veber_compliant'].sum() / len(admet_df)) * 100
                    with cols[0]:
                        st.metric("Veber符合率", f"{veber_rate:.1f}%")
                if 'egan_compliant' in admet_df.columns:
                    egan_rate = (admet_df['egan_compliant'].sum() / len(admet_df)) * 100
                    with cols[1]:
                        st.metric("Egan符合率", f"{egan_rate:.1f}%")
                
                # 分子性质分布
                if 'molecular_weight' in admet_df.columns and 'logp' in admet_df.columns:
                    fig = px.scatter(admet_df, x='molecular_weight', y='logp',
                                   title='分子量 vs LogP', 
                                   color='lipinski_compliant' if 'lipinski_compliant' in admet_df.columns else None)
                    st.plotly_chart(fig, use_container_width=True)

                # TPSA 分布
                if 'tpsa' in admet_df.columns:
                    fig_tpsa = px.histogram(admet_df, x='tpsa', nbins=30, title='TPSA 分布')
                    st.plotly_chart(fig_tpsa, use_container_width=True)

                # QED 分布
                if 'qed' in admet_df.columns and admet_df['qed'].notna().any():
                    fig_qed = px.histogram(admet_df, x='qed', nbins=30, title='QED 分布')
                    st.plotly_chart(fig_qed, use_container_width=True)

                # 溶解度分布（ESOL 近似）
                if 'predicted_logS' in admet_df.columns:
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        fig_logS = px.histogram(admet_df, x='predicted_logS', nbins=30, title='预测溶解度 logS 分布 (ESOL近似)')
                        st.plotly_chart(fig_logS, use_container_width=True)
                    with col_s2:
                        if 'solubility_class' in admet_df.columns:
                            vc_df = (
                                admet_df['solubility_class']
                                .value_counts(dropna=False)
                                .reset_index(name='count')
                                .rename(columns={'index': 'solubility_class'})
                            )
                            fig_sol = px.bar(
                                vc_df,
                                x='solubility_class', y='count',
                                title='溶解度等级分布', labels={'solubility_class':'溶解度等级','count':'数量'}
                            )
                            st.plotly_chart(fig_sol, use_container_width=True)

                # 毒性风险分布与警示
                if 'toxicity_risk_level' in admet_df.columns:
                    fig_tox = px.pie(admet_df, names='toxicity_risk_level', title='毒性风险等级占比')
                    st.plotly_chart(fig_tox, use_container_width=True)

                # 结果表格（展示关键字段）
                show_cols = [c for c in [
                    'compound_id','smiles','molecular_weight','logp','tpsa',
                    'rotatable_bonds','lipinski_compliant','lipinski_violations','lipinski_violation_details',
                    'veber_compliant','egan_compliant','qed',
                    'predicted_logS','solubility_class','toxicity_risk_level','toxicity_alerts','canonical_smiles'
                ] if c in admet_df.columns]
                if show_cols:
                    st.dataframe(admet_df[show_cols].head(50))

                # 下载结果
                st.markdown("---")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_bytes = admet_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载ADMET结果CSV", data=csv_bytes, file_name=f"admet_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
                with col_dl2:
                    # 若已保存到当前run，展示保存路径
                    try:
                        from scripts.result_manager import result_manager as _rm
                        crun = _rm.get_current_run_dir()
                        if crun and (crun/"admet"/"admet_results.csv").exists():
                            st.info(f"💾 已保存: {(crun/ 'admet' / 'admet_results.csv').absolute()}")
                    except Exception:
                        pass
            else:
                st.info("请点击'开始分析'按钮进行ADMET分析")
    
    def get_latest_results(self):
        """获取最新结果统计"""
        try:
            results_dir = Path("results")
            if not results_dir.exists():
                return None
            
            # 查找最新的PRRSV结果
            prrsv_dirs = [d for d in results_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("prrsv_inhibitors_")]
            
            if not prrsv_dirs:
                return None
            
            latest_dir = max(prrsv_dirs, key=lambda x: x.stat().st_mtime)
            
            # 读取结果文件
            ranked_file = latest_dir / "ranked_inhibitors.csv"
            if ranked_file.exists():
                df = pd.read_csv(ranked_file)
                
                return {
                    'total_molecules': len(df),
                    'best_binding': df['binding_affinity'].min() if 'binding_affinity' in df.columns else 0,
                    'lipinski_rate': (df['lipinski_compliant'].sum() / len(df)) * 100 if 'lipinski_compliant' in df.columns else 0,
                    'processing_time': 'N/A'
                }
            
        except Exception as e:
            st.error(f"读取结果失败: {e}")
        
        return None

    def run_ligand_generation(self, num_ligands, method, filter_molecules, save_results,
                              protein_file_dl=None, target_affinity_dl: float = -6.0,
                              use_ext_tf: bool = False, use_ext_df: bool = False,
                              tf_script: str = '', tf_ckpt: str = '',
                              tf_module: str = '', tf_function: str = '', tf_kwargs_json: str = '',
                              df_script: str = '', df_ckpt: str = '',
                              df_module: str = '', df_function: str = '', df_kwargs_json: str = '',
                              top_k: int = 50, final_top_n: int = 100):
        """运行配体生成"""
        with st.spinner("正在生成配体分子..."):
            try:
                # 先缓存旧的配体数据作为种子
                old_ligands = st.session_state.ligands if 'ligands' in st.session_state else None
                # 清除旧的配体数据，确保显示新生成的数据
                if 'ligands' in st.session_state:
                    del st.session_state.ligands

                # 设置当前运行目录
                result_manager.current_run_dir = Path(st.session_state.current_run_dir)
                # 若当前运行信息丢失（例如页面重载后），尝试恢复或初始化run_info
                try:
                    if not result_manager.current_run_info or 'steps_completed' not in result_manager.current_run_info:
                        info_file = result_manager.current_run_dir / "run_info.json"
                        if info_file.exists():
                            import json as _json
                            with open(info_file, 'r', encoding='utf-8') as _f:
                                result_manager.current_run_info = _json.load(_f)
                        else:
                            from datetime import datetime as _dt
                            rid = result_manager.current_run_dir.name
                            result_manager.current_run_info = {
                                "run_id": rid,
                                "start_time": _dt.now().isoformat(),
                                "date": rid.split('_')[1] if '_' in rid else '',
                                "run_number": int(rid.split('_')[-1]) if '_' in rid and rid.split('_')[-1].isdigit() else 1,
                                "status": "started",
                                "steps_completed": [],
                                "files_generated": {}
                            }
                            result_manager.save_run_info()
                except Exception:
                    # 最后兜底：确保必要键存在
                    result_manager.current_run_info = result_manager.current_run_info or {}
                    result_manager.current_run_info.setdefault("steps_completed", [])
                    result_manager.current_run_info.setdefault("files_generated", {})
                    result_manager.save_run_info()

                if method == "CMD-GEN生成":
                    # 尝试使用CMD-GEN，使用关键字参数确保正确传递
                    ligands = self.ligand_generator.generate_cmdgen_ligands(num_ligands=num_ligands)
                elif method == "深度学习生成 (Transformer/Diffusion)":
                    # 合并深度学习流水线：调用 DeepLearningPipeline 进行生成与优化
                    if not HAS_DEEP_LEARNING:
                        raise RuntimeError("深度学习模块不可用，请检查依赖安装")

                    # 选择蛋白文件
                    if not protein_file_dl or not os.path.exists(protein_file_dl):
                        # 尝试使用默认PDB
                        default_pdb = "data/1p65.pdb"
                        if os.path.exists(default_pdb):
                            protein_file_dl = default_pdb
                        else:
                            raise RuntimeError("未提供有效的蛋白质PDB文件")

                    pipeline = DeepLearningPipeline()
                    # 外部推理集成设置
                    pipeline.set_external_options({
                        'use_real_transformer': bool(use_ext_tf),
                        'use_real_diffusion': bool(use_ext_df),
                        'transformer_script': tf_script or 'scripts/external_transformer_infer.py',
                        'diffusion_script': df_script or 'scripts/external_diffusion_infer.py',
                        'transformer_checkpoint': tf_ckpt or None,
                        'diffusion_checkpoint': df_ckpt or None,
                        'tf_module': tf_module or None,
                        'tf_function': tf_function or None,
                        'tf_kwargs_json': tf_kwargs_json or None,
                        'df_module': df_module or None,
                        'df_function': df_function or None,
                        'df_kwargs_json': df_kwargs_json or None
                    })

                    # 准备初始配体（若已有则使用，否则构造少量种子）
                    if old_ligands:
                        initial_ligand_data = [
                            {"smiles": (lig.get('smiles') if isinstance(lig, dict) else (lig if isinstance(lig, str) else '')),
                             "binding_affinity": (lig.get('binding_affinity', -5.0) if isinstance(lig, dict) else -5.0)}
                            for lig in old_ligands
                        ]
                        initial_ligand_data = [x for x in initial_ligand_data if x["smiles"]]
                    else:
                        seed_smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O"]
                        initial_ligand_data = [{"smiles": s, "binding_affinity": -5.0} for s in seed_smiles]

                    # 若系统设置中启用了自定义GNN权重，则优先使用
                    phase1_model = None
                    try:
                        use_custom = bool(st.session_state.get('use_custom_gnn_ckpt', False))
                        custom_ckpt = st.session_state.get('gnn_checkpoint_path', None)
                        if use_custom and custom_ckpt and os.path.exists(custom_ckpt):
                            phase1_model = custom_ckpt
                            st.success(f"已使用自定义 GNN 检查点：{phase1_model}")
                    except Exception:
                        pass

                    # 否则优先使用已有最佳/最终权重；最后回退到默认路径
                    if phase1_model is None:
                        candidates = [
                            "logs/best_model.pth",
                            "logs/final_model.pth",
                            "deep_learning_results/phase1_equivariant_gnn.pth",
                        ]
                        for _p in candidates:
                            if os.path.exists(_p):
                                phase1_model = _p
                                break
                    if phase1_model is None:
                        # 训练一个初始模型以保证流程可用
                        st.info("未找到现有检查点，正在训练初始 GNN 模型以继续流程...")
                        phase1_model = pipeline.phase_1_equivariant_gnn(protein_file_dl, initial_ligand_data)
                        st.success(f"已训练并生成初始 GNN 模型：{phase1_model}")
                    else:
                        st.success(f"已加载 GNN 检查点：{phase1_model}")

                    # 第二阶段生成
                    phase2_path = pipeline.phase_2_transformer_generation(
                        phase1_model, protein_file_dl,
                        generation_targets=[{"target_affinity": float(target_affinity_dl), "num_molecules": int(num_ligands)}]
                    )

                    # 第三阶段优化
                    phase3_path = pipeline.phase_3_diffusion_and_rl(
                        phase2_path, protein_file_dl,
                        optimization_targets={
                            "binding_affinity": float(target_affinity_dl) - 1.0,
                            "top_k": int(max(10, top_k)),
                            "final_top_n": int(max(10, final_top_n))
                        }
                    )

                    # 读取最终结果（若第三阶段无结果则退回第二阶段）
                    df_phase3 = pd.read_csv(phase3_path) if os.path.exists(phase3_path) else pd.DataFrame()
                    if not df_phase3.empty:
                        use_df = df_phase3
                    else:
                        use_df = pd.read_csv(phase2_path)

                    # 统一转为配体结构（只保留关键字段）
                    keep_cols = [c for c in ["compound_id", "smiles", "binding_affinity", "molecular_weight", "logp"] if c in use_df.columns]
                    ligands = use_df[keep_cols].to_dict('records')
                else:
                    # 使用传统方法
                    ligands = self.ligand_generator.generate_optimized_ligands(num_ligands=num_ligands)

                if ligands:
                    st.session_state.ligands = ligands

                    if save_results:
                        # 保存到当前运行目录
                        ligands_dir = result_manager.get_ligands_dir()
                        ligands_file = ligands_dir / "generated_ligands.csv"

                        # 若为深度学习生成，复制阶段产物到当前run的ligands目录，便于对接/追溯
                        copied_files = []
                        try:
                            if method == "深度学习生成 (Transformer/Diffusion)":
                                from pathlib import Path as _P
                                if 'phase2_path' in locals() and phase2_path and os.path.exists(phase2_path):
                                    dst2 = result_manager.copy_file_to_current_run(_P(phase2_path), 'ligands', 'dl_phase2_generated_molecules.csv')
                                    copied_files.append(str(dst2))
                                if 'phase3_path' in locals() and phase3_path and os.path.exists(phase3_path):
                                    dst3 = result_manager.copy_file_to_current_run(_P(phase3_path), 'ligands', 'dl_phase3_optimized_molecules.csv')
                                    copied_files.append(str(dst3))
                        except Exception as _e:
                            st.warning(f"⚠️ 复制深度学习阶段CSV失败: {_e}")

                        # 保存配体数据
                        df = pd.DataFrame(ligands)
                        df.to_csv(ligands_file, index=False)

                        # 生成SMILES文件用于对接
                        smiles_file = ligands_dir / "ligands.smi"
                        with open(smiles_file, 'w') as f:
                            for ligand in ligands:
                                f.write(f"{ligand['smiles']}\t{ligand['smiles']}\n")

                        # 更新运行状态
                        step_files = copied_files + [str(ligands_file), str(smiles_file)]
                        result_manager.update_step_completed("配体生成", step_files)

                        st.success(f"✅ 成功生成 {len(ligands)} 个配体分子")
                        st.info(f"📊 实际生成数量: {len(ligands)} (请求数量: {num_ligands})")
                        st.info(f"💾 结果已保存到: {ligands_dir}")
                    else:
                        st.success(f"✅ 成功生成 {len(ligands)} 个配体分子")
                        st.info(f"📊 实际生成数量: {len(ligands)} (请求数量: {num_ligands})")
                else:
                    st.error("❌ 配体生成失败")
            except Exception as e:
                st.error(f"❌ 配体生成出错: {e}")

    def run_molecular_docking(self, receptor_file, exhaustiveness, num_poses):
        """运行分子对接"""
        with st.spinner("正在进行分子对接..."):
            try:
                ligands = st.session_state.ligands

                # 确保有当前运行目录
                if self.result_manager and not self.result_manager.get_current_run_dir():
                    # 创建新的运行目录
                    run_dir = self.result_manager.create_new_run_directory()
                    st.info(f"📁 创建新的运行目录: {run_dir.name}")
                elif not self.result_manager:
                    st.warning("⚠️ 结果管理器不可用，使用临时目录")

                # 提取SMILES列表
                smiles_list = []
                for ligand in ligands:
                    if isinstance(ligand, dict) and 'smiles' in ligand:
                        smiles_list.append(ligand['smiles'])
                    elif isinstance(ligand, str):
                        smiles_list.append(ligand)
                    else:
                        st.warning(f"跳过无效配体数据: {ligand}")

                if not smiles_list:
                    st.error("❌ 没有有效的SMILES数据进行对接")
                    return

                st.info(f"🧬 准备对接 {len(smiles_list)} 个分子")

                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔄 初始化分子对接...")
                progress_bar.progress(10)

                # 使用MolecularDocking进行对接
                status_text.text("🧬 正在进行分子对接...")
                progress_bar.progress(50)

                docking_results = self.molecular_docking.dock_multiple_ligands(
                    protein_pdb=receptor_file,
                    ligand_smiles=smiles_list
                )

                progress_bar.progress(90)

                if not docking_results.empty:
                    # 转换结果格式以匹配原有接口
                    formatted_results = []
                    for _, row in docking_results.iterrows():
                        result_data = {
                            'compound_id': f"ligand_{row['ligand_id']}",
                            'smiles': row['smiles'],
                            'binding_affinity': row['best_affinity'],
                            'rmsd_lb': row.get('rmsd_lb', 0.0),
                            'rmsd_ub': row.get('rmsd_ub', 0.0),
                            'num_poses': row.get('num_poses', 1),
                            'success': row.get('success', True)
                        }
                        formatted_results.append(result_data)

                    results_df = pd.DataFrame(formatted_results)
                    st.session_state.docking_results = results_df

                    # 完成进度
                    progress_bar.progress(100)
                    status_text.text("✅ 分子对接完成！")

                    st.success(f"✅ 成功对接 {len(results_df)} 个配体")

                    # 显示对接统计信息
                    successful_results = results_df[results_df['success'] == True]
                    if len(successful_results) > 0:
                        best_affinity = successful_results['binding_affinity'].min()
                        avg_affinity = successful_results['binding_affinity'].mean()
                        success_rate = len(successful_results) / len(results_df) * 100
                        st.info(f"📊 对接成功率: {success_rate:.1f}% | 最佳结合亲和力: {best_affinity:.2f} kcal/mol | 平均亲和力: {avg_affinity:.2f} kcal/mol")
                    else:
                        st.warning("⚠️ 所有分子对接都失败了")
                else:
                    st.error("❌ 分子对接失败")
            except Exception as e:
                st.error(f"❌ 分子对接出错: {e}")
                import traceback
                st.error(f"详细错误信息: {traceback.format_exc()}")

    def run_admet_analysis(self, analysis_types):
        """运行ADMET分析"""
        with st.spinner("正在进行ADMET分析..."):
            try:
                ligands = st.session_state.ligands
                admet_results = self.admet_analyzer.batch_admet_analysis(ligands)

                if not admet_results.empty:
                    st.session_state.admet_results = admet_results
                    st.success(f"✅ 成功分析 {len(admet_results)} 个分子")

                    # 保存到当前运行目录
                    try:
                        from scripts.result_manager import result_manager as _rm
                        crun = _rm.get_current_run_dir()
                        if not crun:
                            crun = _rm.create_new_run_directory()
                        admet_dir = _rm.get_admet_dir()
                        admet_dir.mkdir(exist_ok=True, parents=True)
                        out_file = admet_dir / "admet_results.csv"
                        admet_results.to_csv(out_file, index=False)
                        _rm.update_step_completed("ADMET分析", [str(out_file)])
                        st.info(f"💾 结果已保存到: {out_file}")
                    except Exception as _e:
                        st.warning(f"⚠️ 无法保存ADMET结果: {_e}")
                else:
                    st.error("❌ ADMET分析失败")
            except Exception as e:
                st.error(f"❌ ADMET分析出错: {e}")

    def generate_3d_visualization_report(self):
        """生成3D可视化报告"""
        with st.spinner("正在生成3D可视化报告..."):
            try:
                # 使用结果管理器获取当前运行目录
                current_run_dir = self.result_manager.get_current_run_dir()
                data_file = None
                data_source = None  # 'docking' 或 'ligands'

                if not current_run_dir:
                    # 手动查找运行目录：优先有对接结果，其次有配体CSV
                    results_dir = Path("results")
                    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

                    candidates = []
                    for run_dir in sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
                        docking_file = run_dir / "docking" / "docking_results.csv"
                        if docking_file.exists():
                            candidates.append((run_dir, docking_file, 'docking'))
                        lig_dir = run_dir / "ligands"
                        for name in ["generated_ligands.csv", "dl_phase3_optimized_molecules.csv", "dl_phase2_generated_molecules.csv"]:
                            f = lig_dir / name
                            if f.exists():
                                candidates.append((run_dir, f, 'ligands'))
                    if not candidates:
                        st.error("❌ 未找到可用于3D可视化的对接或配体结果，请先生成配体或运行分子对接")
                        return
                    # 选择最新一个
                    current_run_dir, data_file, data_source = max(
                        candidates, key=lambda x: x[1].stat().st_mtime
                    )
                    st.info(f"📁 选定运行目录: {current_run_dir.name}，数据源: {'对接结果' if data_source=='docking' else '配体结果'}")
                else:
                    # 优先使用当前run的对接结果，若没有则回退到配体CSV
                    docking_results_file = current_run_dir / "docking" / "docking_results.csv"
                    if docking_results_file.exists():
                        data_file = docking_results_file
                        data_source = 'docking'
                    else:
                        lig_dir = current_run_dir / "ligands"
                        candidates = [
                            lig_dir / "generated_ligands.csv",
                            lig_dir / "dl_phase3_optimized_molecules.csv",
                            lig_dir / "dl_phase2_generated_molecules.csv",
                        ]
                        for c in candidates:
                            if c.exists():
                                data_file = c
                                data_source = 'ligands'
                                break
                        if not data_file:
                            st.error("❌ 当前运行没有对接或配体结果，请先生成配体或运行分子对接")
                            return

                # 导入3D可视化模块
                from scripts.visualization_3d import Visualizer3D

                # 创建3D可视化器
                visualizer = Visualizer3D(output_dir=str(current_run_dir / "visualization_3d"))

                # 读取数据文件
                import pandas as pd
                df = pd.read_csv(data_file)
                if 'smiles' not in df.columns:
                    st.error(f"❌ 选定的数据文件缺少 'smiles' 列: {data_file}")
                    return
                if 'binding_affinity' not in df.columns:
                    df['binding_affinity'] = 0.0

                # 合并当前run的ADMET结果（按smiles）
                try:
                    admet_path = current_run_dir / "admet" / "admet_results.csv"
                    if admet_path.exists():
                        admet_df = pd.read_csv(admet_path)
                        if 'smiles' in admet_df.columns:
                            # 避免重复列名
                            cols_to_use = [c for c in admet_df.columns if c != 'compound_id']
                            df = df.merge(admet_df[cols_to_use], on='smiles', how='left')
                            st.info("🔗 已合并当前run的ADMET结果用于3D仪表盘展示")
                except Exception as _e:
                    st.warning(f"⚠️ 合并ADMET结果失败: {_e}")

                # 转换为所需格式
                results_data = []
                for _, row in df.iterrows():
                    results_data.append({
                        'compound_id': row.get('compound_id', ''),
                        'smiles': row.get('smiles', ''),
                        'binding_affinity': row.get('binding_affinity', 0),
                        'molecular_weight': row.get('molecular_weight', 0),
                        'logp': row.get('logp', 0),
                        'hbd': row.get('hbd', 0),
                        'hba': row.get('hba', 0),
                        'rotatable_bonds': row.get('rotatable_bonds', 0),
                        'tpsa': row.get('tpsa', 0)
                    })

                # 分析结合位点
                try:
                    from scripts.binding_site_analyzer import analyze_prrsv_binding_sites
                    binding_sites = analyze_prrsv_binding_sites("data/1p65.pdb")
                    st.info(f"🎯 识别到 {len(binding_sites)} 个结合位点")
                except Exception as e:
                    st.warning(f"结合位点分析失败: {e}")
                    binding_sites = None

                # 生成综合报告
                report_file = visualizer.generate_comprehensive_report(
                    results_data=results_data,
                    pdb_file="data/1p65.pdb",
                    binding_sites=binding_sites
                )

                if report_file:
                    st.success("✅ 3D可视化报告生成成功！")

                    # 检查生成的文件
                    viz_dir = current_run_dir / "visualization_3d"
                    main_report = Path(report_file)

                    if main_report.exists():
                        st.markdown(f"""
                        ### 🌐 3D可视化报告已生成

                        **主报告文件**: `{main_report}`

                        **包含的可视化**:
                        - 📊 交互式数据仪表板
                        - 🧬 蛋白质-配体复合物3D结构
                        - 🎯 结合位点分析
                        - 📈 分子性质分析
                        """)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if st.button("🌐 直接打开", use_container_width=True):
                                # 添加调试信息
                                st.info(f"🔍 尝试打开文件: {main_report}")
                                st.info(f"📁 文件存在: {main_report.exists()}")
                                if main_report.exists():
                                    st.info(f"📊 文件大小: {main_report.stat().st_size / 1024:.1f} KB")

                                result = self.open_file_in_browser(main_report, "3D可视化报告")
                                if not result:
                                    st.error("❌ 浏览器打开失败，请尝试以下方法:")
                                    st.markdown("### 🔧 手动打开方法:")
                                    st.markdown("1. **复制文件路径到文件管理器**")
                                    st.code(str(main_report.absolute()))
                                    st.markdown("2. **双击HTML文件直接打开**")
                                    st.markdown("3. **拖拽文件到浏览器窗口**")

                        with col2:
                            if st.button("📱 显示内容", use_container_width=True):
                                try:
                                    with open(main_report, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    st.components.v1.html(html_content, height=600, scrolling=True)
                                except Exception as e:
                                    st.error(f"无法显示内容: {e}")

                        with col3:
                            # 提供下载选项
                            try:
                                with open(main_report, 'rb') as f:
                                    st.download_button(
                                        label="💾 下载报告",
                                        data=f.read(),
                                        file_name=main_report.name,
                                        mime="text/html",
                                        use_container_width=True
                                    )
                            except Exception as e:
                                st.error(f"无法准备下载: {e}")

                        # 显示文件路径信息
                        st.info(f"📁 报告位置: {main_report.absolute()}")
                        st.markdown("💡 **使用提示**: 点击'直接打开'按钮，或复制上述路径到浏览器地址栏")
                else:
                    st.error("❌ 3D可视化报告生成失败")

            except Exception as e:
                st.error(f"❌ 生成3D可视化时出错: {e}")
                st.exception(e)

    def check_existing_3d_reports(self):
        """检查现有的3D可视化报告"""
        st.markdown("### 🔍 检查现有3D可视化报告")

        # 检查可能的报告位置
        possible_locations = [
            Path("visualization_output/comprehensive_3d_report.html"),
            Path("test_visualization_3d/comprehensive_3d_report.html")
        ]

        found_reports = []

        # 检查固定位置
        for location in possible_locations:
            if location.exists():
                found_reports.append(location)

        # 检查results目录下的运行目录
        results_dir = Path("results")
        if results_dir.exists():
            for run_dir in results_dir.glob("run_*"):
                viz_report = run_dir / "visualization_3d" / "comprehensive_3d_report.html"
                if viz_report.exists():
                    found_reports.append(viz_report)

        if found_reports:
            st.success(f"✅ 找到 {len(found_reports)} 个现有报告:")

            for i, report in enumerate(found_reports, 1):
                with st.expander(f"📊 报告 {i}: {report.name}"):
                    abs_path = report.absolute()
                    file_size = report.stat().st_size / 1024  # KB

                    st.info(f"📁 路径: {report}")
                    st.info(f"📊 大小: {file_size:.1f} KB")
                    st.info(f"🕒 修改时间: {datetime.fromtimestamp(report.stat().st_mtime)}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button(f"🌐 打开报告 {i}", key=f"open_report_{i}"):
                            self.open_file_in_browser(report, f"3D可视化报告 {i}")

                    with col2:
                        st.code(str(abs_path))

                    with col3:
                        try:
                            with open(report, 'rb') as f:
                                st.download_button(
                                    label=f"💾 下载 {i}",
                                    data=f.read(),
                                    file_name=f"3d_report_{i}.html",
                                    mime="text/html",
                                    key=f"download_report_{i}"
                                )
                        except Exception as e:
                            st.error(f"下载失败: {e}")
        else:
            st.warning("❌ 未找到现有的3D可视化报告")
            st.info("💡 请先运行完整工作流程或生成3D可视化报告")

    def show_3d_demo(self):
        """显示3D演示"""
        with st.spinner("正在生成3D演示..."):
            try:
                result = subprocess.run(
                    ["python", "demo_3d_simple.py"],
                    capture_output=True,
                    text=True,
                    cwd="."
                )

                if result.returncode == 0:
                    st.success("✅ 3D演示生成成功！")

                    demo_dir = Path("simple_3d_demo")
                    if demo_dir.exists():
                        index_file = demo_dir / "index.html"
                        if index_file.exists():
                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("🎬 查看演示", use_container_width=True):
                                    self.open_file_in_browser(index_file, "3D演示")

                            with col2:
                                st.text_input("演示文件路径",
                                            value=str(index_file.absolute()),
                                            help="复制此路径到浏览器地址栏打开",
                                            key="3d_demo_path")
                else:
                    st.error(f"❌ 3D演示生成失败: {result.stderr}")

            except Exception as e:
                st.error(f"❌ 生成3D演示时出错: {e}")

    def load_latest_results(self):
        """加载最新结果"""
        try:
            results_dir = Path("results")
            if not results_dir.exists():
                return

            # 查找最新的PRRSV结果
            prrsv_dirs = [d for d in results_dir.iterdir()
                         if d.is_dir() and d.name.startswith("prrsv_inhibitors_")]

            if prrsv_dirs:
                latest_dir = max(prrsv_dirs, key=lambda x: x.stat().st_mtime)

                # 加载对接结果
                docking_file = latest_dir / "docking_results.csv"
                if docking_file.exists():
                    st.session_state.docking_results = pd.read_csv(docking_file)

                # 加载ADMET结果
                admet_file = latest_dir / "admet_results.csv"
                if admet_file.exists():
                    st.session_state.admet_results = pd.read_csv(admet_file)

                # 加载配体数据
                ligands_file = latest_dir / "generated_ligands.csv"
                if ligands_file.exists():
                    ligands_df = pd.read_csv(ligands_file)
                    st.session_state.ligands = ligands_df.to_dict('records')

                st.info(f"已加载最新结果: {latest_dir.name}")

        except Exception as e:
            st.error(f"加载结果失败: {e}")

    def show_3d_visualization(self):
        """显示3D可视化页面"""
        st.title("🌐 3D可视化")

        st.markdown("""
        <div class="feature-card">
            <h4>🎯 功能说明</h4>
            <p>交互式3D可视化展示分子结构、蛋白质-配体复合物和数据分析结果。</p>
        </div>
        """, unsafe_allow_html=True)

        # 3D可视化选项
        viz_option = st.selectbox(
            "选择可视化类型",
            [
                "📊 生成完整3D报告",
                "🧪 单个分子3D结构",
                "🧬 蛋白质-配体复合物",
                "📈 交互式数据仪表板",
                "🎬 查看演示"
            ]
        )

        if viz_option == "📊 生成完整3D报告":
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("🚀 生成3D可视化报告", type="primary"):
                    self.generate_3d_visualization_report()

            with col2:
                if st.button("🎬 查看演示", type="secondary"):
                    self.show_3d_demo()

            with col3:
                if st.button("🔍 检查现有报告", type="secondary"):
                    self.check_existing_3d_reports()

        elif viz_option == "🧪 单个分子3D结构":
            self.show_single_molecule_3d()

        elif viz_option == "🧬 蛋白质-配体复合物":
            self.show_protein_ligand_complex()

        elif viz_option == "📈 交互式数据仪表板":
            self.show_interactive_dashboard()

        elif viz_option == "🎬 查看演示":
            self.show_3d_demo()

    def show_single_molecule_3d(self):
        """显示单个分子3D结构"""
        st.subheader("🧪 单个分子3D结构")

        if not VIEWER_3D_AVAILABLE:
            st.warning("⚠️ 3D查看器不可用，请安装: pip install py3dmol rdkit-pypi")
            return

        # 检查是否有分子数据
        if 'docking_results' in st.session_state:
            df = st.session_state.docking_results
            if 'smiles' in df.columns:
                # 分子选择器
                mol_options = [(row['compound_id'], row['smiles']) for _, row in df.head(20).iterrows()]
                if mol_options:
                    selected_idx = st.selectbox(
                        "选择分子",
                        range(len(mol_options)),
                        format_func=lambda x: f"{mol_options[x][0]} (结合能: {df.iloc[x]['binding_affinity']:.2f})"
                    )

                    if selected_idx is not None:
                        selected_row = df.iloc[selected_idx]
                        if selected_row['smiles']:
                            display_molecule_3d(
                                selected_row['smiles'],
                                selected_row['compound_id'],
                                width=800,
                                height=500
                            )
                        else:
                            st.warning("该分子没有SMILES信息")
                else:
                    st.info("没有可用的分子数据")
            else:
                st.warning("分子数据中缺少SMILES信息")
        else:
            st.info("请先运行分子对接以获得分子数据")

    def show_protein_ligand_complex(self):
        """显示蛋白质-配体复合物"""
        st.subheader("🧬 蛋白质-配体复合物")

        if not VIEWER_3D_AVAILABLE:
            st.warning("⚠️ 3D查看器不可用，请安装: pip install py3dmol rdkit-pypi")
            return

        # 检查PDB文件
        pdb_files = ["data/AF-Q9GLP0-F1-model_v4.pdb", "data/1p65.pdb"]
        available_pdb = [f for f in pdb_files if os.path.exists(f)]

        if available_pdb and 'docking_results' in st.session_state:
            selected_pdb = st.selectbox("选择蛋白质结构", available_pdb)

            df = st.session_state.docking_results
            if len(df) > 0 and 'smiles' in df.columns:
                best_row = df.iloc[0]  # 最佳分子

                if best_row['smiles']:
                    st.info(f"显示最佳分子 {best_row['compound_id']} 与蛋白质的复合物")
                    display_protein_ligand_complex(
                        selected_pdb,
                        best_row['smiles'],
                        width=1000,
                        height=600
                    )
                else:
                    st.warning("最佳分子没有SMILES信息")
            else:
                st.warning("没有可用的分子数据")
        else:
            st.warning("没有可用的PDB文件或分子数据")

    def show_interactive_dashboard(self):
        """显示交互式数据仪表板"""
        st.subheader("📈 交互式数据仪表板")

        # 检查是否有数据
        if 'docking_results' not in st.session_state:
            st.info("请先运行分子对接以获得数据")
            return

        df = st.session_state.docking_results

        # 创建交互式图表
        col1, col2 = st.columns(2)

        with col1:
            # 结合亲和力分布
            if 'binding_affinity' in df.columns:
                fig1 = px.histogram(df, x='binding_affinity',
                                   title='结合亲和力分布', nbins=20)
                st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # 分子性质散点图
            if 'molecular_weight' in df.columns and 'logp' in df.columns:
                fig2 = px.scatter(df, x='molecular_weight', y='logp',
                                 title='分子量 vs LogP',
                                 color='binding_affinity' if 'binding_affinity' in df.columns else None)
                st.plotly_chart(fig2, use_container_width=True)

        # 前10名化合物
        st.subheader("🏆 前10名化合物")
        top_10 = df.head(10)

        # 创建条形图
        if 'binding_affinity' in df.columns:
            fig3 = px.bar(top_10, x='compound_id', y='binding_affinity',
                         title='前10名化合物结合亲和力')
            st.plotly_chart(fig3, use_container_width=True)

    def show_results_analysis(self):
        """显示结果分析页面"""
        st.title("📊 结果分析")

        # 检查是否有完整结果
        has_docking = 'docking_results' in st.session_state
        has_admet = 'admet_results' in st.session_state

        if not has_docking:
            # 尝试加载最新结果
            self.load_latest_results()
            has_docking = 'docking_results' in st.session_state
            has_admet = 'admet_results' in st.session_state

        if not all([has_docking, has_admet]):
            st.warning("⚠️ 请先完成对接和ADMET分析，或确保结果文件存在")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 尝试加载最新结果"):
                    self.load_latest_results()
                    st.rerun()

            with col2:
                if st.button("🚀 运行完整工作流程"):
                    st.session_state.page = "🚀 完整工作流程"
                    st.rerun()
            return

        # 整合结果分析
        self.perform_integrated_analysis()

    def perform_integrated_analysis(self):
        """执行综合分析"""
        docking_results = st.session_state.docking_results
        admet_results = st.session_state.admet_results

        st.success("✅ 找到完整的分析结果")

        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总分子数", len(docking_results))

        with col2:
            if 'binding_affinity' in docking_results.columns:
                best_binding = docking_results['binding_affinity'].min()
                st.metric("最佳结合能", f"{best_binding:.2f} kcal/mol")

        with col3:
            if 'lipinski_compliant' in admet_results.columns:
                compliance_rate = (admet_results['lipinski_compliant'].sum() / len(admet_results)) * 100
                st.metric("Lipinski符合率", f"{compliance_rate:.1f}%")

        with col4:
            avg_binding = docking_results['binding_affinity'].mean() if 'binding_affinity' in docking_results.columns else 0
            st.metric("平均结合能", f"{avg_binding:.2f} kcal/mol")

        # 前10个最佳化合物
        st.subheader("🏆 前10个最佳候选化合物")
        top_10 = docking_results.head(10)
        st.dataframe(top_10, use_container_width=True)

    def show_complete_workflow(self):
        """显示完整工作流程页面"""
        st.title("🚀 完整工作流程")

        st.markdown("""
        <div class="feature-card">
            <h4>🎯 一键式完整流程</h4>
            <p>自动执行配体生成 → 分子对接 → ADMET分析 → 结果整合的完整工作流程。</p>
        </div>
        """, unsafe_allow_html=True)

        # 工作流程参数
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("⚙️ 流程参数")

            num_ligands = st.slider("生成分子数量", 10, 200, 100)
            receptor_file = st.selectbox(
                "受体文件",
                [
                    "data/1p65.pdbqt",  # PRRSV核衣壳蛋白（主要靶点）
                    "data/AF-F1SR53-F1-model_v4.pdbqt",  # 整合素复合物2
                    "data/AF-Q9GLP0-F1-model_v4_rigid.pdbqt",  # 整合素复合物1
                    "data/new1p65.pdbqt"  # 旧病毒蛋白文件（备用）
                ],
                format_func=lambda x: {
                    "data/1p65.pdbqt": "🦠 PRRSV核衣壳蛋白 (1p65.pdbqt) ✅ 推荐",
                    "data/AF-F1SR53-F1-model_v4.pdbqt": "🧬 整合素复合物2 (AF-F1SR53) ✅",
                    "data/AF-Q9GLP0-F1-model_v4_rigid.pdbqt": "🧬 整合素复合物1 (AF-Q9GLP0) ✅",
                    "data/new1p65.pdbqt": "🦠 旧病毒蛋白 (new1p65.pdbqt) ⚠️ 备用"
                }.get(x, x)
            )

            include_3d_viz = st.checkbox("包含3D可视化", True)
            save_results = st.checkbox("保存详细结果", True)

            if st.button("🚀 启动完整工作流程", type="primary", disabled=st.session_state.workflow_running):
                self.run_complete_workflow(num_ligands, receptor_file, include_3d_viz, save_results)

        with col2:
            st.subheader("📊 流程步骤")

            steps = [
                "1. 🧪 配体生成 - 生成候选小分子化合物",
                "2. 🔬 分子对接 - 预测结合模式和亲和力",
                "3. 📈 ADMET分析 - 评估药物类似性质",
                "4. 📊 结果整合 - 综合评分和排序",
                "5. 🌐 3D可视化 - 生成交互式3D展示"
            ]

            for step in steps:
                st.write(step)

            if st.session_state.workflow_running:
                st.info("⏳ 工作流程正在运行中...")

    def run_complete_workflow(self, num_ligands, receptor_file, include_3d_viz, save_results):
        """运行完整工作流程"""
        st.session_state.workflow_running = True

        try:
            # 步骤1: 配体生成
            with st.spinner("🧪 正在生成配体..."):
                ligands = self.ligand_generator.generate_optimized_ligands(num_ligands=num_ligands)
                if ligands:
                    st.session_state.ligands = ligands
                    st.success(f"✅ 成功生成 {len(ligands)} 个配体")
                else:
                    st.error("❌ 配体生成失败")
                    return

            # 步骤2: 分子对接
            with st.spinner("🔬 正在进行分子对接..."):
                # 提取SMILES列表
                smiles_list = []
                for ligand in ligands:
                    if isinstance(ligand, dict) and 'smiles' in ligand:
                        smiles_list.append(ligand['smiles'])
                    elif isinstance(ligand, str):
                        smiles_list.append(ligand)

                if smiles_list:
                    docking_results = self.molecular_docking.dock_multiple_ligands(
                        protein_pdb=receptor_file,
                        ligand_smiles=smiles_list
                    )

                    if not docking_results.empty:
                        # 转换结果格式
                        formatted_results = []
                        for _, row in docking_results.iterrows():
                            result_data = {
                                'compound_id': f"ligand_{row['ligand_id']}",
                                'smiles': row['smiles'],
                                'binding_affinity': row['best_affinity'],
                                'success': row.get('success', True)
                            }
                            formatted_results.append(result_data)

                        results_df = pd.DataFrame(formatted_results)
                        st.session_state.docking_results = results_df
                        st.success(f"✅ 成功对接 {len(results_df)} 个配体")
                    else:
                        st.error("❌ 分子对接失败")
                        return
                else:
                    st.error("❌ 没有有效的SMILES数据进行对接")
                    return

            # 步骤3: ADMET分析
            with st.spinner("📈 正在进行ADMET分析..."):
                admet_results = self.admet_analyzer.batch_admet_analysis(ligands)
                if not admet_results.empty:
                    st.session_state.admet_results = admet_results
                    st.success(f"✅ 成功分析 {len(admet_results)} 个分子")
                else:
                    st.error("❌ ADMET分析失败")
                    return

            # 步骤4: 3D可视化
            if include_3d_viz:
                with st.spinner("🌐 正在生成3D可视化..."):
                    self.generate_3d_visualization_report()

            st.success("🎉 完整工作流程执行成功！")

        except Exception as e:
            st.error(f"❌ 工作流程执行失败: {e}")
        finally:
            st.session_state.workflow_running = False

    def show_file_management(self):
        """显示文件管理页面"""
        st.title("📁 文件管理")
        st.info("文件管理功能正在开发中...")



    def show_2d_molecule_images(self):
        """显示2D分子图像页面 - 动态生成基于最新结果"""
        st.title("🖼️ 2D分子结构图像")

        st.markdown("""
        <div class="feature-card">
            <h4>🎯 功能说明</h4>
            <p>基于最新的分子对接结果，动态生成高质量的2D分子结构图像。包括单个分子图像和批量网格图。</p>
            <p><strong>特点:</strong> 每次生成都基于当前最新的对接结果，确保图像与数据同步。</p>
        </div>
        """, unsafe_allow_html=True)

        # 检查是否有结果数据 - 支持多种结果目录格式
        possible_results_dirs = [Path("results"), Path("../results")]
        results_dir = None

        for dir_path in possible_results_dirs:
            if dir_path.exists():
                results_dir = dir_path
                break

        if not results_dir:
            st.warning("⚠️ 未找到结果目录，请先运行分子对接分析")
            return

        # 可选：选择运行目录（run_*），便于指定生成到某个run的可视化目录
        run_dirs_all = sorted([d for d in results_dir.glob("run_*") if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        preferred_run_dir = None
        if run_dirs_all:
            options = ["自动选择(当前运行/最新)"] + [d.name for d in run_dirs_all]
            choice = st.selectbox("选择运行目录（可选）", options, index=0,
                                  help="默认自动选择当前运行目录或最新有数据的运行；也可以手动指定某个run目录")
            if choice != "自动选择(当前运行/最新)":
                preferred_run_dir = results_dir / choice

        # 选择用于2D绘制的数据文件：优先对接结果，其次配体CSV；优先使用当前运行目录
        selected_file = None
        selected_run_dir = None
        selected_source = None

        # 1) 优先使用当前运行目录
        try:
            if preferred_run_dir is not None:
                crun = preferred_run_dir
            elif self.result_manager and self.result_manager.current_run_dir:
                crun = self.result_manager.current_run_dir
            # 对接优先
            docking_file = crun / "docking" / "docking_results.csv"
            if docking_file.exists():
                selected_file = docking_file
                selected_run_dir = crun
                selected_source = "docking"
            else:
                lig_dir = crun / "ligands"
                lig_candidates = [
                    lig_dir / "generated_ligands.csv",
                    lig_dir / "dl_phase3_optimized_molecules.csv",
                    lig_dir / "dl_phase2_generated_molecules.csv",
                ]
                for c in lig_candidates:
                    if c.exists():
                        selected_file = c
                        selected_run_dir = crun
                        selected_source = "ligands"
                        break
        except Exception:
            pass

        # 2) 若当前运行未找到，则全局扫描最新的 run_* 或旧格式
        if not selected_file:
            candidates = []
            # 新格式: run_* 目录
            for run_dir in results_dir.glob("run_*"):
                docking_file = run_dir / "docking" / "docking_results.csv"
                if docking_file.exists():
                    candidates.append((docking_file, docking_file.stat().st_mtime, run_dir, "docking"))
                lig_dir = run_dir / "ligands"
                for name in ["generated_ligands.csv", "dl_phase3_optimized_molecules.csv", "dl_phase2_generated_molecules.csv"]:
                    f = lig_dir / name
                    if f.exists():
                        candidates.append((f, f.stat().st_mtime, run_dir, "ligands"))
            # 旧格式: prrsv_inhibitors_*
            for result_folder in results_dir.glob("prrsv_inhibitors_*"):
                csv_file = result_folder / "ranked_inhibitors.csv"
                if csv_file.exists():
                    candidates.append((csv_file, csv_file.stat().st_mtime, result_folder, "docking"))

            if candidates:
                best = max(candidates, key=lambda x: x[1])
                selected_file, _, selected_run_dir, selected_source = best

        if not selected_file:
            st.warning("⚠️ 未找到可用于2D绘制的对接或配体结果，请先生成配体或进行分子对接分析")
            return

        # 显示将使用的文件信息
        result_time = pd.Timestamp.fromtimestamp(selected_file.stat().st_mtime)
        src_label = "对接结果" if selected_source == "docking" else "配体结果"
        st.info(f"📁 将基于{src_label}生成2D图像: {selected_run_dir.name} (数据时间: {result_time.strftime('%Y-%m-%d %H:%M:%S')})")

        # 参数设置
        st.subheader("⚙️ 生成参数")
        col1, col2 = st.columns(2)

        with col1:
            top_n = st.slider("显示分子数量", min_value=5, max_value=1000, value=50, step=10,
                            help="选择要生成2D图像的分子数量（按结合亲和力排序）")

        with col2:
            image_quality = st.selectbox("图像质量",
                                       options=["标准", "高质量", "超高质量"],
                                       index=1,
                                       help="选择生成图像的质量级别")

        # 网格设置
        st.subheader("🔬 网格图设置")
        col3, col4 = st.columns(2)

        with col3:
            grid_cols = st.slider("网格列数", min_value=3, max_value=6, value=5)

        with col4:
            grid_rows = st.slider("网格行数", min_value=3, max_value=6, value=4)

        # 生成按钮
        if st.button("🎨 动态生成2D分子图像", type="primary", use_container_width=True):
            with st.spinner("正在基于最新结果生成2D分子结构图像..."):
                try:
                    # 设置结果管理器的当前运行目录为选定目录
                    if self.result_manager and selected_run_dir:
                        self.result_manager.current_run_dir = selected_run_dir
                        st.info(f"📁 设置当前运行目录为: {selected_run_dir.name}")

                    # 导入2D生成器
                    from scripts.molecule_2d_generator import Molecule2DGenerator

                    # 创建生成器（现在会使用正确的运行目录）
                    generator = Molecule2DGenerator()

                    # 动态处理最新结果
                    results = generator.process_latest_results(top_n=top_n)

                    if results:
                        st.success("✅ 2D分子图像动态生成完成！")

                        # 显示结果统计
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("处理分子数", results['molecules_count'])
                        with col2:
                            st.metric("单个图像", len(results['individual_images']))
                        with col3:
                            st.metric("网格图", 1 if results['grid_image'] else 0)
                        with col4:
                            st.metric("HTML报告", 1 if results.get('report_path') else 0)

                        # 显示网格图
                        if results['grid_image'] and os.path.exists(results['grid_image']):
                            st.subheader("🔬 分子结构网格图")
                            st.image(results['grid_image'],
                                   caption=f"Top {results['molecules_count']} 分子2D结构网格图 (基于最新结果)",
                                   use_column_width=True)

                        # 显示单个分子图像
                        st.subheader("🧪 单个分子结构 (前9个)")

                        # 创建3x3网格显示
                        for row in range(3):
                            cols = st.columns(3)
                            for col in range(3):
                                idx = row * 3 + col
                                if idx < len(results['individual_images']) and idx < 9:
                                    img_path = results['individual_images'][idx]
                                    if os.path.exists(img_path):
                                        with cols[col]:
                                            st.image(img_path,
                                                   caption=f"分子 #{idx+1}",
                                                   use_column_width=True)

                        # 提供完整报告
                        if results.get('report_path') and os.path.exists(results['report_path']):
                            st.markdown("---")
                            st.subheader("📋 完整报告")

                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("🌐 在浏览器中打开完整报告", use_container_width=True):
                                    self.open_file_in_browser(results['report_path'], "2D分子图像报告")

                            with col2:
                                with open(results['report_path'], 'r', encoding='utf-8') as f:
                                    st.download_button(
                                        label="📥 下载HTML报告",
                                        data=f.read(),
                                        file_name=f"molecule_2d_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                                        mime="text/html",
                                        use_container_width=True
                                    )

                        # 下载选项
                        st.markdown("---")
                        st.subheader("💾 下载选项")

                        col1, col2 = st.columns(2)

                        with col1:
                            if results['grid_image'] and os.path.exists(results['grid_image']):
                                with open(results['grid_image'], 'rb') as f:
                                    st.download_button(
                                        label="📥 下载网格图",
                                        data=f.read(),
                                        file_name=f"molecules_grid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )

                        with col2:
                            # 创建所有单个图像的ZIP包
                            if st.button("📦 下载所有单个图像 (ZIP)", use_container_width=True):
                                import zipfile
                                import io

                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                    for i, img_path in enumerate(results['individual_images']):
                                        if os.path.exists(img_path):
                                            zip_file.write(img_path, f"molecule_{i+1}_{os.path.basename(img_path)}")

                                st.download_button(
                                    label="📥 下载ZIP文件",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"molecules_2d_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip"
                                )

                    else:
                        st.error("❌ 2D分子图像生成失败，请检查是否有有效的对接结果")

                except Exception as e:
                    st.error(f"❌ 生成2D分子图像时出错: {e}")
                    st.exception(e)

        # 显示提示信息
        st.markdown("---")
        st.info("""
        💡 **使用提示:**
        - 每次点击生成按钮都会基于最新的对接结果重新生成2D图像
        - 图像质量越高，生成时间越长
        - 建议先运行分子对接获得结果，再使用此功能
        - 生成的图像包含分子ID、结合亲和力、分子量等关键信息
        """)

    def show_deep_learning_pipeline(self):
        """显示深度学习流水线页面"""
        st.title("🤖 深度学习流水线")

        if not HAS_DEEP_LEARNING:
            st.error("❌ 深度学习模块未安装，请先安装相关依赖")
            st.code("pip install torch torch-geometric e3nn rdkit-pypi")
            return

        st.info(f"🖥️ 当前计算设备: {DEVICE_INFO}")

        # 流水线概述
        st.markdown("""
        ### 🎯 深度学习流水线概述

        本流水线基于最新的深度学习技术，包含三个主要阶段：

        **第一阶段（30天）**: SE(3)-Equivariant GNN评分器
        - 训练空间等变图神经网络
        - 学习蛋白质-配体相互作用
        - 预测结合亲和力

        **第二阶段（60天）**: Pocket-conditioned Transformer生成
        - 基于蛋白口袋的分子生成
        - 交叉注意力机制
        - GNN筛选和MD精修

        **第三阶段（90天）**: Diffusion模型和强化学习优化
        - 扩散模型生成高质量分子
        - 多任务判别器
        - 主动学习闭环
        """)

        # 流水线配置
        st.markdown("### ⚙️ 流水线配置")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧬 输入数据")
            protein_file = st.file_uploader("上传蛋白质PDB文件", type=['pdb'])

            # 初始配体数据
            st.subheader("💊 初始配体数据")
            # 预设默认变量以避免未定义（UnboundLocalError）
            ligand_smiles = ""
            ligand_affinities = ""
            use_existing_ligands = st.checkbox("使用现有配体数据", value=True)

            if not use_existing_ligands:
                ligand_smiles = st.text_area(
                    "输入SMILES（每行一个）",
                    placeholder="CCO\nCCN\nCCC"
                )
                ligand_affinities = st.text_area(
                    "对应的结合亲和力（每行一个）",
                    placeholder="-5.2\n-4.8\n-4.5"
                )

        with col2:
            st.subheader("🎯 生成目标")
            target_affinity = st.slider("目标结合亲和力 (kcal/mol)", -10.0, -3.0, -6.0, 0.1)
            num_molecules = st.slider("生成分子数量", 10, 1000, 100, 10)

            st.subheader("🏋️ 训练参数")
            num_epochs = st.slider("训练轮数", 10, 500, 100, 10)
            batch_size = st.selectbox("批次大小", [16, 32, 64, 128], index=1)
            learning_rate = st.selectbox("学习率", [1e-5, 1e-4, 1e-3], index=1)

        # 阶段选择
        st.markdown("### 🚀 执行阶段")

        phase_options = {
            "第一阶段：GNN评分器": 1,
            "第二阶段：Transformer生成": 2,
            "第三阶段：Diffusion优化": 3,
            "完整流水线": 0
        }

        selected_phase = st.selectbox("选择执行阶段", list(phase_options.keys()))
        phase_num = phase_options[selected_phase]

        # 执行按钮
        if st.button("🚀 开始执行深度学习流水线", type="primary"):
            if protein_file is None:
                st.error("请先上传蛋白质PDB文件")
                return

            # 保存上传的文件
            protein_path = f"temp_protein_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdb"
            with open(protein_path, "wb") as f:
                f.write(protein_file.getbuffer())

            try:
                # 创建深度学习流水线
                pipeline = DeepLearningPipeline()

                # 准备数据
                if use_existing_ligands:
                    if 'ligands' in st.session_state and st.session_state.ligands:
                        initial_ligand_data = [
                            {"smiles": lig['smiles'], "binding_affinity": lig.get('binding_affinity', -5.0)}
                            for lig in st.session_state.ligands
                        ]
                    else:
                        st.error("❌ 未找到现有配体数据，请先在“🧪 配体生成”中生成，或取消“使用现有配体数据”并输入SMILES。")
                        return
                else:
                    # 解析用户输入的配体数据
                    smiles_list = ligand_smiles.strip().split('\n') if ligand_smiles else []
                    affinity_list = [float(x) for x in ligand_affinities.strip().split('\n')] if ligand_affinities else []

                    if not smiles_list:
                        st.error("❌ 未提供任何SMILES，请输入至少一条。")
                        return

                    initial_ligand_data = []
                    for i, smiles in enumerate(smiles_list):
                        affinity = affinity_list[i] if i < len(affinity_list) else -5.0
                        initial_ligand_data.append({"smiles": smiles, "binding_affinity": affinity})

                generation_targets = [{"target_affinity": target_affinity, "num_molecules": num_molecules}]
                optimization_targets = {
                    "binding_affinity": target_affinity,
                    "drug_likeness": 0.8,
                    "synthesizability": 0.7
                }

                # 执行流水线
                with st.spinner(f"正在执行{selected_phase}..."):
                    if phase_num == 1:
                        result = pipeline.phase_1_equivariant_gnn(protein_path, initial_ligand_data)
                        st.success(f"✅ 第一阶段完成！模型保存至: {result}")
                    elif phase_num == 2:
                        phase1_model = "deep_learning_results/phase1_equivariant_gnn.pth"
                        if not os.path.exists(phase1_model):
                            st.error("❌ 未找到第一阶段模型，请先执行第一阶段")
                            return
                        result = pipeline.phase_2_transformer_generation(phase1_model, protein_path, generation_targets)
                        st.success(f"✅ 第二阶段完成！生成分子保存至: {result}")
                    elif phase_num == 3:
                        phase2_data = "deep_learning_results/phase2_generated_molecules.csv"
                        if not os.path.exists(phase2_data):
                            st.error("❌ 未找到第二阶段数据，请先执行第二阶段")
                            return
                        result = pipeline.phase_3_diffusion_and_rl(phase2_data, protein_path, optimization_targets)
                        st.success(f"✅ 第三阶段完成！优化分子保存至: {result}")
                    else:
                        # 完整流水线
                        results = pipeline.run_full_pipeline(
                            protein_path, initial_ligand_data, generation_targets, optimization_targets
                        )
                        st.success("✅ 完整流水线执行成功！")
                        st.json(results)

            except Exception as e:
                st.error(f"❌ 执行失败: {e}")
                import traceback
                st.error(traceback.format_exc())

            finally:
                # 清理临时文件
                if os.path.exists(protein_path):
                    os.remove(protein_path)

    def show_ai_model_training(self):
        """显示AI模型训练页面"""
        st.title("🧠 AI模型训练")

        if not HAS_DEEP_LEARNING:
            st.error("❌ 深度学习模块未安装")
            return

        st.info("🚧 AI模型训练功能正在开发中...")

        # 模型选择
        st.markdown("### 🤖 模型选择")
        model_type = st.selectbox(
            "选择模型类型",
            [
                "SE(3)-Equivariant GNN",
                "Pocket-Ligand Transformer",
                "Multi-task Discriminator"
            ]
        )

        # 训练数据
        st.markdown("### 📊 训练数据")
        data_source = st.selectbox(
            "数据源",
            ["PDBBind数据集", "ChEMBL数据集", "自定义数据", "现有实验数据"]
        )

        # 训练配置
        st.markdown("### ⚙️ 训练配置")
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.slider("训练轮数", 10, 1000, 100)
            batch_size = st.selectbox("批次大小", [8, 16, 32, 64], index=2)
            learning_rate = st.selectbox("学习率", [1e-5, 1e-4, 1e-3], index=1)

        with col2:
            use_gpu = st.checkbox("使用GPU", value=torch.cuda.is_available())
            mixed_precision = st.checkbox("混合精度训练", value=True)
            early_stopping = st.checkbox("早停", value=True)

        # 开始训练按钮
        if st.button("🏋️ 开始训练", type="primary"):
            st.info("训练功能正在开发中...")

    def show_system_settings(self):
        """显示系统设置页面"""
        st.title("⚙️ 系统设置")

        st.subheader("模型权重设置（GNN评分器）")
        # 读取当前状态
        use_custom_default = bool(st.session_state.get('use_custom_gnn_ckpt', False))
        current_path_default = st.session_state.get('gnn_checkpoint_path', '')

        use_custom = st.checkbox("使用自定义GNN权重（优先级最高）", value=use_custom_default, help="启用后，深度学习生成流程将优先加载你指定的检查点文件")
        st.session_state['use_custom_gnn_ckpt'] = use_custom

        # 选择或上传检查点
        col_a, col_b = st.columns([2, 1])
        with col_a:
            ckpt_path = st.text_input("检查点路径（.pth）", value=current_path_default or "logs/best_model.pth", placeholder="logs/best_model.pth")
        with col_b:
            uploaded = st.file_uploader("上传检查点文件", type=["pth"], help="上传后将保存到 logs/uploads/ 目录")
        if uploaded is not None:
            try:
                from pathlib import Path as _P
                up_dir = _P("logs/uploads")
                up_dir.mkdir(parents=True, exist_ok=True)
                save_path = up_dir / uploaded.name
                with open(save_path, 'wb') as f:
                    f.write(uploaded.getbuffer())
                ckpt_path = str(save_path)
                st.success(f"已保存上传文件：{ckpt_path}")
            except Exception as _e:
                st.error(f"保存上传文件失败：{_e}")

        # 保存到 session_state
        st.session_state['gnn_checkpoint_path'] = ckpt_path

        # 验证权重（若安装了 PyTorch 则做加载测试）
        if st.button("验证权重可加载性"):
            if ckpt_path and os.path.exists(ckpt_path):
                try:
                    if HAS_DEEP_LEARNING:
                        import torch as _torch
                        try:
                            _ = _torch.load(ckpt_path, map_location='cpu', weights_only=False)
                        except TypeError:
                            _ = _torch.load(ckpt_path, map_location='cpu')
                        st.success("权重文件可读取 ✅")
                    else:
                        st.info("已找到文件，但当前环境未安装PyTorch，跳过加载测试。")
                except Exception as e:
                    st.error(f"权重读取失败：{e}")
            else:
                st.error("文件不存在，请检查路径或先上传。")

        st.markdown("---")
        st.subheader("默认权重优先级（当未启用自定义时）")
        st.code("""
1) logs/best_model.pth
2) logs/final_model.pth
3) deep_learning_results/phase1_equivariant_gnn.pth
未找到时，将自动训练一个初始模型以保障流程可用。
""", language="text")

    def main_page(self):
        """主页面"""
        self.render_header()

        # 渲染侧边栏并获取选择的页面
        page = self.render_sidebar()

        # 根据选择显示对应页面
        if page == "🏠 项目概览":
            self.show_project_overview()
        elif page == "🧪 配体生成":
            self.show_ligand_generation()
        elif page == "🔬 分子对接":
            self.show_molecular_docking()
        elif page == "📈 ADMET分析":
            self.show_admet_analysis()
        elif page == "📊 结果分析":
            self.show_results_analysis()
        elif page == "🖼️ 2D分子图像":
            self.show_2d_molecule_images()
        elif page == "🌐 3D可视化":
            self.show_3d_visualization()
        elif page == "🚀 完整工作流程":
            self.show_complete_workflow()
        elif page == "🤖 深度学习流水线":
            self.show_deep_learning_pipeline()
        elif page == "🧠 AI模型训练":
            self.show_ai_model_training()
        elif page == "📁 文件管理":
            self.show_file_management()
        elif page == "⚙️ 系统设置":
            self.show_system_settings()

def main():
    """主函数"""
    # 初始化统一界面
    interface = UnifiedPRRSVInterface()

    # 运行主页面
    interface.main_page()

if __name__ == "__main__":
    main()
