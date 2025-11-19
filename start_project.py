#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRRSV抑制剂设计平台启动脚本
提供多种启动方式的统一入口
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def safe_print(message):
    """安全的打印函数"""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        print(safe_message)

def check_dependencies():
    """检查依赖"""
    safe_print("检查项目依赖...")

    # 必需依赖
    required_modules = [
        ("streamlit", "Web界面框架"),
        ("pandas", "数据处理"),
        ("plotly", "交互式图表"),
        ("rdkit", "分子处理"),
        ("py3dmol", "3D可视化")
    ]

    # 可选依赖
    optional_modules = [
        ("meeko", "分子格式转换")
    ]

    missing_modules = []

    # 检查必需依赖
    for module, description in required_modules:
        try:
            if module == "py3dmol":
                import py3Dmol
            else:
                __import__(module.replace("-", "_"))
            safe_print(f"✅ {module} - {description}")
        except ImportError:
            safe_print(f"❌ {module} - {description}")
            missing_modules.append(module)

    # 检查可选依赖
    for module, description in optional_modules:
        try:
            __import__(module.replace("-", "_"))
            safe_print(f"✅ {module} - {description}")
        except ImportError:
            safe_print(f"⚠️ {module} - {description} (可选，功能受限)")

    if missing_modules:
        safe_print(f"\n⚠️ 缺少必需依赖: {missing_modules}")
        safe_print("请运行: pip install -r requirements.txt")
        return False

    safe_print("✅ 所有必需依赖检查通过")
    return True

def show_menu():
    """显示启动菜单"""
    safe_print("""
🧬 PRRSV抑制剂设计平台
========================

请选择启动方式:

1. 🌐 统一Web界面 (推荐)
2. 🚀 完整工作流程 (命令行)
3. 🧪 小分子3D查看器
4. 🔬 AutoDock结果查看器
5. 📊 生成3D可视化报告
6. 🎬 查看所有3D结果
7. 🔧 系统诊断
8. 📖 查看帮助文档
9. ❌ 退出

""")

def start_web_interface():
    """启动Web界面"""
    safe_print("🌐 启动统一Web界面...")
    
    try:
        # 检查文件是否存在
        web_file = "unified_web_interface.py"
        if not os.path.exists(web_file):
            safe_print(f"❌ 找不到文件: {web_file}")
            return False
        
        # 启动Streamlit，使用动态端口
        import socket
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port

        port = find_free_port()
        cmd = [sys.executable, "-m", "streamlit", "run", web_file, f"--server.port={port}"]
        
        safe_print("📡 启动Web服务器...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 等待服务器启动
        safe_print("⏳ 等待服务器启动...")
        time.sleep(3)
        
        # 检查进程是否还在运行
        if process.poll() is None:
            safe_print("✅ Web服务器启动成功！")
            safe_print(f"🌐 访问地址: http://localhost:{port}")

            # 尝试自动打开浏览器
            try:
                webbrowser.open(f"http://localhost:{port}")
                safe_print("🌐 已在浏览器中打开")
            except Exception as e:
                safe_print(f"⚠️ 无法自动打开浏览器: {e}")
                safe_print(f"请手动在浏览器中访问: http://localhost:{port}")
            
            return True
        else:
            stdout, stderr = process.communicate()
            safe_print(f"❌ 启动失败:")
            safe_print(f"stdout: {stdout}")
            safe_print(f"stderr: {stderr}")
            return False
            
    except Exception as e:
        safe_print(f"❌ 启动Web界面时出错: {e}")
        return False

def run_complete_workflow():
    """运行完整工作流程"""
    safe_print("🚀 运行完整工作流程...")
    
    try:
        result = subprocess.run([sys.executable, "run_full_workflow.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            safe_print("✅ 完整工作流程执行成功！")
            safe_print(result.stdout)
        else:
            safe_print("❌ 工作流程执行失败:")
            safe_print(result.stderr)
            
    except Exception as e:
        safe_print(f"❌ 执行工作流程时出错: {e}")

def run_molecule_3d_viewer():
    """运行小分子3D查看器"""
    safe_print("🧪 启动小分子3D查看器...")
    
    try:
        result = subprocess.run([sys.executable, "molecule_3d_viewer.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            safe_print("✅ 小分子3D查看器生成成功！")
            safe_print(result.stdout)
        else:
            safe_print("❌ 小分子3D查看器生成失败:")
            safe_print(result.stderr)
            
    except Exception as e:
        safe_print(f"❌ 运行小分子3D查看器时出错: {e}")

def run_autodock_viewer():
    """运行AutoDock结果查看器"""
    safe_print("🔬 启动AutoDock结果查看器...")
    
    try:
        result = subprocess.run([sys.executable, "autodock_results_viewer.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            safe_print("✅ AutoDock结果查看器生成成功！")
            safe_print(result.stdout)
        else:
            safe_print("❌ AutoDock结果查看器生成失败:")
            safe_print(result.stderr)
            
    except Exception as e:
        safe_print(f"❌ 运行AutoDock结果查看器时出错: {e}")

def generate_3d_visualization():
    """生成3D可视化报告"""
    safe_print("📊 生成3D可视化报告...")
    
    try:
        result = subprocess.run([sys.executable, "visualize_3d_simple.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            safe_print("✅ 3D可视化报告生成成功！")
            safe_print(result.stdout)
        else:
            safe_print("❌ 3D可视化报告生成失败:")
            safe_print(result.stderr)
            
    except Exception as e:
        safe_print(f"❌ 生成3D可视化报告时出错: {e}")

def view_all_3d_results():
    """查看所有3D结果"""
    safe_print("🎬 查看所有3D结果...")
    
    try:
        result = subprocess.run([sys.executable, "view_all_3d_results.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            safe_print("✅ 3D结果总览生成成功！")
            safe_print(result.stdout)
        else:
            safe_print("❌ 3D结果总览生成失败:")
            safe_print(result.stderr)
            
    except Exception as e:
        safe_print(f"❌ 查看3D结果时出错: {e}")

def run_system_diagnosis():
    """运行系统诊断"""
    safe_print("🔧 运行系统诊断...")
    
    # 检查Python版本
    safe_print(f"Python版本: {sys.version}")
    
    # 检查工作目录
    safe_print(f"工作目录: {os.getcwd()}")
    
    # 检查关键文件
    key_files = [
        "unified_web_interface.py",
        "scripts/config.py",
        "scripts/ligand_generator.py",
        "scripts/docking_engine.py",
        "scripts/admet_analyzer.py",
        "data/1p65.pdbqt",
        "data/vina.exe"
    ]
    
    safe_print("\n关键文件检查:")
    for file_path in key_files:
        if os.path.exists(file_path):
            safe_print(f"✅ {file_path}")
        else:
            safe_print(f"❌ {file_path}")
    
    # 检查结果目录
    results_dir = Path("results")
    if results_dir.exists():
        result_dirs = list(results_dir.iterdir())
        safe_print(f"\n结果目录: {len(result_dirs)} 个结果文件夹")
    else:
        safe_print("\n结果目录: 不存在")
    
    # 检查依赖
    safe_print("\n依赖检查:")
    check_dependencies()

def show_help():
    """显示帮助文档"""
    safe_print("""
📖 PRRSV抑制剂设计平台帮助文档
================================

🎯 项目概述:
本平台专注于设计靶向PRRSV衣壳蛋白与整合素相互作用的小分子抑制剂。

✨ 核心功能:
1. 🧪 智能配体生成 - 基于模板库和深度学习
2. 🔬 高精度分子对接 - AutoDock Vina引擎
3. 📈 全面ADMET分析 - 药物类似性评估
4. 🌐 3D可视化系统 - 交互式3D展示
5. 📊 综合结果分析 - 多维度评分排序

🚀 推荐使用流程:
1. 启动统一Web界面 (选项1)
2. 在Web界面中运行完整工作流程
3. 查看结果分析和3D可视化
4. 导出和分析候选分子

🔧 故障排除:
- 如果Web界面无法启动，检查端口8504是否被占用
- 如果对接失败，推荐使用AF-F1SR53-F1-model_v4.pdbqt受体
- 如果3D可视化不显示，检查py3dmol和plotly依赖

📞 技术支持:
查看README.md文件获取详细文档和使用说明。
""")

def main():
    """主函数"""
    safe_print("🧬 PRRSV抑制剂设计平台启动器")
    safe_print("="*50)
    
    # 检查依赖
    if not check_dependencies():
        safe_print("\n❌ 依赖检查失败，请先安装依赖")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("请选择 (1-9): ").strip()
            
            if choice == "1":
                start_web_interface()
            elif choice == "2":
                run_complete_workflow()
            elif choice == "3":
                run_molecule_3d_viewer()
            elif choice == "4":
                run_autodock_viewer()
            elif choice == "5":
                generate_3d_visualization()
            elif choice == "6":
                view_all_3d_results()
            elif choice == "7":
                run_system_diagnosis()
            elif choice == "8":
                show_help()
            elif choice == "9":
                safe_print("👋 再见！")
                break
            else:
                safe_print("❌ 无效选择，请输入1-9")
            
            input("\n按Enter键继续...")
            
        except KeyboardInterrupt:
            safe_print("\n👋 再见！")
            break
        except Exception as e:
            safe_print(f"❌ 出错: {e}")

if __name__ == "__main__":
    main()
