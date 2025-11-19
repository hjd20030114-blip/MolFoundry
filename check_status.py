#!/usr/bin/env python3
"""
检查PRRSV深度学习平台状态
"""

import sys
import os
import json
import subprocess
import platform

def check_python_environment():
    """检查Python环境"""
    print("🐍 Python环境检查")
    print(f"   版本: {sys.version}")
    print(f"   平台: {platform.platform()}")
    print(f"   架构: {platform.machine()}")
    print()

def check_basic_packages():
    """检查基础包"""
    print("📦 基础包检查")
    
    basic_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'scipy'
    ]
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
    print()

def check_deep_learning_packages():
    """检查深度学习包"""
    print("🧠 深度学习包检查")
    
    dl_packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('e3nn', 'E3NN'),
        ('rdkit', 'RDKit'),
        ('sklearn', 'Scikit-learn')
    ]
    
    for package, name in dl_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (可选)")
    print()

def check_project_structure():
    """检查项目结构"""
    print("📁 项目结构检查")
    
    required_files = [
        'unified_web_interface.py',
        'deep_learning_pipeline.py',
        'deep_learning_config.json',
        'deep_learning/__init__.py',
        'deep_learning/models/__init__.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
    print()

def check_web_interface():
    """检查Web界面状态"""
    print("🌐 Web界面状态检查")
    
    try:
        # 检查端口8501是否被占用
        result = subprocess.run(['lsof', '-i', ':8501'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            print("   ✅ Web界面正在运行 (端口8501)")
            print("   🔗 访问地址: http://localhost:8501")
        else:
            print("   ❌ Web界面未运行")
    except Exception as e:
        print(f"   ⚠️  无法检查Web界面状态: {e}")
    print()

def check_config_file():
    """检查配置文件"""
    print("⚙️  配置文件检查")
    
    config_file = 'deep_learning_config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"   ✅ {config_file} (有效)")
            print(f"   📊 配置项: {len(config)} 个主要部分")
        except Exception as e:
            print(f"   ❌ {config_file} (无效): {e}")
    else:
        print(f"   ❌ {config_file} (不存在)")
    print()

def check_data_directories():
    """检查数据目录"""
    print("📂 数据目录检查")
    
    data_dirs = [
        'experiment_report',
        'results',
        'data',
        'scripts'
    ]
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path)) if os.path.isdir(dir_path) else 0
            print(f"   ✅ {dir_path}/ ({file_count} 项)")
        else:
            print(f"   ❌ {dir_path}/")
    print()

def get_installation_suggestions():
    """获取安装建议"""
    print("💡 安装建议")
    
    # 检查缺失的包
    missing_basic = []
    basic_packages = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy']
    
    for package in basic_packages:
        try:
            __import__(package)
        except ImportError:
            missing_basic.append(package)
    
    if missing_basic:
        print("   基础包安装:")
        print(f"   pip install {' '.join(missing_basic)}")
    
    # 深度学习包建议
    print("   深度学习包安装 (可选):")
    print("   pip install torch")
    print("   pip install torch-geometric")
    print("   pip install e3nn")
    print("   pip install rdkit-pypi")
    print("   pip install scikit-learn")
    print()

def main():
    """主函数"""
    print("🚀 PRRSV深度学习平台状态检查")
    print("=" * 60)
    print()
    
    check_python_environment()
    check_basic_packages()
    check_deep_learning_packages()
    check_project_structure()
    check_web_interface()
    check_config_file()
    check_data_directories()
    get_installation_suggestions()
    
    print("=" * 60)
    print("✨ 状态检查完成！")
    print()
    print("📋 快速启动指南:")
    print("   1. 启动Web界面: streamlit run unified_web_interface.py")
    print("   2. 访问地址: http://localhost:8501")
    print("   3. 选择 '🤖 深度学习流水线' 页面")
    print("   4. 开始AI驱动的PRRSV抑制剂设计")

if __name__ == "__main__":
    main()
