#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D分子图像生成器
基于分子对接结果生成2D分子结构图像
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'scripts'))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDKit导入处理
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
    logger.info("RDKit已成功导入，将使用完整2D分子图像生成功能")
except ImportError:
    logger.warning("RDKit不可用，将使用简化模式")
    # 创建虚拟模块
    class MockChem:
        @staticmethod
        def MolFromSmiles(smiles):
            return None
    
    class MockDraw:
        @staticmethod
        def MolToImage(mol, **kwargs):
            return None
    
    class MockDescriptors:
        @staticmethod
        def MolWt(mol):
            return 300.0
    
    Chem = MockChem()
    Draw = MockDraw()
    Descriptors = MockDescriptors()

class Molecule2DGenerator:
    """2D分子图像生成器"""

    def __init__(self, output_dir: str = "molecule_2d_output"):
        """
        初始化2D分子图像生成器

        Args:
            output_dir: 输出目录
        """
        # 尝试使用结果管理器的当前运行目录
        try:
            from scripts.result_manager import result_manager

            # 如果有当前运行目录，使用它；否则使用默认目录
            if result_manager.current_run_dir:
                self.output_dir = result_manager.get_2d_viz_dir()
                logger.info(f"使用结果管理器的2D可视化目录: {self.output_dir}")
            else:
                self.output_dir = Path(output_dir)
                logger.info(f"使用默认2D可视化目录: {self.output_dir}")
        except ImportError:
            self.output_dir = Path(output_dir)
            logger.info(f"结果管理器不可用，使用默认目录: {self.output_dir}")

        self.output_dir.mkdir(exist_ok=True)

        # 创建子目录
        self.individual_dir = self.output_dir / "individual"
        self.individual_dir.mkdir(exist_ok=True)

        self.grid_dir = self.output_dir / "grids"
        self.grid_dir.mkdir(exist_ok=True)

        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def process_latest_results(self, top_n: int = 20) -> Optional[Dict]:
        """
        处理最新结果并生成2D图像
        
        Args:
            top_n: 处理的分子数量
            
        Returns:
            生成结果字典
        """
        try:
            # 尝试使用结果管理器获取当前运行目录的数据文件（优先对接结果，失败回退到配体CSV）
            data_file = None
            data_source = None

            try:
                from scripts.result_manager import result_manager
                if result_manager.current_run_dir:
                    # 1) 优先对接结果
                    docking_file = result_manager.get_latest_docking_file()
                    if docking_file and docking_file.exists():
                        data_file = docking_file
                        data_source = "docking"
                        logger.info(f"使用结果管理器的对接结果: {data_file}")
                    else:
                        # 2) 回退到当前run的配体结果
                        lig_dir = result_manager.get_ligands_dir()
                        candidates = [
                            lig_dir / "generated_ligands.csv",
                            lig_dir / "dl_phase3_optimized_molecules.csv",
                            lig_dir / "dl_phase2_generated_molecules.csv",
                        ]
                        for c in candidates:
                            if c.exists():
                                data_file = c
                                data_source = "ligands"
                                logger.info(f"使用结果管理器的配体结果: {data_file}")
                                break
                        if not data_file:
                            # 任意CSV回退
                            any_csv = list(lig_dir.glob("*.csv"))
                            if any_csv:
                                data_file = sorted(any_csv, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                                data_source = "ligands"
                                logger.info(f"使用结果管理器的配体结果(通配): {data_file}")
            except ImportError:
                logger.warning("结果管理器不可用")

            # 如果结果管理器不可用或没有找到文件，回退到按目录扫描
            if not data_file:
                logger.info("回退到自动查找最新结果（对接优先，其次配体）")
                results_dir = Path("results")
                if not results_dir.exists():
                    logger.error("结果目录不存在")
                    return None

                # 查找最新的运行目录
                run_dirs = [d for d in results_dir.glob("run_*") if d.is_dir()]
                if not run_dirs:
                    logger.error("未找到运行目录")
                    return None

                for run_dir in sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
                    # 1) 对接
                    docking_file_candidate = run_dir / "docking" / "docking_results.csv"
                    if docking_file_candidate.exists():
                        data_file = docking_file_candidate
                        data_source = "docking"
                        logger.info(f"使用最新运行目录的对接结果: {data_file}")
                        break
                    # 2) 配体（多种命名）
                    lig_dir = run_dir / "ligands"
                    candidates = [
                        lig_dir / "generated_ligands.csv",
                        lig_dir / "dl_phase3_optimized_molecules.csv",
                        lig_dir / "dl_phase2_generated_molecules.csv",
                    ]
                    for c in candidates:
                        if c.exists():
                            data_file = c
                            data_source = "ligands"
                            logger.info(f"使用最新运行目录的配体结果: {data_file}")
                            break
                    if data_file:
                        break

            if not data_file:
                logger.error("未找到可用于2D绘制的对接或配体结果文件")
                return None

            # 读取数据文件
            df = pd.read_csv(data_file)
            if 'smiles' not in df.columns:
                logger.error(f"数据文件缺少 'smiles' 列: {data_file}")
                return None
            if 'binding_affinity' not in df.columns:
                logger.warning("数据文件缺少 'binding_affinity' 列，将使用0占位")
                df['binding_affinity'] = 0.0
            logger.info(f"读取到 {len(df)} 个分子，数据源: {data_source}, 文件: {data_file}")

            # 按结合亲和力排序，选择前top_n个分子
            df_sorted = df.sort_values('binding_affinity', ascending=True)
            top_molecules = df_sorted.head(top_n)
            
            # 生成2D图像
            individual_images = []
            for idx, row in top_molecules.iterrows():
                smiles = row.get('smiles', '')
                if smiles:
                    image_path = self.generate_single_molecule_image(
                        smiles, idx + 1, row.get('binding_affinity', 0)
                    )
                    if image_path:
                        individual_images.append(image_path)
            
            # 生成网格图
            grid_image = self.generate_grid_image(individual_images, top_n)
            
            # 生成HTML报告
            report_path = self.generate_html_report(
                individual_images, grid_image, top_molecules
            )
            
            return {
                'molecules_count': len(individual_images),
                'individual_images': individual_images,
                'grid_image': grid_image,
                'report_path': report_path
            }
            
        except Exception as e:
            logger.error(f"处理最新结果失败: {e}")
            return None
    
    def generate_single_molecule_image(self, smiles: str, molecule_id: int, 
                                     binding_affinity: float) -> Optional[str]:
        """
        生成单个分子的2D图像
        
        Args:
            smiles: SMILES字符串
            molecule_id: 分子ID
            binding_affinity: 结合亲和力
            
        Returns:
            图像文件路径
        """
        try:
            if not RDKIT_AVAILABLE:
                # 使用简化模式
                return self._generate_simple_image(smiles, molecule_id, binding_affinity)
            
            # 使用RDKit生成图像
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"无法解析SMILES: {smiles}")
                return None
            
            # 计算分子性质
            mw = Descriptors.MolWt(mol)
            
            # 生成图像
            img = Draw.MolToImage(mol, size=(400, 400))
            
            # 保存图像
            filename = f"molecule_{molecule_id:02d}_{smiles[:20]}.png"
            image_path = self.individual_dir / filename
            
            img.save(image_path)
            logger.info(f"生成单个分子图像: {image_path}")
            
            return str(image_path)
            
        except Exception as e:
            logger.error(f"生成单个分子图像失败: {e}")
            return None
    
    def _generate_simple_image(self, smiles: str, molecule_id: int, 
                             binding_affinity: float) -> Optional[str]:
        """
        生成简化的分子图像（当RDKit不可用时）
        
        Args:
            smiles: SMILES字符串
            molecule_id: 分子ID
            binding_affinity: 结合亲和力
            
        Returns:
            图像文件路径
        """
        try:
            # 创建简单的文本图像
            from PIL import Image, ImageDraw, ImageFont
            
            # 创建图像
            img = Image.new('RGB', (400, 400), color='white')
            draw = ImageDraw.Draw(img)
            
            # 尝试使用默认字体
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 绘制文本
            text_lines = [
                f"分子 #{molecule_id}",
                f"SMILES: {smiles[:30]}...",
                f"结合亲和力: {binding_affinity:.2f} kcal/mol",
                f"分子量: ~{len(smiles) * 12:.0f} Da"
            ]
            
            y_offset = 50
            for line in text_lines:
                draw.text((20, y_offset), line, fill='black', font=font)
                y_offset += 30
            
            # 绘制分子结构表示
            draw.rectangle([50, 200, 350, 350], outline='blue', width=2)
            draw.text((60, 220), "分子结构", fill='blue', font=font)
            draw.text((60, 250), "（需要RDKit支持）", fill='gray', font=font)
            
            # 保存图像
            filename = f"molecule_{molecule_id:02d}_simple.png"
            image_path = self.individual_dir / filename
            img.save(image_path)
            
            logger.info(f"生成简化分子图像: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"生成简化图像失败: {e}")
            return None
    
    def generate_grid_image(self, individual_images: List[str], top_n: int) -> Optional[str]:
        """
        生成分子网格图
        
        Args:
            individual_images: 单个分子图像路径列表
            top_n: 分子数量
            
        Returns:
            网格图文件路径
        """
        try:
            if not individual_images:
                logger.warning("没有单个分子图像，跳过网格图生成")
                return None
            
            # 计算网格大小
            cols = min(5, len(individual_images))
            rows = (len(individual_images) + cols - 1) // cols
            
            # 创建网格图像
            from PIL import Image
            
            # 单个图像大小
            img_size = 200
            grid_width = cols * img_size
            grid_height = rows * img_size
            
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            
            for i, img_path in enumerate(individual_images[:top_n]):
                if os.path.exists(img_path):
                    # 计算位置
                    row = i // cols
                    col = i % cols
                    x = col * img_size
                    y = row * img_size
                    
                    # 加载并调整图像大小
                    img = Image.open(img_path)
                    img = img.resize((img_size, img_size))
                    
                    # 粘贴到网格中
                    grid_img.paste(img, (x, y))
            
            # 保存网格图
            grid_filename = f"Top_{top_n}_Molecules_grid.png"
            grid_path = self.grid_dir / grid_filename
            grid_img.save(grid_path)
            
            logger.info(f"生成网格图: {grid_path}")
            return str(grid_path)
            
        except Exception as e:
            logger.error(f"生成网格图失败: {e}")
            return None
    
    def generate_html_report(self, individual_images: List[str], 
                           grid_image: Optional[str], 
                           molecules_df: pd.DataFrame) -> Optional[str]:
        """
        生成HTML报告
        
        Args:
            individual_images: 单个分子图像路径列表
            grid_image: 网格图路径
            molecules_df: 分子数据框
            
        Returns:
            HTML报告文件路径
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2D分子结构图像报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .grid-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .individual-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .molecule-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .molecule-item {{
            text-align: center;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background: #fafafa;
        }}
        .molecule-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-item {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1976d2;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 2D分子结构图像报告</h1>
        <p>基于最新分子对接结果生成的2D分子结构可视化</p>
    </div>
    
    <div class="summary">
        <h2>📊 生成摘要</h2>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{len(individual_images)}</div>
                <div class="stat-label">生成图像数</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(molecules_df)}</div>
                <div class="stat-label">处理分子数</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{molecules_df['binding_affinity'].min():.2f}</div>
                <div class="stat-label">最佳结合能 (kcal/mol)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{molecules_df['binding_affinity'].mean():.2f}</div>
                <div class="stat-label">平均结合能 (kcal/mol)</div>
            </div>
        </div>
    </div>
"""
            
            # 添加网格图
            if grid_image and os.path.exists(grid_image):
                html_content += f"""
    <div class="grid-section">
        <h2>🔬 分子结构网格图</h2>
        <img src="{os.path.basename(grid_image)}" alt="分子网格图" style="max-width: 100%; height: auto;">
    </div>
"""
            
            # 添加单个分子图像
            if individual_images:
                html_content += """
    <div class="individual-section">
        <h2>🧪 单个分子结构</h2>
        <div class="molecule-grid">
"""
                
                for i, img_path in enumerate(individual_images):
                    if os.path.exists(img_path):
                        img_name = os.path.basename(img_path)
                        html_content += f"""
            <div class="molecule-item">
                <img src="individual/{img_name}" alt="分子 {i+1}">
                <p>分子 #{i+1}</p>
            </div>
"""
                
                html_content += """
        </div>
    </div>
"""
            
            html_content += """
</body>
</html>
"""
            
            # 保存HTML报告
            report_filename = f"top_{len(individual_images)}_molecules_2d_report.html"
            report_path = self.reports_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"生成HTML报告: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            return None

def main():
    """主函数"""
    generator = Molecule2DGenerator()
    results = generator.process_latest_results(top_n=20)
    
    if results:
        print("✅ 2D分子图像生成成功！")
        print(f"生成图像数: {results['molecules_count']}")
        print(f"网格图: {results['grid_image']}")
        print(f"HTML报告: {results['report_path']}")
    else:
        print("❌ 2D分子图像生成失败")

if __name__ == "__main__":
    main()
