#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果管理器
负责创建和管理每次运行的结果文件夹，确保数据连续性
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class ResultManager:
    """结果管理器"""
    
    def __init__(self, base_results_dir: str = "results"):
        """
        初始化结果管理器
        
        Args:
            base_results_dir: 基础结果目录
        """
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # 当前运行的结果目录
        self.current_run_dir = None
        self.current_run_info = {}
        
    def create_new_run_directory(self) -> Path:
        """
        创建新的运行结果目录
        
        Returns:
            新创建的运行目录路径
        """
        # 获取当前日期
        today = datetime.now().strftime("%Y%m%d")
        
        # 查找今天已有的运行次数
        existing_runs = list(self.base_results_dir.glob(f"run_{today}_*"))
        run_count = len(existing_runs) + 1
        
        # 创建新的运行目录
        run_dir_name = f"run_{today}_{run_count:03d}"
        self.current_run_dir = self.base_results_dir / run_dir_name
        self.current_run_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        subdirs = [
            "ligands",           # 生成的配体
            "docking",           # 分子对接结果
            "admet",             # ADMET分析结果
            "visualization_2d",  # 2D可视化
            "visualization_3d",  # 3D可视化
            "reports"            # 综合报告
        ]
        
        for subdir in subdirs:
            (self.current_run_dir / subdir).mkdir(exist_ok=True)
        
        # 记录运行信息
        self.current_run_info = {
            "run_id": run_dir_name,
            "start_time": datetime.now().isoformat(),
            "date": today,
            "run_number": run_count,
            "status": "started",
            "steps_completed": [],
            "files_generated": {}
        }
        
        # 保存运行信息
        self.save_run_info()
        
        logger.info(f"创建新的运行目录: {self.current_run_dir}")
        return self.current_run_dir
    
    def get_current_run_dir(self) -> Optional[Path]:
        """获取当前运行目录"""
        return self.current_run_dir
    
    def get_ligands_dir(self) -> Path:
        """获取配体目录"""
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        return self.current_run_dir / "ligands"
    
    def get_docking_dir(self) -> Path:
        """获取对接结果目录"""
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        return self.current_run_dir / "docking"
    
    def get_admet_dir(self) -> Path:
        """获取ADMET结果目录"""
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        return self.current_run_dir / "admet"
    
    def get_2d_viz_dir(self) -> Path:
        """获取2D可视化目录"""
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        return self.current_run_dir / "visualization_2d"
    
    def get_3d_viz_dir(self) -> Path:
        """获取3D可视化目录"""
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        return self.current_run_dir / "visualization_3d"
    
    def get_reports_dir(self) -> Path:
        """获取报告目录"""
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        return self.current_run_dir / "reports"
    
    def save_run_info(self):
        """保存运行信息"""
        if not self.current_run_dir:
            return
        
        info_file = self.current_run_dir / "run_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_run_info, f, indent=2, ensure_ascii=False)
    
    def update_step_completed(self, step_name: str, files: List[str] = None):
        """
        更新完成的步骤
        
        Args:
            step_name: 步骤名称
            files: 生成的文件列表
        """
        if step_name not in self.current_run_info["steps_completed"]:
            self.current_run_info["steps_completed"].append(step_name)
        
        if files:
            self.current_run_info["files_generated"][step_name] = files
        
        self.save_run_info()
        logger.info(f"步骤完成: {step_name}")
    
    def get_latest_ligands_file(self) -> Optional[Path]:
        """获取最新的配体文件"""
        ligands_dir = self.get_ligands_dir()
        csv_files = list(ligands_dir.glob("*.csv"))
        if csv_files:
            return max(csv_files, key=lambda x: x.stat().st_mtime)
        return None
    
    def get_latest_docking_file(self) -> Optional[Path]:
        """获取最新的对接结果文件"""
        docking_dir = self.get_docking_dir()
        csv_files = list(docking_dir.glob("*docking*.csv"))
        if csv_files:
            return max(csv_files, key=lambda x: x.stat().st_mtime)
        return None
    
    def get_latest_admet_file(self) -> Optional[Path]:
        """获取最新的ADMET结果文件"""
        admet_dir = self.get_admet_dir()
        csv_files = list(admet_dir.glob("*admet*.csv"))
        if csv_files:
            return max(csv_files, key=lambda x: x.stat().st_mtime)
        return None
    
    def copy_file_to_current_run(self, source_file: Path, target_subdir: str, 
                                new_name: str = None) -> Path:
        """
        复制文件到当前运行目录
        
        Args:
            source_file: 源文件路径
            target_subdir: 目标子目录
            new_name: 新文件名（可选）
            
        Returns:
            目标文件路径
        """
        if not self.current_run_dir:
            raise ValueError("未设置当前运行目录")
        
        target_dir = self.current_run_dir / target_subdir
        target_dir.mkdir(exist_ok=True)
        
        if new_name:
            target_file = target_dir / new_name
        else:
            target_file = target_dir / source_file.name
        
        shutil.copy2(source_file, target_file)
        logger.info(f"文件已复制: {source_file} -> {target_file}")
        
        return target_file
    
    def finalize_run(self):
        """完成当前运行"""
        if not self.current_run_dir:
            return
        
        self.current_run_info["status"] = "completed"
        self.current_run_info["end_time"] = datetime.now().isoformat()
        self.save_run_info()
        
        logger.info(f"运行完成: {self.current_run_dir}")
    
    def list_all_runs(self) -> List[Dict]:
        """列出所有运行记录"""
        runs = []
        
        for run_dir in sorted(self.base_results_dir.glob("run_*")):
            info_file = run_dir / "run_info.json"
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        run_info = json.load(f)
                    run_info["directory"] = str(run_dir)
                    runs.append(run_info)
                except Exception as e:
                    logger.warning(f"无法读取运行信息: {info_file}, 错误: {e}")
        
        return runs
    
    def get_run_summary(self) -> Dict:
        """获取当前运行的摘要"""
        if not self.current_run_dir or not self.current_run_info:
            return {}
        
        # 统计生成的文件
        file_counts = {}
        for subdir in ["ligands", "docking", "admet", "visualization_2d", "visualization_3d", "reports"]:
            subdir_path = self.current_run_dir / subdir
            if subdir_path.exists():
                file_counts[subdir] = len(list(subdir_path.glob("*")))
            else:
                file_counts[subdir] = 0
        
        return {
            "run_info": self.current_run_info,
            "file_counts": file_counts,
            "total_files": sum(file_counts.values())
        }

# 全局结果管理器实例
result_manager = ResultManager()
