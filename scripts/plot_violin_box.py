#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 run_YYYYMMDD_xxx/ligands/ 下的 CSV 数据，生成小提琴图与箱线图。
支持文件：
- generated_ligands.csv
- dl_phase2_generated_molecules.csv
- dl_phase3_optimized_molecules.csv

输出：默认保存至 {run_dir}/analysis_plots/
- violin_{metric}.png
- box_{metric}.png
- summary_stats.csv（各数据集的描述性统计）

依赖：pandas、matplotlib，可选 seaborn（更美观）。如未安装 seaborn，将回退到 matplotlib 原生绘图。
示例：
python HJD/scripts/plot_violin_box.py \
  --ligand-dir HJD/results/run_20250929_002/ligands \
  --out-dir HJD/results/run_20250929_002/analysis_plots \
  --metrics binding_affinity molecular_weight logp qed sa_score tpsa
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_context("talk")
    sns.set_style("whitegrid")
except Exception:
    HAS_SEABORN = False


def _read_csv_safe(csv_path: Path, dataset: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        df["dataset"] = dataset
        return df
    except Exception:
        return pd.DataFrame()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def collect_ligand_tables(ligand_dir: Path) -> pd.DataFrame:
    candidates: List[Tuple[str, str]] = [
        ("generated_ligands.csv", "generated"),
        ("dl_phase2_generated_molecules.csv", "phase2"),
        ("dl_phase3_optimized_molecules.csv", "phase3"),
    ]
    frames = []
    for fname, label in candidates:
        fp = ligand_dir / fname
        df = _read_csv_safe(fp, label)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def describe_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    sub = df[["dataset", metric]].dropna()
    if sub.empty:
        return pd.DataFrame()
    desc = sub.groupby("dataset")[metric].agg(
        count="count", mean="mean", std="std", min="min", median="median", max="max"
    ).reset_index()
    # 计算分位数
    q = sub.groupby("dataset")[metric].quantile([0.25, 0.75]).unstack().reset_index()
    q.columns = ["dataset", "q1", "q3"]
    return desc.merge(q, on="dataset", how="left")


def plot_violin(df: pd.DataFrame, metric: str, out: Path):
    sub = df[["dataset", metric]].dropna()
    if sub.empty:
        return
    plt.figure(figsize=(8, 5))
    if HAS_SEABORN:
        ax = sns.violinplot(data=sub, x="dataset", y=metric, inner="box", cut=0)
        ax.set_xlabel("dataset")
        ax.set_ylabel(metric)
        ax.set_title(f"Violin plot of {metric}")
    else:
        # matplotlib 回退实现
        groups = [g[metric].values for _, g in sub.groupby("dataset")]
        labels = list(sub.groupby("dataset").groups.keys())
        parts = plt.violinplot(groups, showmeans=False, showmedians=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel(metric)
        plt.title(f"Violin plot of {metric}")
        # 轻微美化
        for pc in parts['bodies']:
            pc.set_facecolor('#1f77b4')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_box(df: pd.DataFrame, metric: str, out: Path):
    sub = df[["dataset", metric]].dropna()
    if sub.empty:
        return
    plt.figure(figsize=(8, 5))
    if HAS_SEABORN:
        ax = sns.boxplot(data=sub, x="dataset", y=metric, showfliers=False)
        ax = sns.stripplot(data=sub, x="dataset", y=metric, color="0.3", alpha=0.4, jitter=0.25)
        plt.xlabel("dataset")
        plt.ylabel(metric)
        plt.title(f"Box plot of {metric}")
    else:
        groups = [g[metric].values for _, g in sub.groupby("dataset")]
        labels = list(sub.groupby("dataset").groups.keys())
        plt.boxplot(groups, showfliers=False)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel(metric)
        plt.title(f"Box plot of {metric}")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand-dir", type=str, required=True, help="包含 CSV 的 ligands 目录")
    ap.add_argument("--out-dir", type=str, default=None, help="输出目录（默认：{ligand-dir}/../analysis_plots）")
    ap.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            "binding_affinity", "molecular_weight", "logp",
            "qed", "sa_score", "tpsa", "hbd", "hba", "rotatable_bonds",
            "pred_binding_affinity"
        ],
        help="需要绘图的数值列名列表（存在即绘制，缺失将跳过）",
    )
    args = ap.parse_args()

    ligand_dir = Path(args.ligand_dir).expanduser().resolve()
    if not ligand_dir.exists():
        raise SystemExit(f"Ligand 目录不存在: {ligand_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (ligand_dir.parent / "analysis_plots")
    ensure_out_dir(out_dir)

    # 读入并合并
    df = collect_ligand_tables(ligand_dir)
    if df.empty:
        raise SystemExit(f"未在 {ligand_dir} 下找到可用的 CSV 文件")

    # 尝试将常见数值列转为数值
    numeric_candidates = set(args.metrics)
    numeric_candidates.update(["molecular_weight", "logp", "qed", "sa_score", "tpsa", "hbd", "hba", "rotatable_bonds", "binding_affinity", "pred_binding_affinity"])
    df = _coerce_numeric(df, list(numeric_candidates))

    # 过滤所有数值不可用的指标
    available_metrics = []
    for m in args.metrics:
        if m in df.columns and df[m].apply(lambda x: pd.notna(x)).any():
            available_metrics.append(m)

    if not available_metrics:
        raise SystemExit("未找到任何可用的数值指标。请检查 CSV 列或通过 --metrics 指定存在的列")

    # 汇总统计
    all_stats = []
    for metric in available_metrics:
        stats = describe_stats(df, metric)
        if not stats.empty:
            stats.insert(0, "metric", metric)
            all_stats.append(stats)
    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        stats_path = out_dir / f"summary_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"[OK] 保存统计: {stats_path}")

    # 绘图
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for metric in available_metrics:
        vout = out_dir / f"violin_{metric}_{ts}.png"
        bout = out_dir / f"box_{metric}_{ts}.png"
        try:
            plot_violin(df, metric, vout)
            print(f"[OK] 保存: {vout}")
        except Exception as e:
            print(f"[WARN] 绘制小提琴图失败: {metric}: {e}")
        try:
            plot_box(df, metric, bout)
            print(f"[OK] 保存: {bout}")
        except Exception as e:
            print(f"[WARN] 绘制箱线图失败: {metric}: {e}")

    print("完成。")


if __name__ == "__main__":
    main()
