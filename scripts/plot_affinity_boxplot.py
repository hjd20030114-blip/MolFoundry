#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成不同模型关于结合亲和力（kcal/mol）的对比箱线图（示意数据）。
- 排名期望：MolFoundry < BIMODAL < QADD < Transformer < Diffusion（数值越低越好）
- 约束：MolFoundry 最好 -9.201，最差 -7.102
输出：HJD/exports/affinity_boxplot.png 和 affinity_boxplot.svg
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 全局风格
plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
})

rng = np.random.default_rng(20251005)

# 合成数据生成函数：在给定区间内截断正态分布，并强制包含边界点（可选）
def gen_truncated_normal(n: int, mean: float, sd: float, low: float, high: float, force_edges: tuple[float, float] | None = None):
    samples = rng.normal(mean, sd, size=n)
    samples = np.clip(samples, low, high)
    if force_edges is not None and n >= 2:
        samples[0] = force_edges[0]
        samples[1] = force_edges[1]
    return samples

# 生成各模型数据（单位：kcal/mol，数值更低更好）
# MolFoundry：满足用户给定区间 [-9.201, -7.102]
molfoundry = gen_truncated_normal(n=80, mean=-8.15, sd=0.35, low=-9.201, high=-7.102, force_edges=(-9.201, -7.102))
# 对比模型：设置区间与均值以体现排名
bimodal     = gen_truncated_normal(n=80, mean=-7.85, sd=0.40, low=-8.7,  high=-6.9)
qadd        = gen_truncated_normal(n=80, mean=-7.55, sd=0.40, low=-8.3,  high=-6.6)
transformer = gen_truncated_normal(n=80, mean=-7.20, sd=0.35, low=-7.9,  high=-6.2)
diffusion   = gen_truncated_normal(n=80, mean=-6.90, sd=0.35, low=-7.4,  high=-5.9)

labels = ["MolFoundry", "BIMODAL", "QADD", "Transformer", "Diffusion"]
data = [molfoundry, bimodal, qadd, transformer, diffusion]

# 创建输出目录
out_dir = Path("HJD/exports")
out_dir.mkdir(parents=True, exist_ok=True)

# 画图
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
box = ax.boxplot(
    data,
    labels=labels,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    widths=0.6,
)

# 配色与样式
colors = ["#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#e53935"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
for whisker in box['whiskers']:
    whisker.set_color('#455a64')
    whisker.set_linewidth(1.5)
for cap in box['caps']:
    cap.set_color('#455a64')
    cap.set_linewidth(1.5)
for median in box['medians']:
    median.set_color('#263238')
    median.set_linewidth(2.0)
for mean in box['means']:
    mean.set_color('#0d47a1')
    mean.set_linewidth(2.0)

ax.set_title("不同模型结合亲和力对比（箱线图示意）", fontsize=14)
ax.set_ylabel("结合亲和力 (kcal/mol) — 越低越好", fontsize=12)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)

# 可选：反转y轴使更低的值更靠上（符合“越低越好”的直觉）
ax.invert_yaxis()

# 统计注释（在箱线图上方显示每组中位数）
ymax = max(np.min(d) for d in data)  # 注意y轴反转，取每组最小值作为顶部基线
for i, d in enumerate(data, start=1):
    median = float(np.median(d))
    ax.text(i, median, f"median={median:.2f}", ha='center', va='bottom', fontsize=10, color='#263238',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

png_path = out_dir / "affinity_boxplot.png"
svg_path = out_dir / "affinity_boxplot.svg"
fig.tight_layout()
fig.savefig(png_path, dpi=600, bbox_inches='tight')
fig.savefig(svg_path, dpi=600, bbox_inches='tight')
print(f"[OK] 导出: {png_path}")
print(f"[OK] 导出: {svg_path}")
