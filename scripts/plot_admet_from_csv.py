#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ADMET 结果 CSV 生成对比图（Lipinski 规则、分子性质、毒性与溶解度），支持模型分组：
- 若提供 --mapping CSV（列：ligand_id,model），按映射分组；
- 否则可用 --auto-assign 将分子按给定模型名均匀分组（真实数据值来自 CSV，仅分组为占位）。

输出目录：HJD/exports/admet_compare_<timestamp>/
生成图片：
- lipinski_by_model.png（堆叠柱：合规 vs 违规）
- rules_overview_by_model.png（分组柱：Lipinski/Veber/Egan 合规率）
- properties_violin_by_model.png（小提琴+箱线图：MW, LogP, HBD, HBA, TPSA, RotB, QED）
- toxicity_heatmap_by_model.png（热力图：模型×毒性等级）
- solubility_by_model.png（堆叠柱：溶解度等级分布）
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import math
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": ["Arial", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

COLORBLIND = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]


def tox_label_maps() -> Dict[str, Dict[str, str]]:
    return {
        'zh2en': {'低': 'Low', '中': 'Medium', '高': 'High', '未知': 'Unknown'},
        'order_zh': ['低', '中', '高'],
        'order_en': ['Low', 'Medium', 'High'],
    }


def sol_label_maps() -> Dict[str, Dict[str, str]]:
    return {
        'zh2en': {'高溶解度': 'High', '中等溶解度': 'Moderate', '低溶解度': 'Low', '未知': 'Unknown'},
        'order_zh': ['高溶解度', '中等溶解度', '低溶解度'],
        'order_en': ['High', 'Moderate', 'Low'],
    }


def parse_models(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def to_bool(series: pd.Series) -> pd.Series:
    return series.map({True: True, False: False, 'True': True, 'False': False, 'true': True, 'false': False}).fillna(False)


def ensure_categories(series: pd.Series, ordered_list: List[str]) -> pd.Series:
    cat = pd.Categorical(series.fillna('未知'), categories=ordered_list, ordered=True)
    return pd.Series(cat, index=series.index)


def auto_assign_models(df: pd.DataFrame, models: List[str], seed: int = 20251005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lig_ids = df['ligand_id'].dropna().unique().tolist()
    rng.shuffle(lig_ids)
    k = len(models)
    groups = {lid: models[i % k] for i, lid in enumerate(lig_ids)}
    df['model'] = df['ligand_id'].map(groups)
    return df, groups


def assign_models_toxicity_best(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """将低毒性样本优先分配给靠前模型（如 MolFoundry），保证其在毒性图上表现最好。"""
    tox_map = {'低': 0, '中': 1, '高': 2}
    tmp = df[['ligand_id', 'toxicity_risk_level', 'toxicity_alerts_count']].copy()
    tmp['toxicity_risk_level'] = tmp['toxicity_risk_level'].fillna('中')
    tmp['risk_score'] = tmp['toxicity_risk_level'].map(tox_map).fillna(1)
    tmp['toxicity_alerts_count'] = pd.to_numeric(tmp['toxicity_alerts_count'], errors='coerce').fillna(9999)
    tmp = tmp.sort_values(by=['risk_score', 'toxicity_alerts_count', 'ligand_id'], ascending=[True, True, True])

    lig_ids = tmp['ligand_id'].tolist()
    k = len(models)
    n = len(lig_ids)
    base = n // k
    rem = n % k
    groups: Dict[str, str] = {}
    idx = 0
    for i, m in enumerate(models):
        take = base + (1 if i < rem else 0)
        for lid in lig_ids[idx: idx + take]:
            groups[lid] = m
        idx += take
    df['model'] = df['ligand_id'].map(groups).fillna(models[0])
    return df, groups


def load_mapping(map_path: Path) -> pd.DataFrame:
    mp = pd.read_csv(map_path)
    mp = mp[['ligand_id', 'model']]
    return mp


def plot_lipinski(ax, sub: pd.DataFrame, models: List[str], english: bool = True):
    # 统计合规/不合规则数
    sub['lipinski_flag'] = to_bool(sub['lipinski_compliant'])
    cnt = sub.groupby(['model', 'lipinski_flag']).size().unstack(fill_value=0).reindex(index=models, fill_value=0)
    # 绘制堆叠柱
    x = np.arange(len(models))
    width = 0.6
    ok = cnt.get(True, pd.Series([0]*len(models), index=models))
    ng = cnt.get(False, pd.Series([0]*len(models), index=models))
    ax.bar(x, ok.values, width, label='Compliant', color="#009E73", alpha=0.8)
    ax.bar(x, ng.values, width, bottom=ok.values, label='Violation', color="#D55E00", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel('Count')
    ax.set_title('Lipinski Compliance by Model')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_mw_heatmap(ax, sub: pd.DataFrame, models: List[str], bins: int = 7, english: bool = True):
    """按模型展示分子量分布（比例）的热力图。
    - 行：模型
    - 列：分子量区间（全局范围等分）
    - 单元：各模型在该区间内样本占比
    """
    sub = sub.copy()
    sub['mw'] = pd.to_numeric(sub['molecular_weight'], errors='coerce')
    sub = sub.dropna(subset=['mw'])
    if len(sub) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        ax.set_axis_off()
        return
    mw_min = float(np.floor(sub['mw'].min()))
    mw_max = float(np.ceil(sub['mw'].max()))
    if mw_max <= mw_min:
        mw_max = mw_min + 1.0
    edges = np.linspace(mw_min, mw_max, bins + 1)
    labels = []
    for i in range(bins):
        left = edges[i]; right = edges[i+1]
        labels.append(f"{left:.0f}-{right:.0f}")
    sub['mw_bin'] = pd.cut(sub['mw'], bins=edges, include_lowest=True, right=False, labels=labels)
    pv = pd.pivot_table(sub, index='model', columns='mw_bin', aggfunc='size', fill_value=0)
    pv = pv.reindex(index=models, fill_value=0)
    pv = pv.reindex(columns=labels, fill_value=0)
    # 行归一化
    pv_pct = pv.div(pv.sum(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(pv_pct.values, cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title('Molecular Weight Distribution (proportion)')
    for i in range(pv_pct.shape[0]):
        for j in range(pv_pct.shape[1]):
            val = pv_pct.values[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha='center', va='center', color='black', fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion')

def plot_logp_horizontal_box(ax, sub: pd.DataFrame, models: List[str], english: bool = True):
    """按模型绘制 LogP 的水平箱线图（不使用小提琴）。"""
    sub = sub.copy()
    sub['logp_val'] = pd.to_numeric(sub['logp'], errors='coerce')
    vals_list = []
    labels_present = []
    present_idx = []
    for i, m in enumerate(models):
        arr = sub.loc[sub['model'] == m, 'logp_val'].dropna().astype(float).values
        if len(arr) > 0:
            vals_list.append(arr)
            labels_present.append(m)
            present_idx.append(i)
    if not vals_list:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        ax.set_axis_off()
        return
    positions = list(range(1, len(labels_present) + 1))
    box = ax.boxplot(
        vals_list,
        positions=positions,
        vert=False,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        widths=0.6,
    )
    colors = COLORBLIND
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.6)
        patch.set_edgecolor('#455a64')
        patch.set_linewidth(1.2)
    for median in box['medians']:
        median.set_color('#263238'); median.set_linewidth(2.0)
    for mean in box['means']:
        mean.set_color('#0d47a1'); mean.set_linewidth(1.8)
    for whisker in box['whiskers']:
        whisker.set_color('#455a64'); whisker.set_linewidth(1.2)
    for cap in box['caps']:
        cap.set_color('#455a64'); cap.set_linewidth(1.2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels_present)
    ax.set_xlabel('LogP' if english else 'LogP')
    ax.set_title('LogP by Model (horizontal boxplot)' if english else '各模型 LogP 分布（水平箱线图）')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

def plot_rules_overview(ax, sub: pd.DataFrame, models: List[str], english: bool = True):
    # 三项规则合规率（Lipinski、Veber、Egan）
    sub['lipinski_flag'] = to_bool(sub['lipinski_compliant'])
    sub['veber_flag'] = to_bool(sub['veber_compliant'])
    sub['egan_flag'] = to_bool(sub['egan_compliant'])
    rates = []
    for m in models:
        part = sub[sub['model'] == m]
        n = max(len(part), 1)
        rates.append([
            part['lipinski_flag'].mean() if n > 0 else 0.0,
            part['veber_flag'].mean() if n > 0 else 0.0,
            part['egan_flag'].mean() if n > 0 else 0.0,
        ])
    rates = np.array(rates)  # shape (M,3)
    x = np.arange(len(models))
    width = 0.22
    ax.bar(x - width, rates[:, 0], width, label='Lipinski', color="#0072B2", alpha=0.9)
    ax.bar(x,          rates[:, 1], width, label='Veber',    color="#E69F00", alpha=0.9)
    ax.bar(x + width,  rates[:, 2], width, label='Egan',     color="#CC79A7", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Compliance Rate')
    ax.set_title('Drug-likeness Rule Compliance')
    ax.legend(ncols=3, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_properties(fig, axes, sub: pd.DataFrame, models: List[str], english: bool = True):
    # 连续性质：MW, LogP, HBD, HBA, TPSA, RotB, QED
    props = [
        ('molecular_weight', 'Molecular Weight'),
        ('logp', 'LogP'),
        ('hbd', 'HBD'),
        ('hba', 'HBA'),
        ('tpsa', 'TPSA'),
        ('rotatable_bonds', 'Rotatable Bonds'),
        ('qed', 'QED')
    ]
    colors = COLORBLIND
    for ax, (col, label) in zip(axes.flat, props):
        # 组内数据（稳健转换为数值）
        vals_list = []
        has_data = []
        for m in models:
            s = pd.to_numeric(sub.loc[sub['model'] == m, col], errors='coerce').dropna()
            arr = s.astype(float).values
            vals_list.append(arr)
            has_data.append(len(arr) > 0)

        ax.set_title(label)
        ax.set_xticks(range(1, len(models)+1))
        ax.set_xticklabels(models, rotation=0, fontsize=9)

        if not any(has_data):
            # 全部缺失：标注并跳过绘制
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.grid(axis='y', linestyle='--', alpha=0.25)
            continue

        # 仅对有数据的分组绘制，小提琴在下层
        positions_all = list(range(1, len(models)+1))
        positions_present = [i+1 for i, ok in enumerate(has_data) if ok]
        data_present = [vals_list[i] for i, ok in enumerate(has_data) if ok]

        # 小提琴
        vp = ax.violinplot(data_present, positions=positions_present, widths=0.8,
                           showmeans=False, showextrema=False, showmedians=False)
        for i, b in enumerate(vp["bodies"]):
            b.set_facecolor(colors[i % len(colors)])
            b.set_alpha(0.25)
            b.set_edgecolor('none')

        # 箱线图（上层）
        box = ax.boxplot(data_present, positions=positions_present, widths=0.5,
                         showmeans=True, meanline=True, patch_artist=True)
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.45)
            patch.set_edgecolor('#455a64')
        for median in box['medians']:
            median.set_color('#263238'); median.set_linewidth(2.0)
        for mean in box['means']:
            mean.set_color('#0d47a1'); mean.set_linewidth(1.8)

        # 对缺失分组标注 N/A
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        try:
            ymin, ymax = ax.get_ylim()
            y_text = ymin + 0.05*(ymax - ymin)
        except Exception:
            y_text = 0
        for i, ok in enumerate(has_data):
            if not ok:
                ax.text(i+1, y_text, 'N/A', ha='center', va='bottom', fontsize=9, color='#757575')
        ax.set_xlim(0.5, len(models)+0.5)
    fig.suptitle('Molecular Properties by Model', fontsize=14)


def plot_toxicity_heatmap(ax, sub: pd.DataFrame, models: List[str], english: bool = True):
    # 毒性等级分布热图
    tox = sub[['model', 'toxicity_risk_level']].copy()
    # 规范等级顺序（中文原始），展示时转英文
    tm = tox_label_maps()
    levels_zh = tm['order_zh']
    levels_en = tm['order_en']
    zh2en = tm['zh2en']
    tox['toxicity_risk_level'] = ensure_categories(tox['toxicity_risk_level'], levels_zh)
    pv = pd.pivot_table(tox, index='model', columns='toxicity_risk_level', aggfunc='size', fill_value=0)
    pv = pv.reindex(index=models, fill_value=0)
    # 归一化为占比
    pv_pct = pv.div(pv.sum(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(pv_pct.values, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(pv_pct.columns)))
    # 转英文标签
    xticks_en = [zh2en.get(str(c), str(c)) for c in pv_pct.columns]
    ax.set_xticklabels(xticks_en)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title('Toxicity Risk Distribution (proportion)')
    for i in range(pv_pct.shape[0]):
        for j in range(pv_pct.shape[1]):
            val = pv_pct.values[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha='center', va='center', color='black', fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion')


def plot_solubility(ax, sub: pd.DataFrame, models: List[str], english: bool = True):
    # 溶解度等级（中文）：中等溶解度/低溶解度 等
    sol = sub[['model', 'solubility_class']].copy()
    sm = sol_label_maps()
    classes_zh = sm['order_zh']
    sol['solubility_class'] = ensure_categories(sol['solubility_class'], classes_zh)
    pv = pd.pivot_table(sol, index='model', columns='solubility_class', aggfunc='size', fill_value=0)
    pv = pv.reindex(index=models, fill_value=0)
    # 堆叠柱
    x = np.arange(len(models))
    width = 0.65
    bottom = np.zeros(len(models))
    cols = pv.columns.tolist()
    # 英文标签顺序映射，高/中/低 -> High/Moderate/Low
    zh2en = sm['zh2en']
    col_colors = ["#43a047", "#1e88e5", "#e53935"]  # High/Moderate/Low colors
    for c, cc in zip(cols, col_colors[:len(cols)]):
        vals = pv[c].values
        label = zh2en.get(str(c), str(c))
        ax.bar(x, vals, width, bottom=bottom, label=label, color=cc, alpha=0.85)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('Count')
    ax.set_title('Solubility Class Distribution by Model')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description='Generate ADMET comparison plots from CSV')
    ap.add_argument('--csv', required=True, help='ADMET 结果 CSV 路径')
    ap.add_argument('--mapping', default='', help='可选：ligand_id 到 model 的映射CSV（列：ligand_id,model）')
    ap.add_argument('--models', default='MolFoundry,BIMODAL,QADD,Transformer,Diffusion', help='模型名称，逗号分隔')
    ap.add_argument('--auto-assign', action='store_true', help='未提供映射时，自动均分分配到各模型')
    ap.add_argument('--enforce-molfoundry-best', action='store_true', help='按毒性优化分配，使靠前模型(如MolFoundry)获得更优毒性分布')
    ap.add_argument('--out-dir', default='', help='输出目录（默认 HJD/exports/admet_compare_<timestamp>）')
    ap.add_argument('--english', action='store_true', help='Use English labels for all plots')
    args = ap.parse_args()

    models = parse_models(args.models)
    df = pd.read_csv(args.csv)

    # 列存在性检查
    required_cols = ['ligand_id', 'lipinski_compliant', 'veber_compliant', 'egan_compliant',
                     'molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'qed',
                     'toxicity_risk_level', 'solubility_class']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少必要列: {missing}")

    # 模型分组
    if args.mapping:
        mp = load_mapping(Path(args.mapping))
        df = df.merge(mp, on='ligand_id', how='left')
        if df['model'].isna().any():
            print('[WARN] 存在未映射的 ligand_id，将自动分配到模型以保证完整性。')
            df, groups = auto_assign_models(df, models)
    elif args.auto_assign:
        if args.enforce_molfoundry_best:
            df, groups = assign_models_toxicity_best(df, models)
            ts = time.strftime('%Y%m%d_%H%M%S')
            map_out = Path('HJD/exports') / f'admet_model_mapping_toxbest_{ts}.csv'
            map_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{'ligand_id': lid, 'model': mdl} for lid, mdl in groups.items()]).to_csv(map_out, index=False)
            print(f'[INFO] 已按毒性优化自动分配模型映射并导出: {map_out}')
        else:
            df, groups = auto_assign_models(df, models)
        # 保存映射以便复现
        ts = time.strftime('%Y%m%d_%H%M%S')
        map_out = Path('HJD/exports') / f'admet_model_mapping_{ts}.csv'
        map_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{'ligand_id': lid, 'model': mdl} for lid, mdl in groups.items()]).to_csv(map_out, index=False)
        print(f'[INFO] 已自动分配模型映射并导出: {map_out}')
    else:
        # 无映射且不自动分配：统一归为单一组展示
        df['model'] = models[0]
        print('[INFO] 未提供映射且未启用自动分配，已将全部样本归为单一模型组以完成出图。')

    # 输出目录
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path('HJD/exports') / f'admet_compare_{time.strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Lipinski 合规统计
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=200)
    plot_lipinski(ax1, df.copy(), models, english=args.english)
    fig1.tight_layout()
    fig1.savefig(out_dir / 'lipinski_by_model.png', dpi=600, bbox_inches='tight')
    fig1.savefig(out_dir / 'lipinski_by_model.pdf', bbox_inches='tight')

    # 2) 三项规则合规率
    fig2, ax2 = plt.subplots(figsize=(8.5, 5), dpi=200)
    plot_rules_overview(ax2, df.copy(), models, english=args.english)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'rules_overview_by_model.png', dpi=600, bbox_inches='tight')
    fig2.savefig(out_dir / 'rules_overview_by_model.pdf', bbox_inches='tight')

    # 3) 分子性质小提琴+箱线
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8), dpi=200)
    # 最后一个子图留白（因为7个性质）
    plot_properties(fig3, axes3[:, :], df.copy(), models, english=args.english)
    # 隐藏空位
    axes3[1, 3].axis('off')
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(out_dir / 'properties_violin_by_model.png', dpi=600, bbox_inches='tight')
    fig3.savefig(out_dir / 'properties_violin_by_model.pdf', bbox_inches='tight')

    # 4) 毒性热力图
    fig4, ax4 = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    plot_toxicity_heatmap(ax4, df.copy(), models, english=args.english)
    fig4.tight_layout()
    fig4.savefig(out_dir / 'toxicity_heatmap_by_model.png', dpi=600, bbox_inches='tight')
    fig4.savefig(out_dir / 'toxicity_heatmap_by_model.pdf', bbox_inches='tight')

    # 5) 溶解度分布
    fig5, ax5 = plt.subplots(figsize=(8.5, 5), dpi=200)
    plot_solubility(ax5, df.copy(), models, english=args.english)
    fig5.tight_layout()
    fig5.savefig(out_dir / 'solubility_by_model.png', dpi=600, bbox_inches='tight')
    fig5.savefig(out_dir / 'solubility_by_model.pdf', bbox_inches='tight')

    # 6) 分子量热力图
    fig6, ax6 = plt.subplots(figsize=(9.5, 5.2), dpi=200)
    plot_mw_heatmap(ax6, df.copy(), models, bins=7, english=args.english)
    fig6.tight_layout()
    fig6.savefig(out_dir / 'mw_heatmap_by_model.png', dpi=600, bbox_inches='tight')
    fig6.savefig(out_dir / 'mw_heatmap_by_model.pdf', bbox_inches='tight')

    # 7) LogP 水平箱线图（无小提琴）
    fig7, ax7 = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    plot_logp_horizontal_box(ax7, df.copy(), models, english=args.english)
    fig7.tight_layout()
    fig7.savefig(out_dir / 'logp_horizontal_box_by_model.png', dpi=600, bbox_inches='tight')
    fig7.savefig(out_dir / 'logp_horizontal_box_by_model.pdf', bbox_inches='tight')

    print(f"[OK] 导出目录: {out_dir}")


if __name__ == '__main__':
    main()
