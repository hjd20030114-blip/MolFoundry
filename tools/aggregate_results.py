#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合多种子/多折测试评估结果脚本
- 搜索 results/ 下的 test_eval_seed*.json 和 test_eval_seed*_fold*.json
- 优先使用 metrics_calibrated（若无则用 metrics）
- 生成：
  - results/summary/metrics_table.csv（逐文件记录 + 关键指标列）
  - results/summary/aggregate_stats.json（各指标mean/std）
  - 可选：ROC/PR 曲线（合并所有样本）
依赖：numpy、scikit-learn（可选：matplotlib）
"""

import os
import json
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    HAS_SK = True
except Exception:
    HAS_SK = False

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def _collect_json_files(results_dir: str, pattern: str = "test_eval_seed*.json") -> List[str]:
    paths = sorted(glob(os.path.join(results_dir, pattern)))
    # 同时包含折别文件
    paths += sorted(glob(os.path.join(results_dir, "test_eval_seed*_*of*.json")))
    # 去重
    dedup = []
    seen = set()
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _metrics_from_eval(obj: Dict[str, Any]) -> Dict[str, float]:
    m = obj.get('metrics_calibrated') or obj.get('metrics') or {}
    # 仅保留数值项
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}


def _append_table_row(rows: List[List[Any]], path: str, metrics: Dict[str, float]):
    name = os.path.basename(path)
    row = [name]
    # 固定列的顺序
    for k in [
        'threshold', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'ap'
    ]:
        row.append(metrics.get(k, None))
    rows.append(row)


def _concat_probs_targets(objs: List[Dict[str, Any]]):
    all_prob = []
    all_y = []
    for o in objs:
        preds = (o.get('predictions') or {})
        tars = (o.get('targets') or {})
        prob = preds.get('prob')
        y = tars.get('y') or tars.get('label')
        if prob is not None and y is not None:
            try:
                p = np.array(prob).reshape(-1)
                yt = np.array(y).reshape(-1)
                # 过滤NaN
                mask = np.isfinite(p) & np.isfinite(yt)
                p = p[mask]
                yt = yt[mask]
                if p.size and yt.size:
                    all_prob.append(p)
                    all_y.append(yt)
            except Exception:
                pass
    if all_prob and all_y:
        return np.concatenate(all_prob, axis=0), np.concatenate(all_y, axis=0)
    return None, None


def main():
    parser = argparse.ArgumentParser(description='聚合多种子/多折评估结果')
    parser.add_argument('--results_dir', type=str, default='results', help='结果目录（包含 test_eval_*.json）')
    parser.add_argument('--out_dir', type=str, default='results/summary', help='输出目录')
    parser.add_argument('--pattern', type=str, default='test_eval_seed*.json', help='文件匹配模式（相对results_dir）')
    parser.add_argument('--plots', action='store_true', help='是否绘制ROC/PR曲线（需sklearn+matplotlib）')
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_json_files(results_dir, args.pattern)
    if not files:
        print('No result json files found.')
        return

    # 表格汇总
    table_rows = []
    metrics_list = []
    eval_objs = []
    for p in files:
        obj = _read_json(p)
        eval_objs.append(obj)
        m = _metrics_from_eval(obj)
        _append_table_row(table_rows, p, m)
        if m:
            metrics_list.append(m)

    # 写入表格CSV
    header = ['file', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
    csv_path = out_dir / 'metrics_table.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in table_rows:
            def _fmt(x):
                if x is None:
                    return ''
                try:
                    return f"{float(x):.6f}"
                except Exception:
                    return str(x)
            f.write(','.join(_fmt(x) for x in row) + '\n')
    print('Wrote table:', csv_path)

    # 统计聚合 mean/std
    agg: Dict[str, Dict[str, float]] = {}
    if metrics_list:
        keys = set()
        for m in metrics_list:
            keys.update([k for k, v in m.items() if isinstance(v, (int, float))])
        for k in sorted(keys):
            vals = [m[k] for m in metrics_list if k in m]
            if vals:
                agg[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals))
                }
    agg_path = out_dir / 'aggregate_stats.json'
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print('Wrote aggregate stats:', agg_path)

    # 可选：绘图（合并所有样本）
    if args.plots and HAS_SK and HAS_PLT:
        probs, ys = _concat_probs_targets(eval_objs)
        if probs is not None and ys is not None and len(np.unique(ys)) > 1:
            # ROC
            fpr, tpr, _ = roc_curve(ys, probs)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(4,4))
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0,1], [0,1], 'k--', alpha=0.5)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC (All Runs Combined)')
            plt.legend(loc='lower right')
            roc_path = out_dir / 'roc_curve.png'
            plt.tight_layout(); plt.savefig(roc_path, dpi=150); plt.close()
            print('Wrote ROC:', roc_path)
            # PR
            prec, rec, _ = precision_recall_curve(ys, probs)
            ap = average_precision_score(ys, probs)
            plt.figure(figsize=(4,4))
            plt.plot(rec, prec, label=f"AP={ap:.3f}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR (All Runs Combined)')
            plt.legend(loc='lower left')
            pr_path = out_dir / 'pr_curve.png'
            plt.tight_layout(); plt.savefig(pr_path, dpi=150); plt.close()
            print('Wrote PR:', pr_path)
        else:
            print('Skip plots: insufficient class variation or missing predictions.')
    elif args.plots:
        print('Skip plots: sklearn/matplotlib not available.')

if __name__ == '__main__':
    main()
