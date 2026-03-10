#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练与验证结果可视化脚本
- 训练曲线（loss、metrics随epoch变化）
- 验证曲线对比
- ROC/PR曲线
- 混淆矩阵
- 阈值-指标曲线（F1、Precision、Recall）
- 多折/多种子对比
"""

import os
import json
import argparse
import re
from pathlib import Path
from glob import glob
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

try:
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("Warning: scikit-learn not installed, some plots will be skipped")


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """
    解析training.log，提取每个epoch的train_loss、val_loss等指标
    支持两种格式：
    1. Epoch 10 完成 (耗时: 123.45s) - train_loss: 0.1234 - val_loss: 0.2345
    2. Epoch  10: Train Loss: 0.1234, Val Loss: 0.2345, accuracy: 0.8765, LR: 1.23e-05
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': [],
        'val_ap': []
    }
    
    if not os.path.exists(log_path):
        return history
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 格式1: "Epoch X 完成" (ProgressLogger)
            match1 = re.search(r'Epoch (\d+) 完成.*?-\s+(.*)', line)
            # 格式2: "Epoch  X:" (trainer._log_epoch_results)
            match2 = re.search(r'Epoch\s+(\d+):\s+Train Loss:\s+([\d.]+)(?:,\s+Val Loss:\s+([\d.]+))?(?:,\s+(.*))?', line)
            
            if match1:
                epoch = int(match1.group(1))
                metrics_str = match1.group(2)
                metrics = {}
                for item in metrics_str.split(' - '):
                    if ':' in item:
                        k, v = item.split(':', 1)
                        k = k.strip()
                        try:
                            metrics[k] = float(v.strip())
                        except Exception:
                            pass
                
                history['epoch'].append(epoch)
                for key in ['train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'val_auc', 'val_ap']:
                    history[key].append(metrics.get(key, np.nan))
            
            elif match2:
                epoch = int(match2.group(1))
                train_loss = float(match2.group(2))
                val_loss_str = match2.group(3)
                other_metrics_str = match2.group(4)
                
                metrics = {'train_loss': train_loss}
                if val_loss_str:
                    metrics['val_loss'] = float(val_loss_str)
                
                # 解析其他指标
                if other_metrics_str:
                    for item in other_metrics_str.split(','):
                        item = item.strip()
                        if ':' in item:
                            k, v = item.split(':', 1)
                            k = k.strip()
                            v = v.strip()
                            # 跳过LR
                            if k.lower() == 'lr':
                                continue
                            try:
                                # 对于验证指标，如果已有val_前缀则保留，否则添加
                                if not k.startswith('val_'):
                                    k = 'val_' + k
                                metrics[k] = float(v)
                            except Exception:
                                pass
                
                history['epoch'].append(epoch)
                for key in ['train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'val_auc', 'val_ap']:
                    history[key].append(metrics.get(key, np.nan))
    
    return history


def plot_training_curves(history: Dict[str, List[float]], out_path: str, title: str = "Training Curves"):
    """绘制训练与验证曲线（loss + metrics）"""
    epochs = history.get('epoch', [])
    if not epochs:
        print(f"Skip {out_path}: no epoch data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    if history.get('train_loss'):
        ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3, alpha=0.7)
    if history.get('val_loss'):
        ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    if history.get('val_accuracy'):
        ax.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='o', markersize=3, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1
    ax = axes[1, 0]
    if history.get('val_f1'):
        ax.plot(epochs, history['val_f1'], label='Val F1', marker='o', markersize=3, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC/AP
    ax = axes[1, 1]
    if history.get('val_auc'):
        ax.plot(epochs, history['val_auc'], label='Val AUC', marker='o', markersize=3, color='purple')
    if history.get('val_ap'):
        ax.plot(epochs, history['val_ap'], label='Val AP', marker='s', markersize=3, color='brown')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation AUC/AP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_roc_pr(eval_data: Dict[str, Any], out_dir: str, prefix: str = "test"):
    """绘制ROC和PR曲线"""
    if not HAS_SK:
        return
    
    preds = eval_data.get('predictions', {})
    targets = eval_data.get('targets', {})
    
    prob = preds.get('prob')
    y_true = targets.get('y') or targets.get('label')
    
    if prob is None or y_true is None:
        print(f"Skip ROC/PR: missing predictions or targets")
        return
    
    prob = np.array(prob).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    
    # 过滤NaN
    mask = np.isfinite(prob) & np.isfinite(y_true)
    prob = prob[mask]
    y_true = y_true[mask]
    
    if len(np.unique(y_true)) < 2:
        print(f"Skip ROC/PR: only one class in targets")
        return
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, f"{prefix}_roc.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")
    
    # PR
    prec, rec, _ = precision_recall_curve(y_true, prob)
    pr_auc = auc(rec, prec)
    
    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, label=f'AP = {pr_auc:.3f}', linewidth=2, color='darkorange')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"{prefix}_pr.png")
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pr_path}")


def plot_confusion_matrix(eval_data: Dict[str, Any], out_dir: str, prefix: str = "test"):
    """绘制混淆矩阵"""
    if not HAS_SK:
        return
    
    preds = eval_data.get('predictions', {})
    targets = eval_data.get('targets', {})
    
    prob = preds.get('prob')
    y_true = targets.get('y') or targets.get('label')
    
    if prob is None or y_true is None:
        return
    
    prob = np.array(prob).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    
    # 使用校准阈值（若有）
    threshold = eval_data.get('metrics_calibrated', {}).get('threshold', 0.5)
    y_pred = (prob > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix (threshold={threshold:.3f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")


def plot_threshold_metrics(eval_data: Dict[str, Any], out_dir: str, prefix: str = "test"):
    """绘制阈值-指标曲线（F1、Precision、Recall）"""
    if not HAS_SK:
        return
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    preds = eval_data.get('predictions', {})
    targets = eval_data.get('targets', {})
    
    prob = preds.get('prob')
    y_true = targets.get('y') or targets.get('label')
    
    if prob is None or y_true is None:
        return
    
    prob = np.array(prob).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred = (prob > t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1s, label='F1', linewidth=2, color='blue')
    plt.plot(thresholds, precisions, label='Precision', linewidth=2, color='green')
    plt.plot(thresholds, recalls, label='Recall', linewidth=2, color='red')
    
    # 标记最佳F1阈值
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    plt.axvline(best_threshold, color='purple', linestyle='--', alpha=0.7, 
                label=f'Best F1={best_f1:.3f} @ {best_threshold:.3f}')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold vs Metrics', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    th_path = os.path.join(out_dir, f"{prefix}_threshold_metrics.png")
    plt.savefig(th_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {th_path}")


def plot_kfold_comparison(summary_files: List[str], out_dir: str):
    """对比多个K折汇总结果（多种子/多模型）"""
    if not summary_files:
        return
    
    data = []
    labels = []
    for f in summary_files:
        try:
            with open(f, 'r') as fp:
                obj = json.load(fp)
            # 提取文件名作为标签
            name = Path(f).stem  # kfold_summary_seed42_k5
            labels.append(name)
            data.append(obj)
        except Exception:
            pass
    
    if not data:
        return
    
    # 提取关键指标：accuracy, precision, recall, f1, auc, ap
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
    means = {k: [] for k in metrics_keys}
    stds = {k: [] for k in metrics_keys}
    
    for obj in data:
        for k in metrics_keys:
            if k in obj:
                means[k].append(obj[k].get('mean', 0))
                stds[k].append(obj[k].get('std', 0))
            else:
                means[k].append(0)
                stds[k].append(0)
    
    # 绘制条形图
    x = np.arange(len(labels))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, k in enumerate(metrics_keys):
        ax.bar(x + i * width, means[k], width, yerr=stds[k], 
               label=k.upper(), capsize=3)
    
    ax.set_xlabel('Experiments', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('K-Fold Cross-Validation Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics_keys) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    comp_path = os.path.join(out_dir, "kfold_comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化训练与验证结果')
    parser.add_argument('--log_dir', type=str, default='logs/dual_gpu_run',
                        help='训练日志根目录（包含 run_seed* 子目录）')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='评估JSON目录')
    parser.add_argument('--out_dir', type=str, default='results/visualizations',
                        help='输出可视化图片目录')
    parser.add_argument('--seed', type=int, default=None,
                        help='指定种子（留空则处理所有）')
    parser.add_argument('--fold', type=int, default=None,
                        help='指定折（留空则处理所有）')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    
    # 1. 找到所有 run_seed* 目录
    run_dirs = sorted(log_dir.glob('run_seed*'))
    if not run_dirs:
        print(f"No run_seed* directories found in {log_dir}")
        return
    
    for run_dir in run_dirs:
        seed_match = re.search(r'seed(\d+)', run_dir.name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        if args.seed is not None and seed != args.seed:
            continue
        
        print(f"\n=== Processing seed={seed} ===")
        
        # 检查是否有折目录
        fold_dirs = sorted(run_dir.glob('fold*of*'))
        if fold_dirs:
            # K折模式
            for fold_dir in fold_dirs:
                fold_match = re.search(r'fold(\d+)of(\d+)', fold_dir.name)
                if not fold_match:
                    continue
                fold_idx = int(fold_match.group(1))
                k = int(fold_match.group(2))
                if args.fold is not None and fold_idx != args.fold:
                    continue
                
                print(f"  Fold {fold_idx}/{k}")
                log_path = fold_dir / 'training.log'
                if log_path.exists():
                    history = parse_training_log(str(log_path))
                    plot_path = out_dir / f"training_curves_seed{seed}_fold{fold_idx}.png"
                    plot_training_curves(history, str(plot_path), 
                                         title=f"Training Curves (seed={seed}, fold={fold_idx}/{k})")
                
                # 读取对应折的评估JSON
                eval_json = results_dir / f"test_eval_seed{seed}_fold{fold_idx}of{k}.json"
                if eval_json.exists():
                    with open(eval_json, 'r') as f:
                        eval_data = json.load(f)
                    fold_out = out_dir / f"seed{seed}_fold{fold_idx}"
                    fold_out.mkdir(parents=True, exist_ok=True)
                    plot_roc_pr(eval_data, str(fold_out), prefix="test")
                    plot_confusion_matrix(eval_data, str(fold_out), prefix="test")
                    plot_threshold_metrics(eval_data, str(fold_out), prefix="test")
        else:
            # 单次运行模式
            print(f"  Single run")
            log_path = run_dir / 'training.log'
            if log_path.exists():
                history = parse_training_log(str(log_path))
                plot_path = out_dir / f"training_curves_seed{seed}.png"
                plot_training_curves(history, str(plot_path), 
                                     title=f"Training Curves (seed={seed})")
            
            # 读取评估JSON
            eval_json = results_dir / 'test_eval.json'
            if eval_json.exists():
                with open(eval_json, 'r') as f:
                    eval_data = json.load(f)
                seed_out = out_dir / f"seed{seed}"
                seed_out.mkdir(parents=True, exist_ok=True)
                plot_roc_pr(eval_data, str(seed_out), prefix="test")
                plot_confusion_matrix(eval_data, str(seed_out), prefix="test")
                plot_threshold_metrics(eval_data, str(seed_out), prefix="test")
    
    # 2. K折汇总对比
    summary_files = sorted(results_dir.glob('kfold_summary_*.json'))
    if summary_files:
        print(f"\n=== Plotting K-Fold Comparison ===")
        plot_kfold_comparison([str(f) for f in summary_files], str(out_dir))
    
    print(f"\nAll visualizations saved to: {out_dir}")


if __name__ == '__main__':
    main()
