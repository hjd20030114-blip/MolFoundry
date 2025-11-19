#!/bin/bash
#############################################
# 快速可视化训练结果脚本
# 一键生成所有图表
#############################################

set -e

PROJECT_ROOT="/home/hjd/PRRSV"
cd "${PROJECT_ROOT}/HJD" || exit 1

echo "=========================================="
echo "开始可视化训练结果"
echo "=========================================="

# 1. 可视化训练曲线和测试集评估
echo ""
echo "[1/2] 生成训练曲线、ROC/PR、混淆矩阵..."
python tools/visualize_training.py \
  --log_dir logs/dual_gpu_run \
  --results_dir results \
  --out_dir results/visualizations

# 2. 聚合统计与合并ROC/PR
echo ""
echo "[2/2] 聚合多折结果并生成汇总图表..."
python tools/aggregate_results.py \
  --results_dir results \
  --out_dir results/summary \
  --plots

echo ""
echo "=========================================="
echo "可视化完成！"
echo "=========================================="
echo ""
echo "输出文件位置："
echo "  - 训练曲线: results/visualizations/training_curves_*.png"
echo "  - 测试评估: results/visualizations/seed*/test_*.png"
echo "  - K折对比: results/visualizations/kfold_comparison.png"
echo "  - 汇总表格: results/summary/metrics_table.csv"
echo "  - 汇总统计: results/summary/aggregate_stats.json"
echo "  - 合并ROC:  results/summary/roc_curve.png"
echo "  - 合并PR:   results/summary/pr_curve.png"
echo ""
echo "挑战杯材料推荐图表："
echo "  1. 训练过程: training_curves_seed42_fold1.png"
echo "  2. 性能对比: kfold_comparison.png"
echo "  3. ROC曲线:  summary/roc_curve.png"
echo "  4. PR曲线:   summary/pr_curve.png"
echo "  5. 混淆矩阵: seed42/test_confusion_matrix.png"
echo "  6. 阈值选择: seed42/test_threshold_metrics.png"
echo ""
