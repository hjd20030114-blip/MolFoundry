#!/usr/bin/env python3
"""
PRRSV抑制剂设计项目主训练脚本
基于项目现有的trainer.py和模型架构
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import random
import torch
import torch.nn as nn
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from pathlib import Path
from torch.utils.data import DataLoader
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')   # 关闭所有 RDKit 控制台输出
    # 如只想保留错误: RDLogger.DisableLog('rdApp.warning')
except Exception:
    pass
# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def setup_logging(log_dir):
    """设置日志（可重复调用，自动清理旧handlers）。"""
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    # 关闭并移除旧的handlers，避免重复日志
    if root.handlers:
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f"{log_dir}/training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_loss_function(task_name):
    """创建损失函数"""
    if task_name == 'pl_classification':
        return nn.BCEWithLogitsLoss()
    elif task_name == 'binding_affinity':
        return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()

def main():
    parser = argparse.ArgumentParser(description='PRRSV训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, default='equivariant_gnn', 
                       choices=['pocket_ligand_transformer', 'equivariant_gnn'],
                       help='模型类型')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（用于划分与初始化）')
    parser.add_argument('--splits_file', type=str, default='', help='数据划分文件路径（为空则按seed自动生成）')
    parser.add_argument('--kfold', type=int, default=1, help='K-Fold 折数（>1 启用K折在train+val上交叉验证，test固定为初始留出集）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志：按 seed 组织到子目录
    base_log_dir = config['logging']['log_dir']
    run_root_dir = os.path.join(base_log_dir, f"run_seed{args.seed}")
    setup_logging(run_root_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("开始PRRSV抑制剂设计训练")
    logger.info("=" * 50)
    logger.info(f"模型类型: {args.model}")
    logger.info(f"配置: {config}")
    
    try:
        # 导入项目模块
        from deep_learning.training import Trainer, TrainingConfig
        from deep_learning.data.pl_pair_dataset import PLPairDataset, _find_pl_items, _split_ids
        from deep_learning.models import create_model, ModelConfig
        from deep_learning.training.callbacks import ModelCheckpoint
        
        logger.info("✓ 成功导入所有训练模块")
        
        # 检查数据目录
        pl_data_dir = config['data']['pl_data_dir']
        if not os.path.exists(pl_data_dir):
            logger.error(f"✗ P-L数据目录不存在: {pl_data_dir}")
            return
        
        logger.info(f"✓ P-L数据目录: {pl_data_dir}")
        
        # 设置全局随机种子
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # 创建数据集
        logger.info("创建数据集...")

        # 若启用K折：先在ID层面做一次固定留出（seed控制）的 train/val/test，随后在 train+val 上做 K 折划分
        if int(getattr(args, 'kfold', 1)) > 1:
            k = int(args.kfold)
            logger.info(f"启用K-Fold: k={k}, seed={args.seed}")
            items = _find_pl_items(Path(pl_data_dir))
            base_ids = sorted({it['id'] for it in items})
            if len(base_ids) < 3:
                logger.warning("样本量不足以进行K折，回退为单折训练")
            else:
                base_splits = _split_ids(base_ids, seed=args.seed)
                dev_ids = list(base_splits.get('train', [])) + list(base_splits.get('val', []))
                test_ids = list(base_splits.get('test', []))
                rng_local = random.Random(args.seed)
                rng_local.shuffle(dev_ids)
                k = min(k, max(2, len(dev_ids)))
                fold_sizes = [len(dev_ids) // k] * k
                for i in range(len(dev_ids) % k):
                    fold_sizes[i] += 1
                start_idx = 0
                results_dir = Path('results')
                results_dir.mkdir(parents=True, exist_ok=True)
                fold_reports = []
                for fi, fs in enumerate(fold_sizes):
                    val_ids = dev_ids[start_idx:start_idx + fs]
                    train_ids = dev_ids[:start_idx] + dev_ids[start_idx + fs:]
                    start_idx += fs
                    fold_idx = fi + 1
                    # 当前折日志目录
                    run_fold_dir = os.path.join(run_root_dir, f"fold{fold_idx}of{k}")
                    setup_logging(run_fold_dir)
                    # 写入当前折的划分文件
                    fold_splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
                    fold_splits_path = results_dir / f'pl_splits_seed{args.seed}_fold{fold_idx}of{k}.json'
                    with open(fold_splits_path, 'w') as f:
                        json.dump(fold_splits, f, indent=2)
                    # 每折独立的口袋统计文件
                    feature_stats_file_fold = str(results_dir / f'pl_feature_stats_seed{args.seed}_fold{fold_idx}of{k}.json')

                    # 实例化数据集与数据加载器
                    train_dataset = PLPairDataset(
                        root_dir=pl_data_dir,
                        split='train',
                        splits_file=str(fold_splits_path),
                        feature_stats_file=feature_stats_file_fold,
                        normalize_pocket=True,
                        min_fp_bits_on=5,
                        negative_ratio=1,
                        seed=args.seed
                    )
                    val_dataset = PLPairDataset(
                        root_dir=pl_data_dir,
                        split='val',
                        splits_file=str(fold_splits_path),
                        feature_stats_file=feature_stats_file_fold,
                        normalize_pocket=True,
                        min_fp_bits_on=5,
                        negative_ratio=1,
                        seed=args.seed
                    )
                    test_dataset = PLPairDataset(
                        root_dir=pl_data_dir,
                        split='test',
                        splits_file=str(fold_splits_path),
                        feature_stats_file=feature_stats_file_fold,
                        normalize_pocket=True,
                        min_fp_bits_on=5,
                        negative_ratio=1,
                        seed=args.seed
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=int(config['data']['batch_size']),
                        shuffle=True,
                        num_workers=int(config['data']['num_workers']),
                        pin_memory=True,
                        drop_last=True
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=int(config['data']['batch_size']),
                        shuffle=False,
                        num_workers=int(config['data']['num_workers']),
                        pin_memory=True
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=int(config['data']['batch_size']),
                        shuffle=False,
                        num_workers=int(config['data']['num_workers']),
                        pin_memory=True
                    )

                    logger.info(f"[Fold {fold_idx}/{k}] 训练/验证/测试大小: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

                    # 模型与训练器
                    model_config = ModelConfig(
                        model_type=args.model,
                        hidden_dim=config.get('model', {}).get('hidden_dim', 512),
                        num_layers=config.get('model', {}).get('num_layers', 3),
                        dropout=config.get('model', {}).get('dropout', 0.2),
                        input_dim=2060
                    )
                    model = create_model(args.model, model_config)
                    logger.info(f"[Fold {fold_idx}/{k}] 模型参数: {model.get_num_parameters():,}")

                    training_config = TrainingConfig(
                        num_epochs=int(config['training']['epochs']),
                        batch_size=int(config['data']['batch_size']),
                        learning_rate=float(config['training']['learning_rate']),
                        weight_decay=float(config['training']['weight_decay']),
                        gradient_clip_norm=float(config['training']['gradient_clip']),
                        optimizer_type=str(config['training'].get('optimizer_type', 'adamw')),
                        scheduler_type=str(config['training'].get('scheduler_type', 'cosine')),
                        mixed_precision=bool(config['training'].get('mixed_precision', True)),
                        accumulation_steps=int(config['training'].get('accumulation_steps', 1)),
                        early_stopping_patience=int(config.get('validation', {}).get('early_stopping_patience', 20)),
                        use_early_stopping=False,
                        validation_frequency=1,
                        save_frequency=int(config.get('checkpoints', {}).get('save_frequency', 10)),
                        warmup_epochs=int(config['training'].get('warmup_epochs', 5)),
                        eta_min=float(config['training'].get('eta_min', 0.0))
                    )

                    callbacks = [
                        ModelCheckpoint(
                            filepath=f"{run_fold_dir}/checkpoints/ckpt_epoch_{{epoch}}.pth",
                            monitor='val_loss',
                            save_best_only=True,
                            save_freq=training_config.save_frequency,
                            best_save_freq=int(config.get('checkpoints', {}).get('best_save_freq', 0)),
                            final_best_path=f"{run_fold_dir}/best_model.pth"
                        )
                    ]

                    trainer = Trainer(
                        model=model,
                        config=training_config,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        callbacks=callbacks
                    )
                    if args.model == 'pocket_ligand_transformer':
                        trainer.criterion = create_loss_function('pl_classification')
                    else:
                        trainer.criterion = create_loss_function('binding_affinity')

                    logger.info(f"[Fold {fold_idx}/{k}] 开始训练...")
                    trainer.train()

                    # 加载最佳验证模型
                    best_model_path = f"{run_fold_dir}/best_model.pth"
                    if os.path.exists(best_model_path):
                        try:
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            best_ckpt = torch.load(best_model_path, map_location=device)
                            model.load_state_dict(best_ckpt['model_state_dict'])
                            logger.info(f"[Fold {fold_idx}/{k}] 已加载最佳验证权重")
                        except Exception as e:
                            logger.warning(f"[Fold {fold_idx}/{k}] 加载最佳模型失败: {e}")

                    # 阈值校准（验证集）
                    best_threshold = 0.5
                    try:
                        val_eval = trainer.evaluate(val_loader, return_predictions=True)
                        vpreds = val_eval.get('predictions', {})
                        vtargets = val_eval.get('targets', {})
                        prob = vpreds.get('prob')
                        if prob is None and 'logit' in vpreds:
                            prob = 1.0 / (1.0 + np.exp(-np.array(vpreds['logit']).reshape(-1)))
                        if prob is not None:
                            prob = np.array(prob).reshape(-1)
                            y_true = None
                            for key in ('y', 'label'):
                                if key in vtargets:
                                    y_true = np.array(vtargets[key]).reshape(-1)
                                    break
                            if y_true is not None and y_true.size > 0:
                                thresholds = np.linspace(0.05, 0.95, 19)
                                best_f1 = -1.0
                                for t in thresholds:
                                    y_pred = (prob > t).astype(int)
                                    f1 = f1_score(y_true, y_pred, zero_division=0)
                                    if f1 > best_f1:
                                        best_f1 = f1
                                        best_threshold = float(t)
                                logger.info(f"[Fold {fold_idx}/{k}] 验证集阈值校准: best_threshold={best_threshold:.3f}, best_f1={best_f1:.4f}")
                    except Exception as e:
                        logger.warning(f"[Fold {fold_idx}/{k}] 阈值校准失败，使用默认0.5: {e}")

                    # 测试评估与保存
                    test_eval = trainer.evaluate(test_loader, return_predictions=True)
                    # 基于校准阈值的测试集指标
                    try:
                        tpreds = test_eval.get('predictions', {})
                        ttargets = test_eval.get('targets', {})
                        tprob = tpreds.get('prob')
                        if tprob is None and 'logit' in tpreds:
                            tprob = 1.0 / (1.0 + np.exp(-np.array(tpreds['logit']).reshape(-1)))
                        if tprob is not None:
                            tprob = np.array(tprob).reshape(-1)
                            y_true_t = None
                            for key in ('y', 'label'):
                                if key in ttargets:
                                    y_true_t = np.array(ttargets[key]).reshape(-1)
                                    break
                            if y_true_t is not None and y_true_t.size > 0:
                                y_pred_t = (tprob > best_threshold).astype(int)
                                calib_metrics = {
                                    'threshold': float(best_threshold),
                                    'accuracy': float(accuracy_score(y_true_t, y_pred_t)),
                                    'precision': float(precision_score(y_true_t, y_pred_t, zero_division=0)),
                                    'recall': float(recall_score(y_true_t, y_pred_t, zero_division=0)),
                                    'f1': float(f1_score(y_true_t, y_pred_t, zero_division=0))
                                }
                                if len(np.unique(y_true_t)) > 1:
                                    try:
                                        calib_metrics['auc'] = float(roc_auc_score(y_true_t, tprob))
                                        calib_metrics['ap'] = float(average_precision_score(y_true_t, tprob))
                                    except Exception:
                                        pass
                                test_eval['metrics_calibrated'] = calib_metrics
                    except Exception as e:
                        logger.warning(f"[Fold {fold_idx}/{k}] 测试集校准指标计算失败: {e}")

                    # 保存每折评估
                    def _to_jsonable(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, dict):
                            return {k: _to_jsonable(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return [_to_jsonable(v) for v in obj]
                        try:
                            if hasattr(obj, 'item'):
                                return obj.item()
                        except Exception:
                            pass
                        return obj
                    fold_json = _to_jsonable(test_eval)
                    fold_json_path = results_dir / f'test_eval_seed{args.seed}_fold{fold_idx}of{k}.json'
                    with open(fold_json_path, 'w') as f:
                        json.dump(fold_json, f, ensure_ascii=False, indent=2)
                    logger.info(f"[Fold {fold_idx}/{k}] 测试评估JSON: {fold_json_path}")

                    # 汇总指标（优先使用校准后）
                    metrics_to_use = test_eval.get('metrics_calibrated') or test_eval.get('metrics') or {}
                    fold_reports.append(metrics_to_use)

                # 计算K折汇总
                def _agg(keys, reports):
                    out = {}
                    for k_ in keys:
                        vals = [r[k_] for r in reports if (k_ in r and isinstance(r[k_], (int, float)))]
                        if vals:
                            out[k_] = {
                                'mean': float(np.mean(vals)),
                                'std': float(np.std(vals))
                            }
                    return out
                all_keys = set()
                for r in fold_reports:
                    all_keys.update([k for k, v in r.items() if isinstance(v, (int, float))])
                summary = _agg(sorted(all_keys), fold_reports)
                summary_path = results_dir / f'kfold_summary_seed{args.seed}_k{k}.json'
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                logger.info(f"K折汇总已保存: {summary_path}")

                # 将最后一折的评估另存为默认名，便于旧脚本兼容
                try:
                    latest_path = results_dir / 'test_eval.json'
                    with open(latest_path, 'w') as f:
                        json.dump(fold_json, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

                return

        # 使用按seed命名的划分与统计文件，便于多次实验共存
        splits_file = args.splits_file if args.splits_file else str(Path('results') / f'pl_splits_seed{args.seed}.json')
        # 口袋特征标准化统计文件（随seed区分，避免不同划分互相覆盖）
        feature_stats_file = str(Path('results') / f'pl_feature_stats_seed{args.seed}.json')
        
        train_dataset = PLPairDataset(
            root_dir=pl_data_dir,
            split='train',
            splits_file=splits_file,
            feature_stats_file=feature_stats_file,
            normalize_pocket=True,
            min_fp_bits_on=5,
            negative_ratio=1,
            seed=args.seed
        )
        
        val_dataset = PLPairDataset(
            root_dir=pl_data_dir,
            split='val',
            splits_file=splits_file,
            feature_stats_file=feature_stats_file,
            normalize_pocket=True,
            min_fp_bits_on=5,
            negative_ratio=1,
            seed=args.seed
        )
        
        test_dataset = PLPairDataset(
            root_dir=pl_data_dir,
            split='test',
            splits_file=splits_file,
            feature_stats_file=feature_stats_file,
            normalize_pocket=True,
            min_fp_bits_on=5,
            negative_ratio=1,
            seed=args.seed
        )
        
        logger.info(f"✓ 训练数据集大小: {len(train_dataset)}")
        logger.info(f"✓ 验证数据集大小: {len(val_dataset)}")
        logger.info(f"✓ 测试数据集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config['data']['batch_size']),
            shuffle=True,
            num_workers=int(config['data']['num_workers']),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config['data']['batch_size']),
            shuffle=False,
            num_workers=int(config['data']['num_workers']),
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(config['data']['batch_size']),
            shuffle=False,
            num_workers=int(config['data']['num_workers']),
            pin_memory=True
        )
        
        logger.info(f"✓ 训练批次数: {len(train_loader)}")
        logger.info(f"✓ 验证批次数: {len(val_loader)}")
        
        # 创建模型配置
        model_config = ModelConfig(
            model_type=args.model,
            hidden_dim=config.get('model', {}).get('hidden_dim', 512),
            num_layers=config.get('model', {}).get('num_layers', 3),
            dropout=config.get('model', {}).get('dropout', 0.2),
            input_dim=2060  # 配体指纹(2048) + 口袋特征(12)
        )
        
        # 创建模型
        logger.info(f"创建模型: {args.model}")
        model = create_model(args.model, model_config)
        
        logger.info(f"✓ 模型参数数量: {model.get_num_parameters():,}")
        
        # 创建训练配置
        training_config = TrainingConfig(
            num_epochs=int(config['training']['epochs']),
            batch_size=int(config['data']['batch_size']),
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            gradient_clip_norm=float(config['training']['gradient_clip']),
            optimizer_type=str(config['training'].get('optimizer_type', 'adamw')),
            scheduler_type=str(config['training'].get('scheduler_type', 'cosine')),
            mixed_precision=bool(config['training'].get('mixed_precision', True)),
            accumulation_steps=int(config['training'].get('accumulation_steps', 1)),
            early_stopping_patience=int(config.get('validation', {}).get('early_stopping_patience', 20)),
            use_early_stopping=False,
            validation_frequency=1,  # 按方案A：每个epoch都进行验证
            save_frequency=int(config.get('checkpoints', {}).get('save_frequency', 10)),
            warmup_epochs=int(config['training'].get('warmup_epochs', 5)),
            eta_min=float(config['training'].get('eta_min', 0.0))
        )
        
        # 创建回调函数（单次运行使用 run_root_dir）
        callbacks = [
            ModelCheckpoint(
                filepath=f"{run_root_dir}/checkpoints/ckpt_epoch_{{epoch}}.pth",
                monitor='val_loss',
                save_best_only=True,
                save_freq=training_config.save_frequency,
                best_save_freq=int(config.get('checkpoints', {}).get('best_save_freq', 0)),
                final_best_path=f"{run_root_dir}/best_model.pth"
            )
        ]
        
        # 创建训练器
        logger.info("创建训练器...")
        trainer = Trainer(
            model=model,
            config=training_config,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=callbacks
        )
        
        # 设置损失函数：分类模型统一使用 BCEWithLogitsLoss
        if args.model == 'pocket_ligand_transformer':
            trainer.criterion = create_loss_function('pl_classification')
        else:
            trainer.criterion = create_loss_function('binding_affinity')
        
        logger.info("✓ 训练器创建完成")
        
        # 开始训练
        logger.info("=" * 50)
        logger.info("开始训练...")
        logger.info("=" * 50)
        
        trainer.train()
        
        logger.info("=" * 50)
        logger.info("训练完成!")
        logger.info("=" * 50)
        
        # 在测试集上评估前，优先加载验证集表现最佳的权重
        best_model_path = f"{run_root_dir}/best_model.pth"
        if os.path.exists(best_model_path):
            try:
                logger.info("加载最佳验证模型进行测试评估...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                best_ckpt = torch.load(best_model_path, map_location=device)
                model.load_state_dict(best_ckpt['model_state_dict'])
                logger.info("✓ 已加载最佳验证权重")
            except Exception as e:
                logger.warning(f"加载最佳模型失败，使用最后一轮权重进行测试: {e}")
        else:
            logger.warning(f"未找到最佳模型文件: {best_model_path}，将使用最后一轮权重进行测试")

        # 基于验证集做阈值校准（适用于二分类，寻找最佳F1阈值）
        best_threshold = 0.5
        try:
            val_eval = trainer.evaluate(val_loader, return_predictions=True)
            vpreds = val_eval.get('predictions', {})
            vtargets = val_eval.get('targets', {})
            prob = vpreds.get('prob')
            if prob is None and 'logit' in vpreds:
                # sigmoid(logit)
                prob = 1.0 / (1.0 + np.exp(-np.array(vpreds['logit']).reshape(-1)))
            if prob is not None:
                prob = np.array(prob).reshape(-1)
                y_true = None
                for key in ('y', 'label'):
                    if key in vtargets:
                        y_true = np.array(vtargets[key]).reshape(-1)
                        break
                if y_true is not None and y_true.size > 0:
                    thresholds = np.linspace(0.05, 0.95, 19)
                    best_f1 = -1.0
                    for t in thresholds:
                        y_pred = (prob > t).astype(int)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = float(t)
                    logger.info(f"验证集阈值校准完成: best_threshold={best_threshold:.3f}, best_f1={best_f1:.4f}")
        except Exception as e:
            logger.warning(f"阈值校准失败，使用默认0.5: {e}")

        # 在测试集上评估（返回指标与预测，包括概率）
        logger.info("在测试集上评估...")
        test_eval = trainer.evaluate(test_loader, return_predictions=True)
        logger.info(f"测试指标: {test_eval.get('metrics', {})}")
        preds = test_eval.get('predictions', {})
        if 'prob' in preds:
            try:
                # 使用全局导入的 np 进行统计
                prob = preds['prob']
                logger.info(f"测试概率概览: shape={prob.shape}, min={float(np.min(prob)):.6f}, mean={float(np.mean(prob)):.6f}, max={float(np.max(prob)):.6f}")
            except Exception:
                pass
        
        # 计算并记录基于校准阈值的测试集指标
        try:
            tpreds = test_eval.get('predictions', {})
            ttargets = test_eval.get('targets', {})
            tprob = tpreds.get('prob')
            if tprob is None and 'logit' in tpreds:
                tprob = 1.0 / (1.0 + np.exp(-np.array(tpreds['logit']).reshape(-1)))
            if tprob is not None:
                tprob = np.array(tprob).reshape(-1)
                y_true_t = None
                for key in ('y', 'label'):
                    if key in ttargets:
                        y_true_t = np.array(ttargets[key]).reshape(-1)
                        break
                if y_true_t is not None and y_true_t.size > 0:
                    y_pred_t = (tprob > best_threshold).astype(int)
                    calib_metrics = {
                        'threshold': float(best_threshold),
                        'accuracy': float(accuracy_score(y_true_t, y_pred_t)),
                        'precision': float(precision_score(y_true_t, y_pred_t, zero_division=0)),
                        'recall': float(recall_score(y_true_t, y_pred_t, zero_division=0)),
                        'f1': float(f1_score(y_true_t, y_pred_t, zero_division=0))
                    }
                    # AUC/AP 与阈值无关，直接基于概率计算（若标签不止一个类）
                    if len(np.unique(y_true_t)) > 1:
                        try:
                            calib_metrics['auc'] = float(roc_auc_score(y_true_t, tprob))
                            calib_metrics['ap'] = float(average_precision_score(y_true_t, tprob))
                        except Exception:
                            pass
                    test_eval['metrics_calibrated'] = calib_metrics
                    logger.info(f"测试集（校准阈值）: {calib_metrics}")
        except Exception as e:
            logger.warning(f"测试集校准指标计算失败: {e}")

        # 将测试评估另存为 JSON 便于外部分析
        try:
            out_dir = Path('results')
            out_dir.mkdir(parents=True, exist_ok=True)
            def _to_jsonable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _to_jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_jsonable(v) for v in obj]
                try:
                    # numpy 标量
                    if hasattr(obj, 'item'):  # np.generic
                        return obj.item()
                except Exception:
                    pass
                return obj
            test_eval_json = _to_jsonable(test_eval)
            json_path = out_dir / 'test_eval.json'
            with open(json_path, 'w') as f:
                json.dump(test_eval_json, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 测试评估JSON已保存: {json_path}")
        except Exception as e:
            logger.warning(f"保存测试评估JSON失败: {e}")

        # 保存最终模型
        final_model_path = f"{run_root_dir}/final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config.to_dict(),
            'training_config': training_config.__dict__,
            'test_eval': test_eval
        }, final_model_path)
        
        logger.info(f"✓ 最终模型已保存: {final_model_path}")
        
    except ImportError as e:
        logger.error(f"✗ 导入模块失败: {e}")
        logger.info("请检查deep_learning模块路径和依赖")
        import traceback
        traceback.print_exc()
        return
    except Exception as e:
        logger.error(f"✗ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
