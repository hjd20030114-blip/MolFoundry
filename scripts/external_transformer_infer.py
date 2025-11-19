#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部 Transformer 推理脚本模板

用途:
- 被 deep_learning_pipeline.py 通过子进程调用，按口袋条件生成候选分子
- 你可以在 run_inference() 中替换为自己的真实 Transformer 模型推理逻辑

输入参数:
--protein <path>        蛋白质PDB文件路径
--num <int>             生成分子数量
--target_aff <float>    目标结合亲和力 (仅用于占位打分)
--out <path>            输出CSV路径 (至少包含 'smiles' 列; 可选: 'binding_affinity','molecular_weight','logp')
--checkpoint <path>     (可选) 模型检查点

输出CSV列建议:
- smiles (必须)
- binding_affinity (可选, 若未提供, 上层会用 target_aff 填充)
- molecular_weight (可选)
- logp (可选)

你可以自由扩展更多列, 上层会择要读取。
"""

import argparse
import sys
import os
import json
import importlib
import inspect
import pandas as pd
import numpy as np

# 尝试使用 RDKit 计算分子性质(可选)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


def run_inference(protein_pdb: str, num: int, target_aff: float, checkpoint: str | None) -> pd.DataFrame:
    """
    在这里替换为你的真实 Transformer 条件生成逻辑。
    该模板返回一个基于内置片段库的示例结果，便于联调与演示。
    """
    # 示例基础库
    base_library = [
        "CCO", "CCN", "CCC", "CC(C)O", "CC(C)N", "C1CCCCC1", "c1ccccc1",
        "Clc1ccccc1", "Fc1ccccc1", "COc1ccccc1", "CNC(=O)C1=CC=CC=C1",
        "C1=CC(=O)NC(=O)N1", "CCS(=O)(=O)N1CCOCC1", "N1CCOCC1", "O1CCNCC1",
        "c1ccc2ccccc2c1", "O=C(Nc1ccc(F)cc1)C2CC2"
    ]
    rng = np.random.default_rng()
    chosen = list(rng.choice(base_library, size=max(1, int(num)), replace=True))

    rows = []
    for smi in chosen:
        mw = None
        logp = None
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    mw = float(Descriptors.MolWt(mol))
                    logp = float(Crippen.MolLogP(mol))
            except Exception:
                pass
        # 简单占位打分: 围绕 target_aff 加噪声
        aff = float(target_aff + rng.normal(0.0, 0.6))
        aff = float(np.clip(aff, -12.0, -3.0))
        rows.append({
            'smiles': smi,
            'binding_affinity': aff,
            'molecular_weight': mw if mw is not None else 'N/A',
            'logp': logp if logp is not None else 'N/A'
        })
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="External Transformer Inference")
    parser.add_argument("--protein", required=True, type=str)
    parser.add_argument("--num", required=True, type=int)
    parser.add_argument("--target_aff", required=True, type=float)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--module", type=str, default=None, help="Python模块路径，如 mypkg.mymodule")
    parser.add_argument("--function", type=str, default=None, help="模块内函数名，如 generate")
    parser.add_argument("--kwargs_json", type=str, default=None, help="额外关键字参数(JSON字符串)")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None

    # 尝试动态导入用户函数
    df = None
    if args.module and args.function:
        try:
            mod = importlib.import_module(args.module)
            fn = getattr(mod, args.function)

            extra = {}
            if args.kwargs_json:
                try:
                    extra = json.loads(args.kwargs_json)
                except Exception:
                    extra = {}

            # 尝试多种调用方式
            called = False
            for signature in [
                dict(protein_pdb=args.protein, num=args.num, target_aff=args.target_aff, checkpoint=args.checkpoint, **extra),
                dict(protein=args.protein, num=args.num, target_aff=args.target_aff, checkpoint=args.checkpoint, **extra),
                (args.protein, args.num, args.target_aff, args.checkpoint),
            ]:
                try:
                    if isinstance(signature, dict):
                        out = fn(**signature)
                    else:
                        out = fn(*signature)
                    called = True
                    break
                except TypeError:
                    continue

            if not called:
                # 最后尝试仅必须参数
                out = fn(args.protein, args.num)

            # 归一化输出
            if isinstance(out, pd.DataFrame):
                df = out
            elif isinstance(out, list):
                if len(out) == 0:
                    df = pd.DataFrame(columns=['smiles'])
                elif isinstance(out[0], str):
                    df = pd.DataFrame({'smiles': out})
                elif isinstance(out[0], dict):
                    df = pd.DataFrame(out)
            elif isinstance(out, dict) and 'smiles' in out:
                df = pd.DataFrame([out])
        except Exception as e:
            print(f"WARN: 动态导入/调用失败，使用模板占位: {e}", file=sys.stderr)

    if df is None:
        df = run_inference(args.protein, args.num, args.target_aff, args.checkpoint)
    if df is None or df.empty or 'smiles' not in df.columns:
        print("ERROR: No valid output (missing 'smiles' column).", file=sys.stderr)
        return 2

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} molecules to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
