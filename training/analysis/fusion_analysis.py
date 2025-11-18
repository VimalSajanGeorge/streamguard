"""
Fusion Complementarity Analysis

Given Transformer and GNN validation predictions (JSONL), compute:
 - Complementarity matrix (both correct / transformer-only / gnn-only / both wrong)
 - Simple fusion by probability averaging and its P/R/F1
 - Threshold sweep for the fused score (optional)

Assumes each JSONL row contains at least: { 'label': 0/1, 'prob_vuln': float }
If a join key is present (e.g., 'code_hash' or 'id'), use --key to align; otherwise align by index.

Usage:
  python training/analysis/fusion_analysis.py \
    --trans-preds training/outputs/transformer_presets/.../val_predictions.jsonl \
    --gnn-preds training/outputs/gnn_presets/.../val_predictions.jsonl \
    --outdir training/outputs/fusion_analysis \
    --key code_hash --thr 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def align_rows(trans: List[Dict[str, Any]], gnn: List[Dict[str, Any]], key: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if key and all(key in r for r in trans) and all(key in r for r in gnn):
        gmap = {r[key]: r for r in gnn}
        for r in trans:
            k = r[key]
            rg = gmap.get(k)
            if rg is None:
                continue
            out.append({
                'label': r.get('label', r.get('target', 0)),
                'p_t': r.get('prob_vuln'),
                'p_g': rg.get('prob_vuln')
            })
    else:
        # Align by index; length = min
        n = min(len(trans), len(gnn))
        for i in range(n):
            rt, rg = trans[i], gnn[i]
            out.append({
                'label': rt.get('label', rt.get('target', 0)),
                'p_t': rt.get('prob_vuln'),
                'p_g': rg.get('prob_vuln')
            })
    # Filter valid
    out = [r for r in out if r['label'] is not None and r['p_t'] is not None and r['p_g'] is not None]
    return out


def complementarity(rows: List[Dict[str, Any]], thr_t: float, thr_g: float) -> Dict[str, float]:
    y = np.array([int(r['label']) for r in rows], dtype=int)
    pt = np.array([float(r['p_t']) for r in rows], dtype=float)
    pg = np.array([float(r['p_g']) for r in rows], dtype=float)

    pred_t = (pt >= thr_t).astype(int)
    pred_g = (pg >= thr_g).astype(int)

    both_correct = np.mean((pred_t == y) & (pred_g == y))
    t_only = np.mean((pred_t == y) & (pred_g != y))
    g_only = np.mean((pred_g == y) & (pred_t != y))
    both_wrong = np.mean((pred_t != y) & (pred_g != y))
    return {
        'both_correct': float(both_correct),
        'transformer_only': float(t_only),
        'gnn_only': float(g_only),
        'both_wrong': float(both_wrong),
        'n': int(len(rows))
    }


def fused_metrics(rows: List[Dict[str, Any]], thr: float = 0.5, alpha: float = 0.5) -> Dict[str, float]:
    y = np.array([int(r['label']) for r in rows], dtype=int)
    pt = np.array([float(r['p_t']) for r in rows], dtype=float)
    pg = np.array([float(r['p_g']) for r in rows], dtype=float)
    pf = alpha * pt + (1 - alpha) * pg
    y_pred = (pf >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}


def main():
    ap = argparse.ArgumentParser(description='Fusion complementarity analysis')
    ap.add_argument('--trans-preds', type=Path, required=True)
    ap.add_argument('--gnn-preds', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--key', type=str, default=None, help='Join key (e.g., code_hash); if absent, align by index')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=0.5, help='Fusion weight (alpha*Trans + (1-alpha)*GNN)')
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    trans = load_jsonl(args.trans_preds)
    gnn = load_jsonl(args.gnn_preds)
    rows = align_rows(trans, gnn, args.key)
    if not rows:
        raise RuntimeError('No aligned rows between transformer and gnn predictions. Check --key or files.')

    comp = complementarity(rows, args.thr, args.thr)
    fuse = fused_metrics(rows, thr=args.thr, alpha=args.alpha)

    (outdir / 'complementarity.json').write_text(json.dumps(comp, indent=2), encoding='utf-8')
    (outdir / 'fusion_metrics.json').write_text(json.dumps(fuse, indent=2), encoding='utf-8')
    print('[ok] Wrote complementarity and fusion metrics to', outdir)


if __name__ == '__main__':
    main()

