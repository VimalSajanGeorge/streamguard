"""
Post-training evaluation for Transformer runs.

- Threshold sweep (P/R/F1 vs threshold)
- ROC-AUC and PR-AUC (if probabilities available)
- Slice metrics by code length bucket
- Error bank (top false positives/negatives)

Usage:
  python training/analysis/eval_transformer.py \
    --val-jsonl data/processed/codexglue/valid.jsonl \
    --run-dir training/outputs/transformer_presets/graphcodebert_t4/seed_42 \
    --outdir training/outputs/transformer_presets/graphcodebert_t4/seed_42/eval \
    --sweep-start 0.30 --sweep-stop 0.70 --sweep-step 0.01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
    )
except Exception:
    raise RuntimeError("scikit-learn is required for eval (pip install scikit-learn)")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, thr_list: list[float]) -> list[dict]:
    out = []
    for thr in thr_list:
        y_pred = (y_score >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        out.append({
            'threshold': round(float(thr), 4),
            'precision': round(float(prec), 6),
            'recall': round(float(rec), 6),
            'f1': round(float(f1), 6),
        })
    return out


def add_len_bucket(code_list: list[str]) -> list[str]:
    buckets = []
    for code in code_list:
        L = len(code or "")
        if L <= 100:
            b = 'tiny'
        elif L <= 300:
            b = 'short'
        elif L <= 800:
            b = 'medium'
        elif L <= 2000:
            b = 'long'
        else:
            b = 'huge'
        buckets.append(b)
    return buckets


def load_predictions(run_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, list[dict]]]:
    """
    Try to load val predictions saved by training (when --dump-val-logits is enabled).
    Returns (y_true, y_score, raw_rows) or None if file is absent.
    """
    pred_path = run_dir / 'val_predictions.jsonl'
    if not pred_path.exists():
        return None
    rows = load_jsonl(pred_path)
    y_true_list = []
    y_score_list = []
    for r in rows:
        label = r.get('label')
        if label is None:
            label = r.get('target')
        # score: prefer prob_vuln; else logit_1
        prob = r.get('prob_vuln')
        if prob is None:
            logit = r.get('logit_1')
            if logit is None:
                # try generic key
                logit = r.get('logit')
            if logit is None:
                continue
            prob = float(sigmoid(np.array([logit]))[0])
        if label is None:
            continue
        y_true_list.append(int(label))
        y_score_list.append(float(prob))
    if not y_true_list:
        return None
    return np.array(y_true_list, dtype=int), np.array(y_score_list, dtype=float), rows


def main():
    ap = argparse.ArgumentParser(description='Post-training evaluation for Transformer runs')
    ap.add_argument('--val-jsonl', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True, help='Run folder containing metrics.json and (optionally) val_predictions.jsonl')
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--sweep-start', type=float, default=0.30)
    ap.add_argument('--sweep-stop', type=float, default=0.70)
    ap.add_argument('--sweep-step', type=float, default=0.01)
    ap.add_argument('--max-examples', type=int, default=20)
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # Load predictions if available
    pred_pack = load_predictions(args.run_dir)
    have_probs = pred_pack is not None

    # Load val data for slices / error bank
    val_rows = load_jsonl(args.val_jsonl)
    codes = [str(r.get('code') or r.get('func') or '') for r in val_rows]
    buckets = add_len_bucket(codes)
    y_true_val = np.array([int(r.get('label', r.get('target', 0))) for r in val_rows], dtype=int)

    thr_list = []
    t = args.sweep_start
    while t <= args.sweep_stop + 1e-9:
        thr_list.append(round(t, 4))
        t += args.sweep_step

    # If we have probabilities → full eval
    if have_probs:
        y_true, y_score, raw_rows = pred_pack

        # Align lengths (fallback if mismatch)
        if len(y_true) != len(y_true_val):
            # Try to match by min length
            min_n = min(len(y_true), len(y_true_val))
            y_true = y_true[:min_n]
            y_score = y_score[:min_n]
            y_true_val = y_true_val[:min_n]
            codes_trim = codes[:min_n]
            buckets_trim = buckets[:min_n]
        else:
            codes_trim = codes
            buckets_trim = buckets

        # Threshold sweep
        sweep = threshold_sweep(y_true, y_score, thr_list)
        (args.outdir / 'threshold_curve.csv').write_text(
            'threshold,precision,recall,f1\n' + '\n'.join(
                f"{d['threshold']},{d['precision']},{d['recall']},{d['f1']}" for d in sweep
            ),
            encoding='utf-8'
        )

        # AUCs
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
        (args.outdir / 'roc_pr.json').write_text(
            json.dumps({'roc_auc': roc_auc, 'pr_auc': pr_auc}, indent=2), encoding='utf-8'
        )

        # Choose best threshold from sweep by F1
        best = max(sweep, key=lambda d: d['f1'])
        best_thr = best['threshold']

        # Slice metrics (length buckets) at best_thr
        y_pred = (y_score >= best_thr).astype(int)
        rows = []
        for b in sorted(set(buckets_trim)):
            mask = np.array([bb == b for bb in buckets_trim])
            if mask.sum() == 0:
                continue
            prec, rec, f1, _ = precision_recall_fscore_support(y_true[mask], y_pred[mask], average='binary', zero_division=0)
            rows.append({'bucket': b, 'n': int(mask.sum()), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)})
        # save slice metrics
        (args.outdir / 'slice_metrics.csv').write_text(
            'bucket,n,precision,recall,f1\n' + '\n'.join(
                f"{r['bucket']},{r['n']},{r['precision']:.6f},{r['recall']:.6f},{r['f1']:.6f}" for r in rows
            ),
            encoding='utf-8'
        )

        # Error bank
        # Top FPs: pred=1, label=0 with highest score; Top FNs: pred=0, label=1 with lowest score
        details = []
        for i in range(len(y_true)):
            details.append({'i': i, 'label': int(y_true[i]), 'score': float(y_score[i]), 'code': codes_trim[i], 'len': len(codes_trim[i])})
        fps = [d for d in details if d['label'] == 0 and d['score'] >= best_thr]
        fns = [d for d in details if d['label'] == 1 and d['score'] < best_thr]
        fps_sorted = sorted(fps, key=lambda d: d['score'], reverse=True)[: args.max_examples]
        fns_sorted = sorted(fns, key=lambda d: d['score'])[: args.max_examples]
        (args.outdir / 'error_bank_fp.jsonl').write_text('\n'.join(json.dumps({'prob_vuln': d['score'], 'label': d['label'], 'code_len': d['len'], 'code_preview': d['code'][:400]}) for d in fps_sorted), encoding='utf-8')
        (args.outdir / 'error_bank_fn.jsonl').write_text('\n'.join(json.dumps({'prob_vuln': d['score'], 'label': d['label'], 'code_len': d['len'], 'code_preview': d['code'][:400]}) for d in fns_sorted), encoding='utf-8')

        print(f"[ok] Eval saved under {args.outdir}")
        print(f"     best_thr={best_thr} roc_auc={roc_auc:.4f} pr_auc={pr_auc:.4f}")

    else:
        # Fallback: try to dump threshold sweep from metrics.json if provided there
        metrics_path = args.run_dir / 'metrics.json'
        if not metrics_path.exists():
            print('[warn] No val_predictions.jsonl or metrics.json found; nothing to evaluate.')
            return
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        sweep = metrics.get('threshold_sweep')
        if not sweep:
            print('[warn] No predictions and no threshold_sweep in metrics.json; limited evaluation.')
            return
        # write threshold curve only
        (args.outdir / 'threshold_curve.csv').write_text(
            'threshold,precision,recall,f1\n' + '\n'.join(
                f"{round(float(d.get('threshold', 0.0)),4)},{float(d.get('precision',0.0))},{float(d.get('recall',0.0))},{float(d.get('f1',0.0))}" for d in sweep
            ),
            encoding='utf-8'
        )
        print(f"[ok] Wrote threshold curve from metrics.json to {args.outdir/'threshold_curve.csv'}")


if __name__ == '__main__':
    main()

