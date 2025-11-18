"""
EDA for CodexGLUE JSONL data.

Outputs a concise text summary and optional figures into --outdir.

Usage:
  python training/analysis/eda_codexglue.py \
    --train-jsonl data/processed/codexglue/train.jsonl \
    --val-jsonl data/processed/codexglue/valid.jsonl \
    --outdir training/outputs/eda_codexglue
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

import numpy as np


def load_jsonl(path: Path, max_rows: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def summarize_split(name: str, rows: list[dict]) -> Dict[str, Any]:
    n = len(rows)
    labels = np.array([r.get('label', r.get('target', 0)) for r in rows], dtype=int)
    pos = labels.sum()
    neg = n - pos
    label_frac = {'safe_0': float(neg) / n if n else 0.0, 'vuln_1': float(pos) / n if n else 0.0}

    codes = [str(r.get('code') or r.get('func') or '') for r in rows]
    code_len = np.array([len(c) for c in codes], dtype=int)

    tokens = [r.get('tokens') for r in rows]
    token_len = np.array([len(t) if isinstance(t, list) else np.nan for t in tokens], dtype=float)

    stats = {
        'name': name,
        'num_rows': n,
        'label_fraction': label_frac,
        'code_len': {
            'mean': float(np.nanmean(code_len)) if n else 0.0,
            'p50': float(np.nanpercentile(code_len, 50)) if n else 0.0,
            'p75': float(np.nanpercentile(code_len, 75)) if n else 0.0,
            'p90': float(np.nanpercentile(code_len, 90)) if n else 0.0,
            'max': int(np.nanmax(code_len)) if n else 0,
        },
    }
    if np.isfinite(token_len).any():
        stats['token_len'] = {
            'mean': float(np.nanmean(token_len)),
            'p50': float(np.nanpercentile(token_len, 50)),
            'p75': float(np.nanpercentile(token_len, 75)),
            'p90': float(np.nanpercentile(token_len, 90)),
            'max': int(np.nanmax(token_len[np.isfinite(token_len)])),
        }
    return stats


def write_summary(outdir: Path, train_stats: Dict[str, Any], val_stats: Dict[str, Any], overlap_count: int):
    outdir.mkdir(parents=True, exist_ok=True)
    text = []
    for s in (train_stats, val_stats):
        text.append(f"=== {s['name']} ===")
        text.append(f"rows: {s['num_rows']}")
        lf = s['label_fraction']
        text.append(f"labels: safe_0={lf['safe_0']:.3f}, vuln_1={lf['vuln_1']:.3f}")
        cl = s['code_len']
        text.append(f"code_len: mean={cl['mean']:.1f} p50={cl['p50']:.0f} p75={cl['p75']:.0f} p90={cl['p90']:.0f} max={cl['max']}")
        if 'token_len' in s:
            tl = s['token_len']
            text.append(f"token_len: mean={tl['mean']:.1f} p50={tl['p50']:.0f} p75={tl['p75']:.0f} p90={tl['p90']:.0f} max={tl['max']}")
        text.append("")
    text.append(f"train/val code hash overlap: {overlap_count}")
    (outdir / 'summary.txt').write_text("\n".join(text), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='EDA for CodexGLUE JSONL splits')
    ap.add_argument('--train-jsonl', type=Path, required=True)
    ap.add_argument('--val-jsonl', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, default=Path('training/outputs/eda_codexglue'))
    ap.add_argument('--max-rows', type=int, default=None)
    args = ap.parse_args()

    train_rows = load_jsonl(args.train_jsonl, args.max_rows)
    val_rows = load_jsonl(args.val_jsonl, args.max_rows)

    train_stats = summarize_split('train', train_rows)
    val_stats = summarize_split('val', val_rows)

    # Duplicate detection via code hash
    train_hash = {sha256(str(r.get('code') or r.get('func') or '')) for r in train_rows}
    val_hash = {sha256(str(r.get('code') or r.get('func') or '')) for r in val_rows}
    overlap = len(train_hash & val_hash)

    write_summary(args.outdir, train_stats, val_stats, overlap)
    print(f"[ok] Wrote EDA summary to {args.outdir/'summary.txt'}")


if __name__ == '__main__':
    main()

