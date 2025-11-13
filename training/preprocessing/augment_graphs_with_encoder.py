"""
Augment graph JSONL with GraphCodeBERT CLS embeddings (opt-in).

Reads an input JSONL of code samples and writes an output JSONL with an
additional field `graph_cls` (list[float] of length 768 by default) for each
sample. This is used as graph-level encoder features in GNN training
(`--encoder-features cls`).

Usage:
    python training/preprocessing/augment_graphs_with_encoder.py \
      --input data/processed/codexglue/train.jsonl \
      --output data/processed/graphs_encoder/train.jsonl \
      --model-name graphcodebert --max-samples 1000

Notes:
    - Token mode is intentionally not implemented here to avoid requiring
      token→node alignment metadata. CLS mode is simpler and safe.
"""

import json
from pathlib import Path
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def load_encoder(model_key: str):
    model_map = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'unixcoder': 'microsoft/unixcoder-base'
    }
    name = model_map.get(model_key.lower(), model_key)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModel.from_pretrained(name)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device


@torch.no_grad()
def compute_cls(tokenizer, model, device, text: str) -> Optional[list]:
    if not text:
        return None
    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    hidden = outputs.last_hidden_state[:, 0]  # CLS
    return hidden.squeeze(0).detach().cpu().tolist()


def main():
    ap = argparse.ArgumentParser(description='Augment JSONL with GraphCodeBERT CLS embeddings')
    ap.add_argument('--input', type=Path, required=True, help='Input JSONL')
    ap.add_argument('--output', type=Path, required=True, help='Output JSONL (augmented)')
    ap.add_argument('--model-name', type=str, default='graphcodebert',
                   choices=['codebert', 'graphcodebert', 'unixcoder'])
    ap.add_argument('--max-samples', type=int, default=None)
    args = ap.parse_args()

    tokenizer, model, device = load_encoder(args.model_name)

    lines = []
    with args.input.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)

    if args.max_samples:
        lines = lines[:args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open('w', encoding='utf-8') as out:
        for line in tqdm(lines, desc='Augmenting'):
            sample = json.loads(line)
            code = sample.get('code') or sample.get('func') or ''
            cls_vec = compute_cls(tokenizer, model, device, code)
            if cls_vec is not None:
                sample['graph_cls'] = cls_vec
            out.write(json.dumps(sample) + '\n')

    meta = {
        'encoder': args.model_name,
        'feature': 'cls',
        'dim': len(sample['graph_cls']) if 'graph_cls' in sample else 0
    }
    meta_path = args.output.with_suffix('.encoder_metadata.json')
    with meta_path.open('w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)
    print(f"[+] Wrote metadata: {meta_path}")


if __name__ == '__main__':
    main()

