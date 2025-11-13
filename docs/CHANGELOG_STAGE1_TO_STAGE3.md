# StreamGuard – Transformer/GNN Upgrades (Stage 1 → Stage 3)

This changelog captures the work completed across Stage 1, Stage 2, and Stage 3, including file‑level changes, new flags, metrics, and notebook updates. It is intended as a concise engineering record of what shipped and how to run it.

## Summary
- Stage 1: Selectable Transformer backbones (CodeBERT, GraphCodeBERT, UniXcoder).
- Stage 2: Stability knobs + JSON metrics for Transformer (cosine/EMA/AMP/clip/pad_to_multiple_of; F1@0.5 and F1@best saved to metrics.json).
- Stage 3A: GNN parity (preprocessing flags + DataLoader/AMP/scheduler + metrics.json).
- Stage 3B: GraphCodeBERT CLS features for GNN (opt‑in) + token‑mode scaffolding + notebook GNN runners.

## Commits (latest first)
- 83bfe83 feat(gnn): token‑mode scaffolding (node_embeds) + GNN production multi‑seed runner cell
- 225a14f feat(gnn): Stage 3B CLS encoder features + notebook GNN preset/runner cells
- af541f9 feat(gnn): Stage 3A parity — preprocessing flags/metadata, DataLoader knobs, AMP/scheduler, and metrics.json with threshold sweep
- 35443e8 feat(transformer): add backbone presets and stability knobs

## Stage 1 — Backbones (Transformer)
Files touched:
- training/models/backbones.py (new) — `load_backbone(model_key)` returns `(tokenizer, encoder, hidden_size, pooling)`.
- training/train_transformer.py — integrates `--model-name {codebert,graphcodebert,unixcoder}`; uses `load_backbone`.
- StreamGuard_Production_Training.ipynb — presets/runner expose MODEL_NAME and pass it to CLI.

Outcome:
- A/B CodeBERT vs GraphCodeBERT without changing defaults (defaults to CodeBERT).

## Stage 2 — Stability + JSON Metrics (Transformer)
Files touched:
- training/train_transformer.py
  - New flags (default‑safe): `--scheduler {linear,cosine}`, `--amp-dtype {fp16,bf16}`, `--grad-clip-norm`, `--pad-to-multiple-of`, `--ema`, `--ema-decay`, DataLoader knobs (`--num-workers`, `--prefetch-factor`, `--persistent-workers`, `--drop-last`).
  - AMP dtype resolution and safe fallback (bf16→fp16 if unsupported).
  - Cosine step per‑batch; plateau/linear retained as before.
  - Threshold sweep (0.30–0.70 by 0.01); metrics.json now includes:
    - `f1_at_0_5`, `f1_at_best_threshold`, `best_threshold_by_f1`, `balanced_accuracy_at_best_threshold`, confusion matrices at both thresholds.
- StreamGuard_Production_Training.ipynb
  - Preset runner and production cells updated to read metrics.json (not regex) and print both F1@0.5 and F1@best.

Outcome:
- Higher training stability on GPUs, consistent metrics artifacts, unchanged defaults.

## Stage 3A — GNN Parity
Files touched:
- training/preprocessing/create_simple_graph_data.py
  - Added: `--train-jsonl`, `--val-jsonl`, `--output`, `--max-samples`, `--force`.
  - Writes `graphs_metadata.json` with split counts, avg nodes/edges, timestamp.
- training/train_gnn.py
  - DataLoader knobs: `--num-workers`, `--prefetch-factor`, `--persistent-workers`, `--drop-last` (pin_memory if CUDA).
  - AMP: `--mixed-precision`, `--amp-dtype {fp16,bf16}` + per‑batch autocast.
  - Scheduler: `--scheduler {plateau,cosine,none}`.
  - Grad clipping: `--grad-clip-norm`.
  - Threshold sweep + metrics.json (parity with Transformer): `f1_at_0_5`, `f1_at_best_threshold`, `best_threshold_by_f1`, `balanced_accuracy_at_best_threshold`, confusion matrices.

Outcome:
- GNN pipeline has the same stability/features and metrics artifacts as Transformer; defaults unchanged.

## Stage 3B — GraphCodeBERT Features (GNN, Opt‑In)
Files touched:
- training/preprocessing/augment_graphs_with_encoder.py (new)
  - CLS mode: adds `graph_cls` embedding to JSONL.
  - Token mode (scaffolding): when `node_texts` present, writes `node_embeds` per node.
  - Flags: `--input`, `--output`, `--model-name {codebert,graphcodebert,unixcoder}`, `--embedding-type {cls,token}`, `--max-samples`.
- training/train_gnn.py
  - GraphDataset understands `encoder-features {none,cls,token}`; attaches `graph_features` when CLS present.
  - Model concatenates CLS features after pooling; token mode projects precomputed node features to the GNN embedding dimension.
- StreamGuard_Production_Training.ipynb
  - GNN preset catalog/runner (single/multi seed) + production multi‑seed cell.
  - Appended markdown notes explaining Stage 3 usage with copy‑paste commands.

Outcome:
- CLS features provide an opt‑in path to leverage GraphCodeBERT; token‑mode is available when node_texts metadata is present. Defaults remain unchanged.

## Quick Start (CLI)
- Create small graphs:
  - `python training/preprocessing/create_simple_graph_data.py --output data/processed/graphs_sample --max-samples 200`
- GNN quick run:
  - `python training/train_gnn.py --train-data data/processed/graphs_sample/train.jsonl --val-data data/processed/graphs_sample/val.jsonl --epochs 2 --batch-size 8 --quick-test`
- Augment CLS features:
  - `python training/preprocessing/augment_graphs_with_encoder.py --input data/processed/codexglue/train.jsonl --output data/processed/graphs_encoder/train.jsonl --model-name graphcodebert --max-samples 500`
- GNN + CLS:
  - `python training/train_gnn.py --train-data data/processed/graphs_encoder/train.jsonl --val-data data/processed/graphs_encoder/val.jsonl --encoder-features cls --epochs 2 --batch-size 8 --quick-test`
- Transformer (unchanged baseline):
  - `python training/train_transformer.py --train-data data/processed/codexglue/train.jsonl --val-data data/processed/codexglue/valid.jsonl --output-dir outputs/test --quick-test --epochs 1`

## Notes / Open Items
- Fusion Stage 4: ensure `training/train_fusion.py` writes a parity metrics.json; add fusion preset/runner/production cells (planned next).
- Ensembling Stage 5: add `training/scripts/aggregate_seeds.py` + notebook ensemble cells.

