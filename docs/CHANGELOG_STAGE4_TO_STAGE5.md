# Changelog — Stage 4 (Fusion) and Stage 5 (Seed Ensembling)

## Overview

Stage 4 introduced a late‑fusion baseline that combines Transformer and GNN representations with full metric/runner parity. Stage 5 added a seed aggregation utility to ensemble validation logits across seeds and report final metrics.

## Changes by File

- training/train_fusion.py
  - Late‑fusion model: concat(Transformer CLS, GNN pooled) → MLP head (dropout 0.2–0.3).
  - CLI: `--transformer-checkpoint`, `--gnn-checkpoint`, `--model-name {codebert,graphcodebert,unixcoder}`, `--freeze-transformer`, `--freeze-gnn`.
  - Stability knobs (default‑off): `--scheduler {linear,cosine}`, `--amp-dtype {fp16,bf16}`, `--grad-clip-norm`, `--ema --ema-decay`, `--pad-to-multiple-of`, DataLoader options.
  - Loss options: CE + class weights + label smoothing; optional focal loss in head.
  - Metrics.json parity: `f1_at_0_5`, `f1_at_best_threshold`, `best_threshold_by_f1`, `balanced_accuracy_at_best_threshold`, confusion matrices, `threshold_sweep`, timestamp.

- training/scripts/aggregate_seeds.py
  - CLI: `--track {transformer,gnn,fusion}` `--input-dir` `--output` `--recompute`.
  - Loads per‑seed validation logits, averages logits, sweeps thresholds 0.30–0.70 (step 0.01), writes `ensemble_metrics.json`.
  - Graceful fallback when logits missing; prints actionable hints.

- StreamGuard_Production_Training.ipynb
  - Added Fusion preset catalog and runner cells (single‑seed and multi‑seed).
  - Added Ensemble runner cells for Transformer/GNN/Fusion + “Final Metrics” summary cell.
  - All runners consume metrics JSON; no regex parsing.

## Presets

- fusion_balanced: head 5–8 epochs, dropout=0.2–0.3, EMA off, cosine scheduler, warmup 0.1, grad clip 1.0.
- fusion_high_recall: focal loss γ=2.0 in head; same scheduler/clip.

## Run Recipes (examples)

- Fusion (A100 recommended flags):
  - `python training/train_fusion.py \
     --transformer-checkpoint training/outputs/transformer/best.ckpt \
     --gnn-checkpoint training/outputs/gnn/best.ckpt \
     --model-name graphcodebert --epochs 6 --batch-size 16 \
     --scheduler cosine --amp-dtype bf16 --grad-clip-norm 1.0 --pad-to-multiple-of 8`

- Ensemble Transformer seeds:
  - `python training/scripts/aggregate_seeds.py --track transformer \
     --input-dir training/outputs/transformer_seeds \
     --output training/outputs/transformer_seeds/ensemble_metrics.json`

- Ensemble GNN seeds:
  - `python training/scripts/aggregate_seeds.py --track gnn \
     --input-dir training/outputs/gnn_seeds \
     --output training/outputs/gnn_seeds/ensemble_metrics.json`

- Ensemble Fusion seeds:
  - `python training/scripts/aggregate_seeds.py --track fusion \
     --input-dir training/outputs/fusion_seeds \
     --output training/outputs/fusion_seeds/ensemble_metrics.json`

## Outputs

- Fusion metrics per run: `training/outputs/fusion/<run_id>/metrics.json`
- Ensemble metrics: `<input-dir>/ensemble_metrics.json`

## Notes

- All new flags are additive; defaults maintain existing behavior.
- For non‑A100 GPUs, set `--amp-dtype fp16` or disable AMP as needed.
- Ensure checkpoints exist before launching fusion; otherwise single‑branch inference will run with warnings.
