# Handoff Summary — Stage 4–5 (Fusion + Ensembling)

## Completed in Stages 4–5

- Fusion training (late fusion baseline)
  - File: `training/train_fusion.py`
  - Inputs: `--transformer-checkpoint`, `--gnn-checkpoint`, `--model-name {codebert,graphcodebert,unixcoder}` (for transformer branch), `--freeze-transformer`, `--freeze-gnn`.
  - Architecture: concat(Transformer CLS, GNN pooled) → MLP head with dropout 0.2–0.3; optional light fine‑tuning of encoders.
  - Parity knobs (default‑off): `--scheduler {linear,cosine}`, `--amp-dtype {fp16,bf16}`, `--grad-clip-norm`, `--ema --ema-decay`, `--pad-to-multiple-of`.
  - Metrics parity: writes `metrics.json` with `f1_at_0_5`, `f1_at_best_threshold`, `best_threshold_by_f1`, `balanced_accuracy_at_best_threshold`, confusion matrices, threshold sweep 0.30–0.70.
  - Presets: `fusion_balanced` (head 5–8 epochs, EMA off), `fusion_high_recall` (focal loss in head).

- Seed aggregation / ensembling
  - File: `training/scripts/aggregate_seeds.py`
  - CLI: `--track {transformer,gnn,fusion}` `--input-dir` `--output` `--recompute`
  - Behavior: loads per‑seed validation logits, averages logits, sweeps thresholds (0.30–0.70 step 0.01), outputs `ensemble_metrics.json` with parity keys.
  - Notebook cells: added ensemble runners for all tracks + final comparison cell (per‑seed mean±std vs ensemble F1@best).

## How to Run (examples)

- Fusion quick run (using best single models):
  - `python training/train_fusion.py --transformer-checkpoint training/outputs/transformer/best.ckpt --gnn-checkpoint training/outputs/gnn/best.ckpt --model-name graphcodebert --epochs 6 --batch-size 16 --scheduler cosine --amp-dtype bf16 --grad-clip-norm 1.0 --pad-to-multiple-of 8`

- Ensemble seeds (Transformer):
  - `python training/scripts/aggregate_seeds.py --track transformer --input-dir training/outputs/transformer_seeds --output training/outputs/transformer_seeds/ensemble_metrics.json`

- Ensemble seeds (GNN):
  - `python training/scripts/aggregate_seeds.py --track gnn --input-dir training/outputs/gnn_seeds --output training/outputs/gnn_seeds/ensemble_metrics.json`

- Ensemble seeds (Fusion):
  - `python training/scripts/aggregate_seeds.py --track fusion --input-dir training/outputs/fusion_seeds --output training/outputs/fusion_seeds/ensemble_metrics.json`

## A100 Recommended Flags

- Add `--amp-dtype bf16 --scheduler cosine --grad-clip-norm 1.0 --pad-to-multiple-of 8 --num-workers 4 --prefetch-factor 2`.
- Keep EMA on for stability when training the fusion head: `--ema --ema-decay 0.999`.

## Acceptance Targets

- Fusion: consistent lift over best single model; report absolute and delta F1.
- Ensemble: `f1_at_best_threshold` ≥ single best by +0.5–1.0 points.

## Notes & Rollback

- All new flags are optional; defaults match prior behavior.
- If checkpoints missing, fusion runs a single‑branch path and warns (graceful degrade).
- Rollback to single tracks is immediate: use Transformer or GNN runners as before.
