# Handoff Summary — StreamGuard Training Pipeline (Context Checkpoint)

## Current Progress and Key Decisions

- Transformer (Stages 1–2)
  - Selectable backbones implemented: `--model-name {codebert,graphcodebert,unixcoder}` via `training/models/backbones.py`; integrated in `training/train_transformer.py`.
  - Stability knobs added (all default-off): `--scheduler {linear,cosine}`, `--amp-dtype {fp16,bf16}`, `--grad-clip-norm`, `--pad-to-multiple-of`, `--ema --ema-decay`, DataLoader knobs (`--num-workers`, `--prefetch-factor`, `--persistent-workers`, `--drop-last`).
  - Metrics parity: Transformer writes `metrics.json` with `f1_at_0_5`, `f1_at_best_threshold`, `best_threshold_by_f1`, `balanced_accuracy_at_best_threshold`, and confusion matrices. Threshold sweep: 0.30–0.70 step 0.01.
  - Notebook cells updated to read JSON metrics (not regex) and expose model/scheduler/AMP flags.

- GNN (Stage 3A Parity, 3B CLS + token scaffolding)
  - Preprocessing (`training/preprocessing/create_simple_graph_data.py`): added `--output`, `--max-samples`, `--force`; writes `graphs_metadata.json`.
  - Training (`training/train_gnn.py`): aligned with Transformer (AMP/scheduler/clip/DataLoader knobs), added threshold sweep and `metrics.json` parity, collapse detection preserved.
  - Encoder features (Stage 3B):
    - New `training/preprocessing/augment_graphs_with_encoder.py`: adds GraphCodeBERT CLS embeddings (`graph_cls`) or token embeddings (`node_embeds` when `node_texts` exist).
    - `GraphDataset` loads `graph_cls` (opt-in via `--encoder-features cls`); model concatenates post-pooling features. Token-mode scaffolding: projects precomputed node features when `--encoder-features token`.
  - Notebook: Added GNN preset catalog + runner + production multi-seed runner; appended Stage 3 documentation with commands.

- Repo docs
  - Added `docs/CHANGELOG_STAGE1_TO_STAGE3.md` summarizing all work, flags, metrics, and usage.

## Important Context, Constraints, Preferences

- Defaults must remain unchanged; all new behavior is opt-in via flags/presets.
- Runners and notebook cells should stream logs and read `metrics.json`; avoid regex parsing.
- Outputs belong under `training/outputs/*` and should not be committed.
- A100 recommended settings: `--mixed-precision --amp-dtype bf16`, cosine scheduler, `--grad-clip-norm 1.0`, `--pad-to-multiple-of 8`.
- Prefer clear, small commits; notebooks include brief comments on changes.

## What Remains (Next Steps)

- Stage 4 (Fusion baseline parity and runners)
  - Ensure `training/train_fusion.py` emits parity `metrics.json` (include F1@0.5 and F1@best, threshold, balanced accuracy, confusion matrices).
  - Add fusion preset catalog, preset runner, and production multi-seed cells to `StreamGuard_Production_Training.ipynb` mirroring Transformer/GNN UX.
  - Verify CLI flags parity (support for scheduler, AMP dtype, grad clipping, pad_to_multiple_of, EMA).

- Stage 5 (Seed ensembling and final metrics)
  - Add `training/scripts/aggregate_seeds.py`:
    - `--track {transformer,gnn,fusion}`, `--input-dir`, `--output ensemble_metrics.json`, optional `--recompute`.
    - Average per-seed val logits (or recompute), sweep thresholds 0.30–0.70; write ensemble metrics (F1@best, best threshold, balanced accuracy, confusion matrix).
  - Notebook: Add ensemble runner cells for all tracks and a small “Final Metrics” cell comparing per-seed mean ± std vs ensemble F1@best.

## Critical Data, Examples, References

- Key files
  - Transformer: `training/train_transformer.py`, `training/models/backbones.py`
  - GNN: `training/train_gnn.py`, `training/preprocessing/create_simple_graph_data.py`, `training/preprocessing/augment_graphs_with_encoder.py`
  - Notebook: `StreamGuard_Production_Training.ipynb`
  - Docs: `docs/CHANGELOG_STAGE1_TO_STAGE3.md`

- Metrics JSON schema expectations (all tracks)
  - `best_f1_vulnerable`
  - `f1_at_0_5`
  - `f1_at_best_threshold`
  - `best_threshold_by_f1`
  - `balanced_accuracy_at_best_threshold`
  - `confusion_matrix_threshold_0_5`, `confusion_matrix_best_threshold`
  - `threshold_sweep` list and `timestamp`

- Example commands (quick)
  - Small graphs:
    - `python training/preprocessing/create_simple_graph_data.py --output data/processed/graphs_sample --max-samples 200`
  - GNN quick:
    - `python training/train_gnn.py --train-data data/processed/graphs_sample/train.jsonl --val-data data/processed/graphs_sample/val.jsonl --epochs 2 --batch-size 8 --quick-test`
  - CLS augmentation:
    - `python training/preprocessing/augment_graphs_with_encoder.py --input data/processed/codexglue/train.jsonl --output data/processed/graphs_encoder/train.jsonl --model-name graphcodebert --max-samples 500`
  - GNN+CLS:
    - `python training/train_gnn.py --train-data data/processed/graphs_encoder/train.jsonl --val-data data/processed/graphs_encoder/val.jsonl --encoder-features cls --epochs 2 --batch-size 8 --quick-test`

- A100 recommended switches
  - Add `--mixed-precision --amp-dtype bf16 --scheduler cosine --grad-clip-norm 1.0 --pad-to-multiple-of 8 --num-workers 4 --prefetch-factor 2` to CLI commands.

## Final Notes

- Fusion runners and ensemble scripts are the only major pieces remaining to fully close Stages 4–5.
- Keep using metrics JSON across runners; ensure fusion emits the same keys for downstream aggregation.
- Token mode is enabled only if graphs carry `node_texts` and augmentation writes `node_embeds`. CLS mode is recommended and stable.
