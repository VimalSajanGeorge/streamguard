# Next Steps Plan — Novel Features (No Explainability)

## Scope
- Goal: Improve F1 using structural code signals and GraphCodeBERT, without adding explainability.
- Tracks: taint‑lite graphs, node encoder features, GNN upgrades, SQL intent upgrades, fusion refinements.
- Constraint: Defaults unchanged; all new behavior is opt‑in via flags/presets.

## Taint Flow Feasibility & Risks
- Feasible (lite) by constraining to simple Python rules:
  - Sources: request/form params, `input()`, env vars, framework request objects.
  - Sinks: DB executes (`execute`, `executemany`), SQL string patterns.
  - Flow: def‑use chains and string propagation (concat, `%`, `.format`, f‑strings).
- Risks
  - Parser fragility across languages; noisy heuristics lead to false positives.
  - Label drift if taint edges dominate and are low‑precision.
  - Preprocessing cost on large datasets.
- Mitigation
  - Start with Python only and keep taint edges additive (baseline edges remain).
  - Gate behind `--use-taint-lite`; graceful fallback when parsing fails; log coverage.

## Track A — Taint‑Lite Graphs (additive)
- Add `training/preprocessing/create_taint_graphs.py`:
  - Parse code → tokens/AST via lightweight parser (pluggable backend).
  - Build edges: def‑use, call→arg, string concat; mark `source_mask`, `sink_mask`, and `edge_types`.
  - Output JSONL: `ast_nodes`, `edge_index`, `edge_types`, optional `node_texts`.
- Add `training/utils/taint_rules.py`:
  - Regex/rule lists for sources, sinks, sanitizers.
- Update `training/train_gnn.py`:
  - Flags: `--use-taint-lite`, `--edge-types on/off`.
  - If provided, encode `edge_types` and masks; otherwise ignore.

## Track B — Node Encoder Features (token mode)
- Ensure `training/preprocessing/create_taint_graphs.py` emits `node_texts` (canonical snippet per node).
- Extend `training/preprocessing/augment_graphs_with_encoder.py` (existing):
  - When `--embedding-type token` and `node_texts` exist, write `node_embeds`.
- Update `training/train_gnn.py`:
  - Flag: `--encoder-features token` to load `node_embeds` and project via linear layer (+dropout).
  - Strict shape checks; on mismatch skip sample or zero‑pad with warning.
  - Keep CLS mode as default; token mode opt‑in after coverage >70%.

## Track C — GNN Architecture Upgrades (toggleable)
- Add `training/models/gnn_layers.py`:
  - R‑GCN (edge types), GATv2 with edge gate, Jumping Knowledge, optional virtual node.
- Update `training/train_gnn.py`:
  - Flags: `--gnn-arch {sage,gatv2,rgcn}`, `--jk on/off`, `--virtual-node on/off`, `--pool {mean,max,attn}`.
  - Defaults unchanged; presets enable new options.

## Track D — SQL Intent Transformer Upgrades (no XAI)
- Add `core/features/sql_stats.py`:
  - Features: template style (f‑string/%/.format/+), SQL keyword counts, presence of literals (e.g., `OR 1=1`), placeholder coverage, driver API usage.
- Update `training/train_transformer.py`:
  - Flag: `--use-sql-stats` to append normalized SQL stats to CLS projection.
  - Optional multi‑task: `--intent-head on`, `--intent-alpha 0.1` to predict coarse intent {select, insert, update, delete, other}.

## Track E — Fusion Refinements
- Update `training/train_fusion.py`:
  - Flag: `--cross-attn on` to add shallow cross‑attention between Transformer CLS (query) and GNN pooled (key/value), then concat → MLP head.
  - Flag: `--freeze-head-epochs N` to warm up the head before unfreezing encoders.

## Presets
- GNN
  - `taint_lite_cls`: taint‑lite graphs + GraphCodeBERT CLS features; edge types off.
  - `taint_lite_rgcn`: taint‑lite + edge types on + R‑GCN + JK; epochs 12–15, cosine, EMA.
  - `token_mode_smoke`: `--encoder-features token` on a small subset to validate shapes.
- Transformer
  - `sql_stats`: `--use-sql-stats` + GraphCodeBERT + cosine + EMA.
  - `sql_multitask`: add `--intent-head on --intent-alpha 0.1`.
- Fusion
  - `fusion_xattn`: `--cross-attn on`, freeze encoders initially, head 6–8 epochs.

## Validation & Acceptance
- Keep metrics parity and threshold sweep across tracks.
- Ablations
  - Taint edges on/off (same data), edge types on/off with R‑GCN vs GATv2.
  - SQL stats on/off vs baseline transformer.
  - CLS vs token‑mode on the same graph subset.
- Targets
  - GNN: +0.3–0.8 F1@best when taint‑lite coverage >60%.
  - Transformer: +0.2–0.5 F1@best with `--use-sql-stats` on SQL‑heavy data.
  - Fusion cross‑attention: consistent +0.1–0.3 over late concat.

## Risks & Mitigation
- Taint parser failures → try/catch per sample; skip with log; fallback to baseline graphs.
- Node alignment drift → strict count checks; skip or zero‑pad; never crash a run.
- Overfitting to heuristics → keep class weights/label smoothing; enable focal loss only when recall lags.
- Compute costs → stage features (CLS full set first; token mode on subset; R‑GCN only with good edge coverage).

## File Worklist
- Add
  - `training/preprocessing/create_taint_graphs.py`
  - `training/utils/taint_rules.py`
  - `training/models/gnn_layers.py`
  - `core/features/sql_stats.py`
- Edit
  - `training/preprocessing/augment_graphs_with_encoder.py`
  - `training/train_gnn.py`
  - `training/train_transformer.py`
  - `training/train_fusion.py`
  - `StreamGuard_Production_Training.ipynb`

## Execution Order (safe)
1) Implement taint‑lite generator + rules; run on a small split and log coverage.
2) Enable CLS‑mode GNN on taint‑lite graphs; compare vs baseline.
3) Add GNN arch flags and R‑GCN preset; ablate edge types/JK.
4) Add SQL stats to transformer; A/B on SQL‑heavy subset.
5) Add fusion cross‑attention; short fine‑tuning with frozen encoders.
