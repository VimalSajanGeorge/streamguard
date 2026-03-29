# StreamGuard — Phase 4: Full Model + Training
## Production Risk Analysis Document

*What Can Go Wrong, How Bad It Is, and What You Must Do About It*

| **Version** | 1.0 — March 2026 |
|-------------|-----------------|
| **Status** | ACTIVE — Read before writing a single line of training code |
| **Risks Catalogued** | 32 total: 8 Critical, 12 High, 8 Medium, 4 Low |
| **Tone** | Blunt. No sugar-coating. |

---

## 1. Executive Summary

> **Phase 4 is the phase where months of data collection and preprocessing either pay off or get wasted.**
>
> Every risk in this document is a real failure mode that will happen during a 30-hour GPU training run across 5 ablation configs. Some are paper-invalidating. Some are week-long detours. A few are simple two-hour fixes that only feel catastrophic because they're discovered after training finishes.
>
> The single most expensive mistake you can make in Phase 4 is touching the test set during training. Reviewers at ISSTA and ICSE specifically look for test contamination. If it happens, every number in the paper must be regenerated.
>
> Read this document completely. Implement all mitigations before starting any full training run. Every checkpoint exists because someone didn't.

| **Category** | **Count** | **Worst Case If All Ignored** |
|-------------|----------|-------------------------------|
| CRITICAL | 8 | Research paper invalid. All F1 results wrong. Full re-train required. |
| HIGH | 12 | 1–5 days of wasted GPU time per incident. F1 targets missed. |
| MEDIUM | 8 | Model underperforms target. Ablation results inconclusive. |
| LOW | 4 | Hours of cleanup. No fundamental impact. |

---

## 2. Master Risk Register

| **ID** | **Risk Title** | **Story** | **Severity** | **Prob.** | **Worst-Case Impact** |
|--------|---------------|-----------|-------------|----------|----------------------|
| **R-01** | Test set evaluated before training ends — F1 inflated | P4-S5 | **CRITICAL** | MEDIUM | All paper results invalid. Full re-train. |
| **R-02** | Wrong L_CFA sign: relu(sim − 0.5) instead of relu(sim + 0.5) | P4-S4 | **CRITICAL** | MEDIUM | CFA benefit disappears. Core paper claim fails. |
| **R-03** | BatchNorm1d instead of GroupNorm — model breaks at batch_size=1 during serving | P4-S1 | **CRITICAL** | HIGH | Serving gives wrong predictions for every single function |
| **R-04** | CFA pairs split across batches — L_CFA trains on random pairs | P4-S3 | **CRITICAL** | MEDIUM | CFA loss trains on noise. Config C cannot beat Config B. |
| **R-05** | Different train.h5 used across ablation configs — results incomparable | P4-S7 | **CRITICAL** | LOW | Paper reviewers will invalidate ablation table. |
| **R-06** | Different seeds across ablation configs — results not reproducible | P4-S7 | **CRITICAL** | LOW | Reviewers cannot reproduce paper results. Rejection. |
| **R-07** | CodeBERT tokenization skipped — model trains without sequence signal | P4-S5 | **CRITICAL** | MEDIUM | Model is graph-only for all configs. N2 (CodeBERT+CFA) contribution is zero. |
| **R-08** | No checkpoint saved every epoch — GPU crash loses all progress | P4-S5 | **CRITICAL** | HIGH | 15–20 hours of training lost. Full restart required. |
| **R-09** | GPU OOM crash mid-epoch — training halts with no partial save | P4-S5 | **HIGH** | HIGH | Re-run from last checkpoint (up to 1 epoch lost) |
| **R-10** | NaN loss after epoch 1 — gradient explosion | P4-S5 | **HIGH** | MEDIUM | Training diverges. Re-run with lower LR or clipping |
| **R-11** | CodeBERT catastrophic forgetting — LR too high for pre-trained weights | P4-S5 | **HIGH** | MEDIUM | BERT loses pre-trained knowledge. F1 drops to ~0.5 |
| **R-12** | Differential LR not applied — CodeBERT and GGNN use same LR | P4-S5 | **HIGH** | MEDIUM | CodeBERT fine-tunes too aggressively. F1 degrades. |
| **R-13** | MLflow not running — no experiment tracking | P4-S5 | **HIGH** | MEDIUM | Can't reproduce or compare ablation runs. Paper reproducibility fails. |
| **R-14** | AMP (fp16) disabled on GPU — OOM on 8GB VRAM | P4-S5 | **HIGH** | HIGH | Training crashes in first batch on most consumer GPUs |
| **R-15** | type_aware_edges=False not implemented for Config B — all ablations use same type-aware arch | P4-S7 | **HIGH** | MEDIUM | Config B doesn't actually test type-blind baseline. B vs B' comparison invalid. |
| **R-16** | Config A (no graph) not implemented — uses full model with empty graph | P4-S7 | **HIGH** | MEDIUM | Config A doesn't test the correct hypothesis. Paper baseline wrong. |
| **R-17** | Edge type ≥ 4 in HDF5 (CDG leak) — CUDA OOB in GatedGraphConv | P4-S1 | **HIGH** | LOW | Training crashes with cryptic CUDA error. Stage 4 re-run needed. |
| **R-18** | pair_id missing from HDF5 attrs — all samples treated as singletons | P4-S3 | **HIGH** | LOW | L_CFA trains on random pairs. Already caught by Phase 3 Gate G-08 — verify. |
| **R-19** | Code text missing from metadata — tokenization gets empty strings | P4-S5 | **HIGH** | MEDIUM | CodeBERT gets padding tokens for every sample. Sequence signal = zero. |
| **R-20** | Gradient accumulation step count off-by-one — scheduler misaligned | P4-S5 | **MEDIUM** | MEDIUM | Learning rate schedule wrong — performance slightly degraded |
| **R-21** | pairwise_accuracy stuck at 0.0 — cfa_batch always None | P4-S6 | **MEDIUM** | MEDIUM | Core VISION metric shows 0 — paper metrics incomplete |
| **R-22** | Per-CWE F1 skips all CWEs (too-strict min-sample threshold) | P4-S6 | **MEDIUM** | MEDIUM | Worst-group F1 metric missing from paper Table 3 |
| **R-23** | Severity head trained on all-zero labels (-1 not filtered) | P4-S4 | **MEDIUM** | MEDIUM | L_severity wastes gradient signal — model learns wrong severity |
| **R-24** | Config E inter-proc crashes with None callee_embeddings | P4-S2 | **MEDIUM** | MEDIUM | Config E ablation cannot complete |
| **R-25** | Checkpoint not atomic — partial file on crash corrupts best model | P4-S5 | **MEDIUM** | MEDIUM | Best checkpoint is unreadable. Re-train from scratch or use epoch checkpoint. |
| **R-26** | CWE_LABEL_MAP missing from eval.py — per-CWE F1 always empty | P4-S6 | **MEDIUM** | LOW | Per-CWE table missing from results. Easy to add but requires re-evaluation. |
| **R-27** | torch.compile breaks scatter ops — training crashes in cross-attention | P4-S1 | **MEDIUM** | LOW | Disable compile if crash occurs — performance but not correctness issue |
| **R-28** | Early stopping patience too low — model stops before convergence | P4-S5 | **LOW** | MEDIUM | Suboptimal F1. Re-train with higher patience. |
| **R-29** | val.h5 and test.h5 paths swapped in config — test evaluated during training | P4-S5 | **LOW** | LOW | Same as R-01 if test.h5 path is accidentally used for validation |
| **R-30** | callee_summarizer.py not initialized for Config E — use_interproc=True but no CalleeSummarizer | P4-S2 | **LOW** | LOW | Config E produces same result as Config D (no inter-proc signal) |
| **R-31** | freeze_codebert_layers set too high — top layers frozen, model can't learn | P4-S1 | **LOW** | LOW | F1 plateaus below Config B. Easy to fix — unfreeze and re-train. |
| **R-32** | MLflow run name collision — two configs log to same run name | P4-S7 | **LOW** | LOW | Ablation comparison unclear in MLflow UI. Rename and re-log. |

---

## 3. Critical Risks — Full Detail

### R-01: Test Set Contamination (The Career Risk)

> **This is the single most dangerous risk in the entire project.**

The test set exists for ONE purpose: reporting final numbers in the paper. It must never be used for:
- Early stopping decisions
- Hyperparameter tuning
- Model selection between checkpoints
- Checking "how training is going"

If test.h5 is evaluated during training and those numbers influence ANY decision (even informally), the final reported F1 is inflated. The model has effectively seen the test set. ISSTA and ICSE reviewers run their own deduplication checks and have seen this exact issue in the papers they reject.

**How it accidentally happens:**
```python
# WRONG: using test loader for early stopping
val_metrics = evaluate(model, test_loader, device)  # ← test_loader, not val_loader
if val_metrics["f1"] > best_val_f1:
    save_checkpoint(...)  # model selected based on test performance

# WRONG: accidentally pointing val_h5 at test.h5
TRAINING_CONFIG = {
    "val_h5":  "training/data/final/test.h5",  # ← typo that invalidates everything
    "test_h5": "training/data/final/test.h5",
}
```

**Correct approach:**
```
train.h5  → Training loss computation, weight updates
val.h5    → All intermediate evaluation, early stopping, model selection
test.h5   → ONE evaluation call, after training loop exits, on BEST val checkpoint
```

**Verification:**
```bash
# After training, check MLflow: how many times was test_f1 logged?
python3 -c "
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['1'])
for run in runs:
    test_metrics = [k for k in run.data.metrics if k.startswith('test_')]
    val_metrics_by_epoch = [k for k in run.data.metrics if 'val_f1' in k]
    print(f'Run: {run.info.run_name}')
    print(f'  test metrics logged: {len(test_metrics)} (should be 1 per metric)')
    print(f'  val_f1 evaluations: {len(val_metrics_by_epoch)} (should equal num epochs)')
"
```

**Mitigation status:** Mitigated in P4-S5 train.py by:
1. Only creating `test_loader` AFTER the training loop exits
2. Final test evaluation uses `best_{config}.pt` checkpoint, not the current model state

---

### R-02: Wrong L_CFA Sign (The Silent Research Killer)

This is the subtlest risk in Phase 4. The math looks similar but the effect is completely different.

**The formula:**
```
Goal: push emb_v (vulnerable) and emb_v' (counterfactual safe) APART
      cosine_similarity should be NEGATIVE (opposite hemispheres)
      Penalize when cosine_sim > −0.5

CORRECT formula: relu( cosine_sim + 0.5 )
  - When cosine_sim = +0.9: relu(0.9 + 0.5) = 1.4  (LARGE penalty — not separated)
  - When cosine_sim = −0.3: relu(−0.3 + 0.5) = 0.2 (small penalty — almost there)
  - When cosine_sim = −0.8: relu(−0.8 + 0.5) = 0.0 (no penalty — correctly separated)

WRONG formula: relu( cosine_sim − 0.5 )
  - When cosine_sim = +0.9: relu(0.9 − 0.5) = 0.4  (SMALL penalty — nearly useless)
  - When cosine_sim = −0.3: relu(−0.3 − 0.5) = 0.0 (NO penalty — when it should fire!)
  - When cosine_sim = −0.8: relu(−0.8 − 0.5) = 0.0 (no penalty — correct)
```

The wrong formula only fires when embeddings are *extremely* similar (cosine sim > 0.5). It does almost nothing to enforce the antipodal separation that the paper claims. The model will converge on L_CE alone while L_CFA barely participates.

**Detection:**
```python
# Run this unit test BEFORE any training:
import torch, torch.nn.functional as F

emb_v  = torch.randn(4, 128); emb_v  = F.normalize(emb_v,  dim=-1)
emb_vp = -emb_v + torch.randn(4, 128) * 0.1  # nearly opposite embeddings
emb_vp = F.normalize(emb_vp, dim=-1)

cosine_sim = F.cosine_similarity(emb_v, emb_vp, dim=-1)
print(f"Cosine similarity (near-opposite pairs): {cosine_sim.mean():.3f}")  # ~-0.9

# CORRECT formula should give ~0.0 loss for well-separated pairs
loss_correct = F.relu(cosine_sim + 0.5).mean()  # ~0.0 ✓
loss_wrong   = F.relu(cosine_sim - 0.5).mean()  # also ~0.0

# CORRECT formula should give HIGH loss for similar pairs
emb_same = emb_v + torch.randn(4, 128) * 0.01
emb_same = F.normalize(emb_same, dim=-1)
cosine_sim_same = F.cosine_similarity(emb_v, emb_same, dim=-1)  # ~0.99

loss_correct_same = F.relu(cosine_sim_same + 0.5).mean()  # ~1.49 ✓
loss_wrong_same   = F.relu(cosine_sim_same - 0.5).mean()  # ~0.49 ✗ (too weak)

print(f"L_CFA (correct, similar pairs): {loss_correct_same:.3f}")   # ~1.49
print(f"L_CFA (wrong,   similar pairs): {loss_wrong_same:.3f}")     # ~0.49
```

**Mitigation:** The unit test above is in `tests/test_p4s4_losses.py`. Run it before any training.

---

### R-03: BatchNorm1d at Serving Time

This risk is unique in that it does NOT appear during training. Training uses batch_size=8, so BatchNorm statistics are meaningful. The problem only surfaces when Phase 6 serving sends one function at a time.

**What happens with BatchNorm1d at batch_size=1:**
```
BatchNorm: normalize x using batch mean and variance
At batch_size=1: mean = x itself, variance = 0
→ (x - x) / sqrt(0 + epsilon) = 0 / epsilon ≈ 0 (tiny non-zero value)
→ ALL activations collapse toward zero regardless of input
→ Model outputs near-constant values for every function
→ Every prediction is the same confidence (0.5 or the bias term)
```

**Training F1 may be 0.93. Serving F1 will be ~0.50** (random chance for binary).

**Check:**
```python
from model import StreamGuardModel
import torch.nn as nn

model = StreamGuardModel()
for i, norm in enumerate(model.ggnn_norm):
    assert isinstance(norm, nn.GroupNorm), \
        f"Layer {i} uses {type(norm).__name__} — must be GroupNorm!"
    print(f"Layer {i}: GroupNorm(groups={norm.num_groups}, channels={norm.num_channels}) ✓")
```

**Mitigation:** GroupNorm(32, 256) used throughout. Verified in test_p4s1_model.py.

---

### R-04: CFA Pairs Split Across Batches

This is the operational failure of the research contribution.

```
What the code is supposed to do:
  Batch 1: [vuln_a, cfa_a, vuln_b, cfa_b]  ← L_CFA(emb_vuln_a, emb_cfa_a) ✓

What happens if sampler is wrong:
  Batch 1: [vuln_a, vuln_b, cfa_c, singleton_d]
  Batch 2: [cfa_a, vuln_e, cfa_b, singleton_f]
  
  Collate sees: pair_id_a in Batch 1 → orig; pair_id_a in Batch 2 → missing!
  L_CFA computes: cosine_sim(emb_vuln_a, emb_cfa_c)  ← WRONG PAIR
  Training pushes vuln_a away from cfa_c (unrelated function)
  Model learns nothing useful from L_CFA
```

**Detection:**
```python
from cfa_dataloader import build_dataloader
from collections import defaultdict

loader = build_dataloader("training/data/final/train.h5", "train", batch_size=8)
pair_seen_in_batches = defaultdict(set)

for batch_idx, (orig, cfa, o_meta, c_meta) in enumerate(loader):
    all_meta = list(o_meta) + (list(c_meta) if c_meta else [])
    for m in all_meta:
        if m.get("pair_id"):
            pair_seen_in_batches[m["pair_id"]].add(batch_idx)
    if batch_idx >= 19:  # check first 20 batches
        break

split_pairs = {p: b for p, b in pair_seen_in_batches.items() if len(b) > 1}
if split_pairs:
    raise AssertionError(f"CRITICAL: {len(split_pairs)} pairs split across batches!")
print(f"PASS: All pairs in same batch ({len(pair_seen_in_batches)} pairs checked)")
```

**Mitigation:** CFAAwareBatchSampler groups by pair_id BEFORE shuffling. Verified in test_p4s3_dataloader.py.

---

### R-05 & R-06: Ablation Configs Use Different Data or Seeds

The ablation table is the core of the paper. If Config B and Config C used different splits or seeds, the comparison is meaningless. A reviewer who implements Config B independently and gets different numbers will reject the paper.

**Anti-pattern to avoid:**
```python
# WRONG: each config re-splits the data
for config_name, config in ABLATION_CONFIGS.items():
    # Accidentally re-running Stage 7 split between configs
    split_data(all_data, seed=config["seed"])  # ← different seeds → different splits
    train(config)
```

**Correct approach:**
```python
# One canonical split, produced once by Stage 7, used for ALL configs
BASE_CONFIG = {
    "train_h5": "training/data/final/train.h5",  # SAME FOR ALL
    "val_h5":   "training/data/final/val.h5",    # SAME FOR ALL
    "test_h5":  "training/data/final/test.h5",   # SAME FOR ALL
    "seed":     42,                              # SAME FOR ALL
}

# Ablation runner asserts this at startup:
for name, cfg in ABLATION_CONFIGS.items():
    assert cfg["train_h5"] == BASE_CONFIG["train_h5"], f"{name} uses different train.h5!"
    assert cfg["seed"]     == BASE_CONFIG["seed"],     f"{name} uses different seed!"
```

---

### R-07: CodeBERT Tokenization Skipped

The model has two encoders: graph (GGNN) and sequence (CodeBERT). If tokenization is not called inside the training loop, `input_ids=None` falls through to the zero-vector fallback:

```python
# In model.forward():
if input_ids is not None:
    bert_cls = self.encode_sequence(input_ids, attention_mask)  # (B, 768) — real signal
else:
    B = graph_embed.size(0)
    bert_cls = torch.zeros(B, 768, ...)  # (B, 768) — zeros — NO SIGNAL
```

If tokenization is skipped, the model trains with zero-vector BERT input for all configs, including Config A (CodeBERT sequence only). Config A would produce the same result as a GGNN with no sequence encoder at all.

**Detection:**
```python
# Add this assertion in the training loop after tokenization:
assert input_ids is not None, "Tokenization failed — input_ids is None!"
assert not (input_ids == 0).all(), "All padding tokens — code text may be empty!"
assert input_ids.shape == (B, 512), f"Wrong shape: {input_ids.shape}"
```

**Mitigation:** The metadata dict from the DataLoader must include `"code"` key. If HDF5 doesn't store code text, load `{sample_id: code}` lookup dict from the Stage 3 JSONL at DataLoader init time.

---

### R-08: No Per-Epoch Checkpoint

Training 20 epochs on an A100 takes ~6 hours. On a T4 (Colab), ~18 hours. Colab sessions disconnect after 12 hours. Without a per-epoch checkpoint:

- Session disconnects at epoch 12 of 20
- All 12 epochs of training are lost
- Must restart from epoch 0

```python
# WRONG: only save when val_F1 improves
if val_metrics["f1"] > best_val_f1:
    save_checkpoint(...)  # never saved if F1 never exceeds initial val

# CORRECT: save EVERY epoch (for resume), PLUS save best separately
# Every epoch:
torch.save(latest_state, resume_path + ".tmp")
os.replace(resume_path + ".tmp", resume_path)

# When F1 improves:
save_checkpoint(model, optimizer, epoch, metrics, config, best_path)
```

---

## 4. High Risks — Full Detail

### R-09: GPU OOM During Training

OOM crashes are the most common Phase 4 failure. They happen when a batch contains a large graph (200 nodes × 824 features × 8 batch = 1.3M floats just for features).

**OOM hierarchy — try in order:**

| Step | Change | VRAM Saved | Speed Cost |
|------|--------|-----------|------------|
| 1 | Enable AMP (use_amp=True) | ~40% | ~5% faster |
| 2 | Reduce batch_size=4, accum=8 | ~50% | ~0% (same effective batch) |
| 3 | freeze_codebert_layers=9 | ~30% activations | ~20% faster |
| 4 | gradient_checkpointing_enable() | ~25% | ~15% slower |
| 5 | max_cpg_nodes=150 (re-run Stage 4) | ~25% | Stage 4 re-run needed |

**OOM diagnostic:**
```bash
# Before training, estimate peak memory:
python3 -c "
import torch
from torch_geometric.data import Data, Batch
from model import StreamGuardModel

model = StreamGuardModel().cuda()
# Simulate worst-case batch: 8 graphs × 200 nodes × 824 features
graphs = [Data(
    x=torch.randn(200, 824),
    edge_index=torch.randint(0, 200, (2, 400)),
    edge_attr=torch.randint(0, 4, (400,)),
    y=torch.tensor([1])
) for _ in range(8)]
batch = Batch.from_data_list(graphs).cuda()

torch.cuda.reset_peak_memory_stats()
with torch.cuda.amp.autocast():
    out = model(batch)
    out['logits'].sum().backward()

peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f'Peak VRAM: {peak_mb:.0f} MB')
print(f'Available: {torch.cuda.get_device_properties(0).total_memory / 1e6:.0f} MB')
"
```

### R-10: NaN Loss After Epoch 1

NaN propagates instantly through all subsequent computations. Once loss is NaN, all gradients are NaN, all weights are NaN, the model is destroyed.

**Most common causes:**

1. **Learning rate too high for CodeBERT**: 2e-4 → NaN in 2 epochs. Keep CodeBERT at 2e-5.
2. **Missing gradient clipping**: A single large gradient spike propagates backward through 12 GGNN layers. Always clip at max_norm=1.0.
3. **NaN in input features**: If HDF5 features have NaN (Stage 6 bug), they propagate through the model.
4. **Division by zero in scatter softmax**: If a graph has 0 nodes (Stage 6 Gate 1 bug), `sum_exp[batch]` = 0.

**Pre-training NaN check:**
```python
import h5py, numpy as np

with h5py.File("training/data/final/train.h5", "r") as f:
    nan_count = 0
    for sid in list(f.keys())[:1000]:  # check first 1000
        if np.isnan(f[sid]["x"][:]).any():
            nan_count += 1
    print(f"NaN features in first 1000 samples: {nan_count}")
    assert nan_count == 0, "NaN in training data — fix Stage 6!"
```

**Recovery:** If NaN detected during training, restore last clean checkpoint and lower `lr_codebert` to 1e-5.

### R-11: CodeBERT Catastrophic Forgetting

CodeBERT was pre-trained on 6 million code files. It encodes rich contextual knowledge about C syntax, variable names, function patterns. If fine-tuned with too high a learning rate, it forgets this knowledge within 2–3 epochs and becomes no better than a randomly initialized encoder.

**Symptom:** val_F1 improves for 2 epochs, then collapses to ~0.50.

**Signs:**
- `lr_codebert` set to 1e-4 or higher (same as GGNN LR)
- No `freeze_codebert_layers` during warmup

**Fix:** Use differential LR: CodeBERT at 2e-5, everything else at 1e-4. Optionally, freeze CodeBERT layers 0–8 for the first 3 epochs, then unfreeze.

```python
# Verify differential LR in optimizer:
for i, group in enumerate(optimizer.param_groups):
    print(f"Param group {i}: LR={group['lr']}, params={len(group['params'])}")
# Expected:
# Param group 0: LR=2e-05, params=199  (CodeBERT)
# Param group 1: LR=0.0001, params=67  (GGNN + fusion + heads)
```

### R-13: MLflow Not Running

All ablation results must be logged to MLflow for reproducibility. If MLflow is not running when training starts, the run silently fails to log. The experiment is not reproducible from params alone.

**Pre-flight check:**
```bash
# Start MLflow UI before training
mlflow ui --host 0.0.0.0 --port 5000 &
# Verify accessible:
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# In training config:
mlflow.set_tracking_uri("http://localhost:5000")
# OR use file-based backend (no server needed):
mlflow.set_tracking_uri("file:./mlruns")  # safer for long runs
```

**Fallback:** Always add CSV logging as backup alongside MLflow:
```python
import csv, os
def log_metrics_csv(metrics, epoch, config_name, csv_path="results/training_log.csv"):
    row = {"epoch": epoch, "config": config_name, **metrics}
    exists = os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: writer.writeheader()
        writer.writerow(row)
```

### R-15 & R-16: Config B and A Not Properly Implemented

The ablation table is only meaningful if each config tests a clearly different hypothesis.

**Config B (type-blind)** requires a genuinely different architecture — a single GatedGraphConv that sees all edge types together, ignoring the type label. If the StreamGuardModel's `type_aware_edges=False` path just uses the first of the 4 type-specific convolutions and ignores the rest, that's a different kind of wrong (it only sees AST edges). The correct type-blind behavior:

```python
# WRONG (not type-blind — only AST):
if not self.type_aware_edges:
    h = self.edge_type_convs[layer_idx][0](h, edge_index)  # AST only!

# CORRECT (type-blind — all edges, no type distinction):
if not self.type_aware_edges:
    h = self.single_conv[layer_idx](h, edge_index)  # ignores edge_attr
    # ALL edges processed equally regardless of type
```

**Config A (no graph)** requires removing the graph encoder entirely, not just passing zeros. The cross-attention fusion must also be bypassed. The simplest correct implementation:

```python
# Config A: BERT sequence only
if not self.use_graph:
    # No graph encoding — use only CodeBERT representation
    # Fused = BERT_768 repeated 3 times to match 1280-d MLP input
    # OR use a separate A-config MLP that takes 768-d input
    assert input_ids is not None, "Config A requires tokenization"
    shared = self.config_a_mlp(bert_cls)  # separate MLP: 768 → 512 → 128
    # skip all GGNN and cross-attention
```

### R-19: Code Text Missing from Metadata

The HDF5 stores graph tensors but may not store the raw C function code. The training loop needs the code text to call the tokenizer. If code is missing:

```python
codes = [m.get("code", "") for m in orig_metas]
# If all empty: codes = ["", "", "", "", "", "", "", ""]
input_ids, attn_mask = tokenize_batch(codes, tokenizer, 512, device)
# tokenize_batch replaces "" with "void placeholder(){}" 
# → All functions get the SAME placeholder token sequence
# → bert_cls is identical for ALL samples in ALL batches
# → CodeBERT contributes only a constant offset, no discriminative signal
```

**Solutions (in priority order):**

1. **Best:** Re-run Stage 6 to store code as HDF5 attr: `grp.attrs["code"] = sample.get("code", "")[:8000]`
2. **Good:** Load `{sample_id: code}` lookup dict at DataLoader init time from JSONL
3. **Fallback:** Accept degraded performance; report in paper that code stored separately

**Detection:**
```python
import h5py
with h5py.File("training/data/final/train.h5","r") as f:
    sample = list(f.keys())[0]
    has_code = "code" in f[sample].attrs
    print(f"Code in HDF5: {has_code}")
    if not has_code:
        print("WARNING: Code not stored in HDF5. DataLoader needs JSONL lookup.")
```

---

## 5. Medium Risks — Summary

| **Risk** | **Root Cause** | **Impact** | **Fix** |
|----------|----------------|-----------|---------|
| R-20: Scheduler misalignment | `(step+1) % grad_accum` off-by-one | LR schedule slightly wrong | Verify `scheduler.step()` called exactly `total_steps` times |
| R-21: pairwise_accuracy = 0.0 | cfa_batch is always None in eval | Core metric missing | Verify val.h5 has CFA pairs; check pair_id attrs |
| R-22: Per-CWE F1 empty | min_samples threshold too high | Worst-group metric missing | Lower threshold to 5; add per-CWE count to logs |
| R-23: Severity loss on -1 labels | Missing `severity >= 0` filter | Loss wastes gradient | Add `valid_mask = severity_labels >= 0` before HuberLoss |
| R-24: Config E crashes on None callees | CalleeSummarizer not initialized | Config E ablation fails | Add zero-vector fallback in _prepare_interproc_features() |
| R-25: Partial checkpoint on kill -9 | Non-atomic write | Best model corrupted | Always write to .tmp then os.replace() |
| R-26: CWE_LABEL_MAP missing in eval | Import not added to eval.py | CWE accuracy = 0 | Import from losses.py; one-liner fix |
| R-27: torch.compile breaks scatter | PyTorch 2.2 compile + torch_scatter | Crash in cross-attention | Disable compile; test without it first |

---

## 6. The Five Things Most Likely to Waste Your GPU Time

In order of probability:

### 1. GPU OOM on First Full Training Batch (P = ~70% without AMP)

AMP is mandatory on GPUs with less than 16 GB VRAM. Do this FIRST, before debugging anything else.

```python
# Add to the very top of train():
use_amp = device.type == "cuda"
scaler  = torch.cuda.amp.GradScaler() if use_amp else None
```

### 2. CFA Loss Not Decreasing After Epoch 3 (P = ~40% if pair integrity not verified)

If L_CFA is flat or increasing after epoch 3, pairs are likely split. Run the batch-pair verification test immediately.

```python
# Monitor: if L_CFA is > L_CE after epoch 5, something is wrong
# Correct behavior: L_CFA starts high, decreases as pairs separate
```

### 3. All Val F1 Values Below 0.6 After Epoch 2 (P = ~30% without LR differential)

If val_F1 < 0.6 after 2 epochs of training on 30K+ samples, CodeBERT is likely being destroyed by too-high LR. Check:
```python
for g in optimizer.param_groups:
    print(g["lr"])  # must show TWO different values: 2e-5 and 1e-4
```

### 4. NaN Loss After Epoch 1 (P = ~20% without gradient clipping)

```python
# Always clip gradients BEFORE optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5. Config C F1 == Config B F1 (CFA doesn't work) (P = ~15% if data issues exist)

If the ablation table shows Config C ≤ Config B, check these in order:
1. Are CFA pairs actually in train.h5? (`pair_id` attr present and non-empty)
2. Are pairs in the same batch? (Run batch-pair verification test)
3. Is L_CFA decreasing? (Check MLflow — L_CFA should start ~0.7 and decrease)
4. Is the L_CFA sign correct? (Run `relu(0.8 + 0.5)` test)

---

## 7. Risk Mitigation Implementation Tracker

Use this as a pre-flight checklist. All items must be verified before starting any full training run.

| **Risk** | **Mitigation** | **Implemented In** | **Verify Before** |
|----------|---------------|-------------------|------------------|
| R-01: Test contamination | test.h5 evaluated once at end only | P4-S5 train.py | Starting any training run |
| R-02: Wrong L_CFA sign | Unit test relu(sim + 0.5) | P4-S4 tests | First training batch |
| R-03: BatchNorm instead of GroupNorm | GroupNorm(32, 256) throughout | P4-S1 model.py | Model smoke test |
| R-04: Pairs split across batches | CFAAwareBatchSampler + batch verification | P4-S3 dataloader.py | First dry-run |
| R-05/06: Different data/seeds | Assertion in ablation runner | P4-S7 run_ablations.py | Ablation dry-run |
| R-07: Tokenization skipped | Assert input_ids not None | P4-S5 train.py | First training batch |
| R-08: No checkpoint every epoch | Atomic save to resume_path each epoch | P4-S5 train.py | Long runs |
| R-09: GPU OOM | AMP enabled by default | P4-S5 train.py | GPU memory check |
| R-10: NaN loss | gradient clip + pre-training NaN check | P4-S5 train.py | Before full run |
| R-11: CodeBERT forgetting | Differential LR (2e-5 vs 1e-4) | P4-S5 build_optimizer() | LR verification |
| R-12: Same LR for all params | param_groups with different LRs | P4-S5 train.py | Optimizer inspection |
| R-13: MLflow not running | File-based backend as default | P4-S5 train.py | Before full run |
| R-14: No AMP | use_amp=True default | P4-S5 train.py | GPU memory check |
| R-15: Config B not type-blind | type_aware_edges=False path | P4-S1 model.py | Ablation dry-run |
| R-16: Config A not graph-free | use_graph=False path | P4-S1 model.py | Ablation dry-run |
| R-17: CDG in HDF5 | Phase 3 Gate G-05 | Stage 4 pre-check | Before Phase 4 |
| R-18: No pair_id in HDF5 | Phase 3 Gate G-08 | Stage 6 pre-check | Before Phase 4 |
| R-19: No code in metadata | HDF5 attrs or JSONL lookup | P4-S5 DataLoader | First training batch |
| R-20: Scheduler misalignment | global_step tracked manually | P4-S5 train.py | Scheduler debug |
| R-21: pairwise_accuracy = 0 | Verify val.h5 has CFA pairs | P4-S6 eval.py | After epoch 1 |
| R-22: Per-CWE F1 empty | min_samples=5 threshold | P4-S6 eval.py | After epoch 1 |
| R-23: Severity on -1 labels | valid_mask filter | P4-S4 losses.py | Loss analysis |
| R-24: Config E crashes | Zero-vector fallback | P4-S2 callee_summarizer.py | Config E dry-run |
| R-25: Partial checkpoint | Atomic os.replace() | P4-S5 train.py | Every checkpoint |
| R-26: CWE_LABEL_MAP missing | Import from losses.py | P4-S6 eval.py | First evaluation |
| R-27: compile breaks scatter | Disable torch.compile | P4-S1 model.py | Smoke test |

---

## 8. Go / No-Go Decision Criteria

These are hard gates. If any condition is RED, do NOT start full training.

| **Gate** | **Condition** | **Verify With** | **Status** |
|----------|--------------|----------------|-----------|
| G-P4-01 | GroupNorm in all GGNN layers (not BatchNorm) | `isinstance(model.ggnn_norm[0], nn.GroupNorm)` | ☐ |
| G-P4-02 | Edge type isolation: TPG ≠ AST outputs | Smoke test in test_p4s1_model.py | ☐ |
| G-P4-03 | L_CFA sign: relu(sim + 0.5) | Unit test in test_p4s4_losses.py | ☐ |
| G-P4-04 | CFA pairs in same batch | Batch pair verification script | ☐ |
| G-P4-05 | Differential LR: CodeBERT=2e-5, GGNN=1e-4 | Print optimizer param_groups | ☐ |
| G-P4-06 | Tokenization active: input_ids not None | Assert in first training batch | ☐ |
| G-P4-07 | AMP enabled on GPU | `use_amp = True` in config | ☐ |
| G-P4-08 | No NaN in training features | NaN scan on first 1000 HDF5 samples | ☐ |
| G-P4-09 | Checkpoint saves every epoch | Dry-run: verify file created | ☐ |
| G-P4-10 | MLflow running and logging | Dry-run: verify run appears in UI | ☐ |
| G-P4-11 | test.h5 NOT used during training loop | Code review: no test_loader in epoch loop | ☐ |
| G-P4-12 | All ablation configs share same HDF5 paths | Assert in run_ablations.py | ☐ |
| G-P4-13 | All ablation configs share seed=42 | Assert in run_ablations.py | ☐ |
| G-P4-14 | Dry-run completes: 2 epochs, finite loss, checkpoint saved | `--dry-run` flag | ☐ |

---

## 9. Recovery Playbook

For each critical failure mode — what to do when it happens.

| **Failure** | **Detected By** | **Recovery Time** | **Steps** |
|------------|----------------|------------------|-----------|
| Test set contamination | Post-training audit | **2–30 GPU hours** | Fix code → re-train all configs → re-evaluate |
| Wrong L_CFA sign | test_p4s4_losses.py | **20–30 GPU hours** | Fix sign → re-train Config C, D, E |
| GroupNorm missing | test_p4s1_model.py | **20–30 GPU hours** | Fix model → re-train all configs |
| Pairs split across batches | Batch verification | **20–30 GPU hours** | Fix sampler → re-train Config C, D, E |
| NaN loss after epoch 1 | Training log | **1 hour** | Restore checkpoint → lower lr_codebert to 1e-5 → resume |
| GPU OOM | CUDA OOM error | **1 hour** | Enable AMP → reduce batch_size → resume |
| Checkpoint corrupted | File unreadable | **0–20 GPU hours** | Use latest epoch checkpoint → re-train from that epoch |
| MLflow not logging | Missing experiment | **0 hours** | Add CSV backup logging; add to MLflow retroactively from CSV |
| Config B is actually type-aware | Ablation B == B' | **10 GPU hours** | Fix type_aware_edges=False path → re-train Config B |

---

## 10. Appendix: Quick Reference Verification Commands

Run these commands in sequence before starting full training:

```bash
# === PRE-TRAINING VERIFICATION SEQUENCE ===

echo "=== 1. Phase 3 Audit ==="
python training/scripts/preprocessing/pre_training_audit.py --m2
# Expected: ALL 9 PASS

echo "=== 2. GPU Memory ==="
python3 -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name}, VRAM: {p.total_memory/1e9:.1f}GB')
    if p.total_memory < 8e9: print('WARNING: Less than 8GB — ensure AMP is enabled')
else:
    print('WARNING: No GPU — training will take 90+ hours')
"

echo "=== 3. HDF5 Structure ==="
python3 -c "
import h5py, numpy as np
with h5py.File('training/data/final/train.h5','r') as f:
    ids = list(f.keys())
    print(f'Train samples: {len(ids)}')
    s = f[ids[0]]
    print(f'x shape: {s[\"x\"].shape}, edge_index: {s[\"edge_index\"].shape}')
    print(f'pair_id: {s.attrs.get(\"pair_id\",\"MISSING\")}')
    print(f'code in attrs: {\"code\" in s.attrs}')
    nan_count = sum(1 for i in ids[:100] if np.isnan(f[i]['x'][:]).any())
    print(f'NaN features (first 100): {nan_count}')
"

echo "=== 4. Model Architecture ==="
python3 -m pytest tests/test_p4s1_model.py -v

echo "=== 5. Loss Function ==="
python3 -m pytest tests/test_p4s4_losses.py -v

echo "=== 6. DataLoader Pair Integrity ==="
python3 -m pytest tests/test_p4s3_dataloader.py -v

echo "=== 7. Dry Run (2 epochs, Config C) ==="
python training/scripts/model/train.py --config C_plus_cfa --dry-run
# Expected: 2 epochs complete, checkpoint saved, MLflow logged, no NaN

echo "=== ALL PRE-TRAINING CHECKS COMPLETE ==="
```

---

*StreamGuard Phase 4 — Risk Analysis | v1.0 | March 2026*
