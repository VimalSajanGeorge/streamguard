# GNN Training Phase 2/3 Roadmap

**Status:** Phase 1 Complete ✅
**Last Updated:** 2025-01-05

This document outlines deferred features for GNN training that were identified during Phase 1 implementation but postponed for later consideration.

---

## Phase 1 Recap (Completed)

✅ LR Finder with safety caps and caching
✅ Weighted sampler for class balance (PyG-correct)
✅ Class weighting with inverse-frequency + safety caps
✅ Focal loss integration
✅ Triple weighting auto-adjustment
✅ Robust collapse detection (2 consecutive epochs)
✅ Gradient clipping (AMP-compatible)
✅ Prediction distribution tracking
✅ CSV metrics history
✅ Enhanced checkpoint metadata
✅ Diagnostic reports (JSON + TXT)

---

## Phase 2: High Priority Features (Selective Implementation)

### 1. Parameter Groups / Discriminative LR ⚠️ EVALUATE FIRST

**Why:** Encoder (GNN layers) and head (classifier) may benefit from different learning rates, similar to Transformer fine-tuning.

**Implementation:**
```python
# Separate encoder and head parameters
encoder_params = []  # GCN layers
head_params = []      # Classifier
no_decay_params = []  # Bias, LayerNorm

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(k in name.lower() for k in ['bias', 'norm']):
        no_decay_params.append(param)
    elif any(k in name.lower() for k in ['conv', 'embedding']):
        encoder_params.append(param)
    else:
        head_params.append(param)

# Build optimizer with parameter groups
optimizer = torch.optim.Adam([
    {'params': encoder_params, 'lr': scaled_lr * 0.1, 'weight_decay': args.weight_decay},
    {'params': head_params, 'lr': scaled_lr * 1.0, 'weight_decay': args.weight_decay},
    {'params': no_decay_params, 'lr': scaled_lr * 1.0, 'weight_decay': 0.0}
])
```

**Risk:** GNN architectures vary significantly; may need careful tuning per model architecture.

**Estimated Effort:** 2-3 hours + testing

**Dependencies:** Stable Phase 1 baseline for A/B comparison

**Recommendation:** Only implement if:
- Transfer learning is used (pretrained GNN encoder)
- Ablation shows clear F1 gain (>2%)
- Consider gradual unfreezing first (simpler approach)

---

### 2. Mixed Precision Training (AMP) ⚠️ TEST COMPATIBILITY FIRST

**Why:** 1.5-2x speedup, lower memory usage (critical for large graphs)

**Implementation:**
```python
parser.add_argument('--mixed-precision', action='store_true')

scaler = torch.amp.GradScaler() if args.mixed_precision else None

# In train_epoch():
if scaler:
    with torch.amp.autocast(device_type='cuda'):
        logits = model(data)
        loss = criterion(logits, data.y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Risk:** PyG operations may not fully support AMP; needs validation on all GNN layers.

**Estimated Effort:** 3-4 hours + compatibility testing

**Dependencies:** Gradient clipping integration (✅ already in Phase 1)

**Testing Checklist:**
- [ ] Run with GCN, GAT, GraphSAGE models
- [ ] Verify no NaN losses
- [ ] Compare F1 scores (AMP vs FP32)
- [ ] Profile speedup on T4/V100/A100

**Recommendation:** Test on single GPU first, then enable in production if stable.

---

### 3. TensorBoard Logging 📊

**Why:** Better visualization of training dynamics (loss curves, metrics, LR traces, predictions)

**Implementation:**
```python
parser.add_argument('--tensorboard', action='store_true')

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(args.output_dir / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')))

# In training loop:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Metrics/f1_vulnerable', val_metrics['binary_f1_vulnerable'], epoch)
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
writer.add_histogram('Predictions/distribution', all_preds, epoch)
```

**Risk:** Low - purely observability, no training impact

**Estimated Effort:** 2-3 hours

**Dependencies:** None

**Recommendation:** Implement for debugging complex training runs, especially when tuning hyperparameters.

---

### 4. Gradient Accumulation ⚠️ ASSESS NEED FIRST

**Why:** Larger effective batch sizes on limited GPU memory

**Implementation:**
```python
parser.add_argument('--accumulation-steps', type=int, default=1)

# In train_epoch():
for step, data in enumerate(dataloader):
    loss = criterion(logits, data.y) / args.accumulation_steps
    loss.backward()

    if (step + 1) % args.accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler_per_step:
            scheduler.step()
```

**Risk:** GNN already has `--auto-batch-size` which dynamically adjusts batch size based on graph statistics.

**Estimated Effort:** 2-3 hours

**Dependencies:** Check if `--auto-batch-size` is sufficient first

**Recommendation:** DEFER unless `--auto-batch-size` fails for extremely large graphs (>10k nodes).

---

## Phase 3: Advanced Features (Optional)

### 1. Stochastic Weight Averaging (SWA) 🔬

**Why:** Improved generalization via weight averaging

**Implementation:**
```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

swa_model = AveragedModel(model)
swa_start_epoch = args.epochs // 2
swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

for epoch in range(args.epochs):
    train_epoch(...)

    if epoch >= swa_start_epoch:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

# Before final eval:
update_bn(train_loader, swa_model, device)
```

**Risk:**
- Complex interaction with GNN architectures
- Requires batch norm recalculation
- May not work well with ReduceLROnPlateau scheduler

**Estimated Effort:** 1-2 days

**Dependencies:**
- Stable Phase 1+2 baseline
- Understand SWA benefits for GNNs (literature review)

**Recommendation:** DEFER until empirical evidence shows GNNs benefit from SWA (run ablation on 3+ datasets).

---

### 2. Advanced Ensembling 🎯

**Why:** Combine multiple models for better predictions

**Approaches:**
- **Bagging:** Train on different data subsets
- **Different seeds:** Train same model with 5+ random seeds
- **Different architectures:** GCN + GAT + GraphSAGE

**Implementation:**
```python
# Train multiple models
models = [train_model(seed=i) for i in range(5)]

# Ensemble predictions
def ensemble_predict(models, data):
    logits_list = [model(data) for model in models]
    avg_logits = torch.mean(torch.stack(logits_list), dim=0)
    return torch.argmax(avg_logits, dim=1)
```

**Risk:**
- High storage overhead (5x models)
- Slower inference (5x forward passes)

**Estimated Effort:** 2-3 days

**Dependencies:** Multiple trained models

**Recommendation:** DEFER to production deployment phase. Focus on single model optimization first.

---

### 3. Hyperparameter Optimization (Optuna) 🔧

**Why:** Auto-tune LR, weight_multiplier, focal_gamma, hidden_dim, num_layers, etc.

**Implementation:**
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_multiplier = trial.suggest_float('weight_multiplier', 1.0, 3.0)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])

    # Train model with these hyperparameters
    val_f1 = train_and_evaluate(lr, weight_multiplier, focal_gamma, hidden_dim)
    return val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Risk:**
- Expensive (50+ training runs)
- May overfit to validation set

**Estimated Effort:** 3-5 days (includes infrastructure setup)

**Dependencies:** Stable baseline, clear optimization objective

**Recommendation:** DEFER until manual tuning establishes reasonable hyperparameter ranges.

---

### 4. Distributed Data Parallel (DDP) 🚀

**Why:** Multi-GPU training for faster convergence

**Implementation:**
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
if args.use_weighted_sampler:
    # WARNING: WeightedRandomSampler not DDP-ready
    # Need custom DistributedWeightedSampler
    raise NotImplementedError("Weighted sampler not DDP-compatible")
else:
    sampler = DistributedSampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
```

**Risk:**
- WeightedRandomSampler needs custom DDP wrapper
- Debugging is harder (multiple processes)
- Requires multi-GPU hardware

**Estimated Effort:** 2-3 days

**Dependencies:** Access to multi-GPU setup

**Recommendation:** DEFER until single-GPU training is fully optimized. DDP adds complexity without always improving final F1.

---

## Implementation Priority Ranking

### Implement Next (if Phase 1 is stable):

1. **TensorBoard Logging** (2-3h) - Low risk, high debugging value
2. **Mixed Precision** (3-4h) - Test compatibility, 2x speedup if works
3. **Parameter Groups** (2-3h) - IF using transfer learning

### Evaluate Before Implementing:

1. **Gradient Accumulation** - Check if `--auto-batch-size` sufficient
2. **SWA** - Run literature review on GNN + SWA
3. **Ensembling** - Measure single model F1 first

### Defer Indefinitely:

1. **Optuna** - Manual tuning sufficient for now
2. **DDP** - Single-GPU optimization first

---

## Testing Requirements for Phase 2/3 Features

Each feature MUST have:

1. **Unit Tests**
   - Test feature in isolation (e.g., AMP forward/backward pass)
   - Test interaction with existing features (e.g., AMP + gradient clipping)

2. **Smoke Tests**
   - 64 samples, 10 epochs (should complete without errors)
   - Full dataset, 3 epochs (should not crash)

3. **Ablation Tests**
   - Baseline (Phase 1 only)
   - Baseline + new feature
   - Measure F1 delta (must be >2% to justify complexity)

4. **Documentation**
   - Update `docs/TRAINING_QUICK_START.md`
   - Add flag to `--help`
   - Document known limitations

---

## Success Metrics

A Phase 2/3 feature is worth implementing if:

✅ F1 improvement >2% on validation set
✅ No stability issues (0 collapses in 10 runs)
✅ Training time increase <20%
✅ Code complexity increase is justified by gains
✅ Works across different GPU types (T4/V100/A100)

---

## Reference Implementation: Transformer v1.7

For reference, see `training/train_transformer.py` lines 958-1027 (parameter groups) and lines 1084-1091 (mixed precision).

The Transformer implementation can serve as a template, but **GNN architectures differ significantly** - always test on GNN models before assuming feature parity.

---

## Questions to Answer Before Phase 2

1. Does the Phase 1 baseline achieve acceptable F1 (>0.80)?
2. Are there systematic failure modes (e.g., always predicts safe for large graphs)?
3. Is training time a bottleneck (>24 hours on V100)?
4. Is memory a bottleneck (OOM on batch_size=32)?
5. Do we need better observability (hard to debug collapses)?

**If answer is "no" to all:** Phase 2/3 features may not be necessary. Focus on data quality and model architecture instead.

---

## Contact & Questions

For questions about this roadmap or to propose new Phase 2/3 features, please:
- Open an issue: https://github.com/VimalSajanGeorge/streamguard/issues
- Reference this document: `docs/GNN_PHASE_2_3_ROADMAP.md`

---

**Last Updated:** 2025-01-05
**Author:** StreamGuard ML Team
**Status:** Living Document (update as Phase 1 results become available)
