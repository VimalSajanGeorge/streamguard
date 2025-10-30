# Issue #11: Final Cautions & Recommendations

**Status:** ✅ All Fixes Applied
**Date:** 2025-10-30
**Version:** v1.6 Final

---

## ✅ All Tweaks Applied

### 1. Label Smoothing ✓
**Current Setting:** `label_smoothing=0.05` (conservative, safer default)

**Rationale:**
- 0.1 can sometimes slow or bias training
- 0.05 is a safer default for initial training

**Adjustment Guide:**
```python
# If you see underfitting (train loss still high, validation not improving):
label_smoothing=0.0  # Disable completely

# If training is stable but performance plateaus:
label_smoothing=0.1  # Try slightly more regularization
```

**Location:** `train_transformer.py:749-754`

---

### 2. Total Steps & Warmup Calculation ✓
**Current Implementation:** Correctly accounts for gradient accumulation

```python
# Line 753: Correct calculation
total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
warmup_steps = int(total_steps * adjusted_warmup_ratio)
```

**Why This Matters:**
- With `accumulation_steps=2` and 684 batches/epoch:
  - Wrong: `total_steps = 684 * epochs` (too many steps)
  - Correct: `total_steps = ceil(684 / 2) * epochs = 342 * epochs`
- Scheduler steps once per **optimizer update**, not per batch

**Verification:**
```bash
# Look for this in training output:
# [*] Scheduler config: total_steps=3420, warmup_steps=684 (20.0%)
```

---

### 3. Loss Accounting with Accumulation ✓
**Current Implementation:** Uses `loss.detach().item()` for accurate tracking

```python
# Line 509, 530: Correct accounting
running_loss += loss.detach().item() * accumulation_steps
```

**Why This Approach:**
1. `loss.detach()` prevents keeping computation graph in memory
2. Multiplying back by `accumulation_steps` gives raw per-step loss
3. Final average: `running_loss / len(dataloader)` is correct

**Alternative (if you prefer):**
```python
# Track raw loss before dividing
raw_loss = loss.detach().item()
running_loss += raw_loss
# Then divide by (len(dataloader) * accumulation_steps) at end
```

Both approaches are valid if done consistently.

---

### 4. Scheduler Type ✓
**Current:** `get_linear_schedule_with_warmup` (per-step scheduler)

**Implementation:** ✅ Correctly steps inside `train_epoch()` after `optimizer.step()`

**IMPORTANT:** If you ever switch to a **per-epoch scheduler** (e.g., `ReduceLROnPlateau`, `StepLR`), remember to:
1. Remove `scheduler.step()` from inside `train_epoch()`
2. Add it back to main training loop after validation
3. Update documentation

```python
# Per-epoch scheduler example (NOT current implementation):
for epoch in range(epochs):
    train_loss = train_epoch(...)  # No scheduler stepping inside
    val_metrics = evaluate(...)
    scheduler.step(val_metrics['loss'])  # Step here based on validation
```

---

### 5. Saving Checkpoints ✓
**Current Implementation:** ✅ Saves state_dicts only

```python
# Lines 307-313: Correct approach
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'metrics': metrics_to_save  # Cleaned dictionary
}
```

**Why This Matters:**
- PyTorch 2.6+ changed `torch.load()` default to `weights_only=True`
- Saving full model objects fails with this setting
- State dicts are version-agnostic and safer

**Loading:**
```python
# Line 369: Correct loading
checkpoint = torch.load(local_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**If You Must Save Complex Objects:**
```python
# Document the decision and risk
checkpoint = {
    'model': model,  # Full model (NOT recommended)
    'custom_object': my_object  # Complex object
}

# Load with weights_only=False (understand the security risk)
checkpoint = torch.load(path, weights_only=False)
```

---

### 6. AMP (Mixed Precision Training) ✓
**Current Status:** ✅ Disabled for debugging

**Notebook Cells:** Cell 7 & Cell 13 have `--mixed-precision` flag removed

**Re-enabling After Stability (3-4 stable epochs):**

```python
# Update notebook cells:
!python training/train_transformer.py \
  ... \
  --mixed-precision \  # Re-enable
  --seed 42

# The code already handles this correctly:
# Line 767: scaler = GradScaler('cuda') if args.mixed_precision and torch.cuda.is_available() else None
# Line 503: with autocast(device_type='cuda'):
```

**Best Practice:**
```python
# Even better (more explicit):
use_amp = args.mixed_precision and torch.cuda.is_available()
scaler = GradScaler('cuda') if use_amp else None

# In training loop:
with autocast(device_type='cuda', enabled=use_amp):
    logits = model(input_ids, attention_mask)
```

**Expected Impact:**
- ✅ 1.5-2x faster training
- ✅ ~40% less GPU memory
- ⚠️ Slightly different numerical behavior (usually negligible)

---

### 7. Class Weight dtype & Device ✓
**Current Implementation:** ✅ Correctly specified

```python
# Lines 686-689: Correct
class_weights = torch.tensor(
    [weight_safe, weight_vulnerable],
    dtype=torch.float32  # ✅ Must be float32
).to(device)  # ✅ Must be on same device as model
```

**Why This Matters:**
```python
# ❌ WRONG: Will cause dtype mismatch
class_weights = torch.tensor([0.923, 1.091])  # Defaults to float64

# ❌ WRONG: Will cause device mismatch
class_weights = torch.tensor([0.923, 1.091], dtype=torch.float32)  # On CPU

# ✅ CORRECT:
class_weights = torch.tensor([0.923, 1.091], dtype=torch.float32).to(device)
```

---

### 8. Enhanced Label Distribution & Early Stopping ✓
**Current Implementation:** ✅ Multiple collapse detection mechanisms

```python
# Lines 841-851: Enhanced detection
# 1. Absolute collapse (predicting only one class)
if dist['predicted_vulnerable'] == 0 or dist['predicted_safe'] == 0:
    print(f"\n[!] STOPPING: Complete collapse detected")
    break

# 2. Severe under-prediction (< 20% of actual)
if pred_vuln < 0.2 * actual_vuln:
    print(f"\n[!] WARNING: Severe under-prediction of vulnerable class")
    print(f"    Consider reducing class weights or label smoothing")
```

**Collapse Detection Strategy:**
| Condition | Action | When |
|-----------|--------|------|
| `pred_vuln == 0` | **STOP training** | Immediate (epoch 3+) |
| `pred_safe == 0` | **STOP training** | Immediate (epoch 3+) |
| `pred_vuln < 0.2 * actual_vuln` | **WARNING only** | Epoch 3+ |

**Adjusting Thresholds:**
```python
# More sensitive (detect earlier):
if pred_vuln < 0.3 * actual_vuln:  # 30% threshold

# Less sensitive (more tolerance):
if pred_vuln < 0.1 * actual_vuln:  # 10% threshold
```

---

## 🧪 Testing Plan

### Phase 1: Stability Test (2-3 epochs)
**Goal:** Confirm no immediate collapse

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --epochs 3 \
  --batch-size 64 \
  --max-seq-len 512 \
  --lr 2e-5 \
  --seed 42
```

**Success Criteria:**
- ✅ F1 > 0.50 from epoch 1
- ✅ F1 improves or stays stable across 3 epochs
- ✅ Both classes predicted (no "Complete collapse" errors)
- ✅ Prediction distribution reasonable (~40-60% each class)

**Red Flags:**
- ❌ F1 drops below 0.30
- ❌ "Complete collapse detected" message
- ❌ Prediction distribution very skewed (>90% one class)

---

### Phase 2: Full Training (10-20 epochs)
**Goal:** Confirm long-term stability and performance

```bash
# T4 Configuration (10 epochs)
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 10 \
  --batch-size 32 \
  --max-seq-len 512 \
  --lr 2e-5 \
  --seed 42

# A100 Configuration (20 epochs with LR scaling)
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 20 \
  --batch-size 64 \
  --max-seq-len 512 \
  --lr 2e-5 \
  --seed 42
```

**Success Criteria:**
- ✅ Final F1 > 0.70
- ✅ Training completes without collapse
- ✅ Early stopping may trigger (patience=2-5) - this is OK
- ✅ Test set F1 within 5% of validation F1

**Expected Performance Range:**
| Epochs | Expected F1 | Good F1 | Excellent F1 |
|--------|-------------|---------|--------------|
| 3 | 0.60-0.70 | 0.70-0.75 | 0.75+ |
| 10 | 0.70-0.80 | 0.80-0.85 | 0.85+ |
| 20 | 0.75-0.85 | 0.85-0.90 | 0.90+ |

---

### Phase 3: Re-enable AMP (after Phase 2 success)
**Goal:** Faster training with same performance

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --epochs 10 \
  --batch-size 64 \
  --max-seq-len 512 \
  --lr 2e-5 \
  --mixed-precision \  # Re-enabled
  --seed 42
```

**Success Criteria:**
- ✅ Training time reduced by ~1.5-2x
- ✅ F1 within 2% of non-AMP results
- ✅ No new stability issues

---

## 🔬 Hyperparameter Search (Optional)

**After confirming stability**, run small sweeps:

### Batch Size Sweep
```bash
for bs in 16 32 48 64; do
  python training/train_transformer.py \
    --batch-size $bs \
    --epochs 5 \
    ... \
    --output-dir models/sweep_bs_${bs}
done
```

**Expect:**
- Larger batches → slightly better F1 (better gradient estimates)
- Larger batches → need higher LR (square-root scaling already applied)

### Learning Rate Sweep (with fixed batch size)
```bash
for lr in 1e-5 2e-5 3e-5 5e-5; do
  python training/train_transformer.py \
    --lr $lr \
    --batch-size 32 \
    --epochs 5 \
    ... \
    --output-dir models/sweep_lr_${lr}
done
```

**Expect:**
- Too low LR → slow convergence
- Too high LR → unstable / divergence
- Optimal usually in 2e-5 to 5e-5 range

### Warmup Ratio Sweep
```bash
for warmup in 0.05 0.1 0.15 0.2; do
  python training/train_transformer.py \
    --warmup-ratio $warmup \
    --batch-size 32 \
    --epochs 5 \
    ... \
    --output-dir models/sweep_warmup_${warmup}
done
```

### Weight Decay Sweep
```bash
for wd in 0.0 0.005 0.01 0.02; do
  python training/train_transformer.py \
    --weight-decay $wd \
    --batch-size 32 \
    --epochs 5 \
    ... \
    --output-dir models/sweep_wd_${wd}
done
```

**Expect:**
- Too low → overfitting on training set
- Too high → underfitting (model too regularized)
- Optimal usually 0.01 (current default)

---

## 🎯 Conservative Expectations

### Immediate Impact (After These Fixes)
**Expect:**
- ✅ **Big stability improvement:** No more collapse by epoch 3
- ✅ **Solid F1 uplift:** From 0.43 (epoch 1) → 0.65-0.70 (epoch 1)
- ✅ **Consistent training:** F1 improves steadily over epochs
- ✅ **Balanced predictions:** Both classes predicted throughout

### Short-Term (10 epochs, current setup)
**Realistic Target:** F1 = 0.70-0.80
- This is a **big win** compared to complete collapse
- Represents solid baseline for vulnerability detection

### Medium-Term (20 epochs, tuned hyperparameters)
**Realistic Target:** F1 = 0.75-0.85
- With careful hyperparameter tuning
- May require label smoothing adjustment
- May benefit from slight architecture tweaks

### Long-Term (85-90% F1)
**This will require:**
1. ✅ Stable training (you'll have this after fixes)
2. 🔄 Hyperparameter optimization (sweep batch size, LR, warmup, weight decay)
3. 🔄 Better features / preprocessing
4. 🔄 Data augmentation (if applicable)
5. 🔄 Model architecture changes (e.g., deeper classifier, attention mechanisms)
6. 🔄 Ensemble methods (fusion with GNN)
7. 🔄 Domain-specific fine-tuning

**Treat 85-90% as a long-term target, not immediate expectation.**

---

## ⚠️ Important Cautions

### 1. Don't Overfit to Validation Set
- If you do many hyperparameter sweeps, you may overfit to validation set
- Always report final results on **held-out test set**
- Consider using cross-validation if dataset is small

### 2. Random Seeds Matter
- Results can vary ±2-3% F1 with different seeds
- Run with multiple seeds (42, 123, 456, 789, 2024) for confidence intervals
- Report mean ± std dev

### 3. Class Imbalance Can Shift
- If you collect more data, class distribution may change
- Recalculate class weights with new data
- Monitor prediction distribution on new data

### 4. Learning Rate Scaling is Approximate
- Square-root rule is conservative and safe
- Linear scaling (LR × batch_size/16) may work but test carefully
- Always monitor first few epochs when changing batch size

### 5. AMP Can Introduce Numerical Issues
- Usually negligible, but can cause problems in edge cases
- If you see NaN losses or divergence, disable AMP first
- Use gradient clipping (already applied) to mitigate

### 6. Early Stopping Can Be Too Aggressive
- `patience=2` is quite aggressive (stops after 2 epochs without improvement)
- Consider `patience=5` for longer training runs
- Balance between overfitting prevention and allowing model to learn

---

## 📝 Monitoring Checklist

During training, watch for:

**Good Signs:**
- ✅ F1 > 0.50 from epoch 1
- ✅ F1 steadily improving or stable
- ✅ Prediction distribution ~40-60% each class
- ✅ Train loss decreasing smoothly
- ✅ Val loss tracking train loss (not diverging)
- ✅ No "Complete collapse" or "Severe under-prediction" warnings

**Warning Signs:**
- ⚠️ F1 < 0.30 at any epoch
- ⚠️ Prediction distribution very skewed (>80% one class)
- ⚠️ Train loss not decreasing after epoch 2
- ⚠️ Val loss diverging from train loss (overfitting)
- ⚠️ "Severe under-prediction" warnings

**Critical Issues:**
- 🚨 "Complete collapse detected" message
- 🚨 NaN losses or gradients
- 🚨 F1 dropping to 0.0
- 🚨 Out of memory errors

---

## 🔧 Troubleshooting Guide

### Issue: F1 still drops to 0.0
**Possible Causes:**
1. Class weights not calculated correctly
2. Label smoothing too high
3. Learning rate too high (causing divergence)

**Solutions:**
```python
# Check class weights in training output:
# Should see: "Class weights: Safe=0.9231, Vulnerable=1.0911"

# Try disabling label smoothing:
label_smoothing=0.0

# Try lower learning rate:
--lr 1e-5
```

### Issue: F1 plateaus around 0.60-0.65
**Possible Causes:**
1. Model capacity insufficient
2. Learning rate too low
3. Need more epochs

**Solutions:**
```python
# Try higher learning rate:
--lr 3e-5

# Try more epochs:
--epochs 20

# Try unfreezing more layers (if using frozen encoder)
```

### Issue: Training loss not decreasing
**Possible Causes:**
1. Learning rate too low
2. Gradient clipping too aggressive
3. Weight decay too high

**Solutions:**
```python
# Increase learning rate:
--lr 5e-5

# Reduce weight decay:
--weight-decay 0.005

# Disable gradient clipping temporarily (edit code):
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increase from 1.0
```

### Issue: Val loss much higher than train loss
**Possible Causes:**
1. Overfitting
2. Train/val distribution mismatch
3. Need more regularization

**Solutions:**
```python
# Increase dropout:
--dropout 0.2  # From 0.1

# Increase label smoothing:
label_smoothing=0.1  # From 0.05

# Increase weight decay:
--weight-decay 0.02  # From 0.01
```

---

## 📚 Summary of All Changes

### Files Modified
1. ✅ `training/train_transformer.py` (~150 lines changed)
2. ✅ `StreamGuard_Complete_Training.ipynb` (2 cells updated)
3. ✅ `docs/ISSUE_11_TRAINING_COLLAPSE_COMPLETE_FIX.md` (created)
4. ✅ `docs/ISSUE_11_FINAL_CAUTIONS_AND_RECOMMENDATIONS.md` (this file)

### Key Fixes Applied
1. ✅ Class-balanced loss (inverse frequency weights)
2. ✅ Learning rate scaling for large batches
3. ✅ Correct per-step scheduler behavior
4. ✅ Gradient clipping for stability
5. ✅ Enhanced collapse detection
6. ✅ PyTorch 2.6+ checkpoint compatibility
7. ✅ Conservative label smoothing (0.05)
8. ✅ Proper total_steps calculation with accumulation
9. ✅ Accurate loss accounting with `loss.detach().item()`

### Current Status
- ✅ **All fixes implemented and tested (code review)**
- ⏳ **Awaiting runtime validation (2-3 epochs test)**
- ⏳ **Full training pending (10-20 epochs)**

---

## 🚀 Next Steps

1. **Run Phase 1 Test (2-3 epochs)**
   - Confirm stability
   - Check for any immediate issues
   - Validate fix effectiveness

2. **If Phase 1 Succeeds: Run Full Training**
   - 10-20 epochs depending on GPU
   - Monitor throughout
   - Document final metrics

3. **If Full Training Succeeds: Re-enable AMP**
   - Compare speed improvement
   - Validate performance equivalence
   - Update notebook cells

4. **Optional: Hyperparameter Sweep**
   - After confirming stability
   - Focus on batch size and LR
   - Document findings

5. **Long-Term: Advanced Improvements**
   - Feature engineering
   - Architecture changes
   - Ensemble methods
   - Target: 85-90% F1

---

**Status:** ✅ All cautions addressed and recommendations provided

**Ready for Testing:** Yes

**Confidence Level:** High (based on expert corrections and best practices)

---

**Last Updated:** 2025-10-30
**Version:** v1.6 Final
**Author:** Claude Code (with expert corrections)
