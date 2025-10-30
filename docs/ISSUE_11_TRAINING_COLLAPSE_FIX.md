# Issue #11: Critical Training Collapse Fix

**Status:** 🔴 CRITICAL - Model Predicting Only Safe Class
**Severity:** Catastrophic
**Affected Component:** Transformer Training
**Date:** 2025-10-30

---

## 📊 Problem Summary

### Observed Behavior

```
Epoch 1/20: F1 = 0.4337 ✓ (model learns both classes)
Epoch 2/20: F1 = 0.1125 ⚠️ (model starting to collapse)
Epoch 3/20: F1 = 0.0000 ❌ (model ONLY predicts safe class)
Epoch 4/20: F1 = 0.0000 ❌ (complete collapse continues)
Epoch 5/20: F1 = 0.0000 ❌
Epoch 6/20: F1 = 0.0000 ❌ (early stopping triggered)

Warning: "Precision and F-score are ill-defined due to no predicted samples"
Error: "_pickle.UnpicklingError: Weights only load failed" (on test evaluation)
```

### Root Causes Identified

1. **No Class Balancing (PRIMARY CAUSE)**
   - Dataset: 45.8% vulnerable, 54.2% safe
   - No class weights in loss function
   - Model learns "always predict safe" = 54.2% accuracy

2. **Learning Rate Too Low for Large Batch**
   - Batch size: 64 (AGGRESSIVE config)
   - Learning rate: 2e-5 (designed for batch 16)
   - Effective learning is too slow

3. **Scheduler Misplacement**
   - `get_linear_schedule_with_warmup` is per-step scheduler
   - Currently stepping per-epoch (wrong)

4. **Unnecessary Complexity**
   - Using `reduction='none'` + manual weighting
   - Should use `reduction='mean'` for simplicity

5. **PyTorch 2.6 Compatibility**
   - New default `weights_only=True` breaks checkpoint loading
   - Need to save only state_dicts

6. **No Monitoring**
   - Can't detect collapse until too late
   - Need per-class prediction counts

---

## 🔧 Complete Fix Strategy (Corrected & Refined)

### Fix 1: Class-Balanced Loss (CRITICAL - Prevents Collapse)

**Location:** `training/train_transformer.py`, after loading datasets (~line 670)

```python
# Calculate class weights from training data
print("[*] Calculating class weights for balanced training...")
train_labels = []
for sample in train_dataset:
    train_labels.append(sample['label'])

class_counts = torch.bincount(torch.tensor(train_labels))
num_safe = class_counts[0].item()              # 11836
num_vulnerable = class_counts[1].item()        # 10018
total = len(train_labels)                      # 21854

# Inverse frequency weights (balanced)
weight_safe = total / (2.0 * num_safe)              # ~0.923
weight_vulnerable = total / (2.0 * num_vulnerable)  # ~1.091

# CRITICAL: Must be float32 and on correct device
class_weights = torch.tensor(
    [weight_safe, weight_vulnerable],
    dtype=torch.float32
).to(device)

print(f"[+] Class balancing enabled:")
print(f"    Dataset: {num_vulnerable} vulnerable ({num_vulnerable/total*100:.1f}%), "
      f"{num_safe} safe ({num_safe/total*100:.1f}%)")
print(f"    Weights: Safe={weight_safe:.4f}, Vulnerable={weight_vulnerable:.4f}")
```

**Location:** `training/train_transformer.py`, criterion creation (~line 703)

```python
# Loss criterion with class balancing
criterion = nn.CrossEntropyLoss(
    weight=class_weights,      # ✅ Class balancing (prevents collapse)
    label_smoothing=0.05,      # ✅ Conservative smoothing (was 0.1)
    reduction='mean'           # ✅ Simple mean reduction (not 'none')
)

print(f"[+] Loss criterion: CrossEntropyLoss with class weights and label smoothing")
```

**Key Points:**
- ✅ Use `reduction='mean'` (default), NOT `'none'`
- ✅ Only use `reduction='none'` if you need per-sample weighting
- ✅ `label_smoothing=0.05` is safer than 0.1 (less risk of underfitting)
- ✅ Class weights must be `dtype=torch.float32` and `.to(device)`

---

### Fix 2: Remove Sample-Level Weighting (Simplification)

**Location:** `training/train_transformer.py`, train_epoch function (~line 470-550)

**BEFORE (Complex):**
```python
weights = batch['weight'].to(device)
loss = criterion(logits, labels)  # reduction='none' → [batch_size]
if weights is not None and len(weights) > 0:
    loss = (loss * weights).mean()  # Manual weighting
else:
    loss = loss.mean()
```

**AFTER (Simple):**
```python
# No need for sample weights - class weights handle imbalance
loss = criterion(logits, labels)  # Already scalar with reduction='mean'
```

**Complete Updated train_epoch Function:**

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    accumulation_steps: int = 1
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model instance
        dataloader: Data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device
        scheduler: LR scheduler (optional, stepped per optimizer step)
        scaler: GradScaler for mixed precision (optional)
        accumulation_steps: Gradient accumulation steps

    Returns:
        Average loss
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        # ✅ REMOVED: weights = batch['weight'].to(device)

        # Forward pass (with or without AMP)
        if scaler:
            with autocast(device_type='cuda'):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)  # ✅ Already scalar
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                # ✅ Unscale before gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # ✅ Step scheduler AFTER optimizer (per-step scheduler)
                if scheduler is not None:
                    scheduler.step()

        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)  # ✅ Already scalar
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # ✅ Step scheduler AFTER optimizer (per-step scheduler)
                if scheduler is not None:
                    scheduler.step()

        # ✅ Accurate loss accounting
        running_loss += loss.detach().item() * accumulation_steps
        num_batches += 1

    return running_loss / num_batches
```

**Key Points:**
- ✅ Removed all sample weight logic (unnecessary with class weights)
- ✅ Added scheduler parameter and step it per optimizer step
- ✅ Added gradient clipping (max_norm=1.0)
- ✅ Use `loss.detach().item()` for accurate loss tracking
- ✅ Account for accumulation_steps correctly

---

### Fix 3: Scale Learning Rate for Large Batch (WITH WARMUP)

**Location:** `training/train_transformer.py`, before creating optimizer (~line 688)

```python
import math

# Learning rate scaling for large batches
base_lr = args.lr  # 2e-5
base_batch = 16

if args.batch_size > base_batch:
    # Square-root scaling (conservative and stable)
    scale_factor = math.sqrt(args.batch_size / base_batch)
    scaled_lr = base_lr * scale_factor

    print(f"[+] LR scaling for large batch:")
    print(f"    Batch size: {args.batch_size} (base: {base_batch})")
    print(f"    Base LR: {base_lr:.2e}")
    print(f"    Scaled LR: {scaled_lr:.2e} (×{scale_factor:.2f})")
else:
    scaled_lr = base_lr
    print(f"[+] Using base LR: {base_lr:.2e}")

# Create optimizer with scaled LR
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=scaled_lr,  # ✅ SCALED LR
    weight_decay=args.weight_decay,
    eps=1e-8,
    betas=(0.9, 0.999)
)

# ✅ CRITICAL: Adjust warmup proportionally for stability
if args.batch_size > base_batch:
    adjusted_warmup_ratio = args.warmup_ratio * scale_factor
    # Cap at 0.2 (20% warmup maximum)
    adjusted_warmup_ratio = min(adjusted_warmup_ratio, 0.2)
else:
    adjusted_warmup_ratio = args.warmup_ratio

# ✅ CORRECTED: Account for gradient accumulation in total_steps
total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
warmup_steps = int(total_steps * adjusted_warmup_ratio)

print(f"[+] Scheduler configuration:")
print(f"    Total steps: {total_steps}")
print(f"    Warmup steps: {warmup_steps} ({adjusted_warmup_ratio*100:.1f}%)")
print(f"    Effective batch size: {args.batch_size * args.accumulation_steps}")

scheduler = get_linear_schedule_with_warmup(
    optimizer, warmup_steps, total_steps
)
```

**Scaling Examples:**
```
Batch 64 (AGGRESSIVE):
  - Scaled LR = 2e-5 * sqrt(64/16) = 2e-5 * 2.0 = 4e-5
  - Warmup = 10% * 2.0 = 20% (capped)

Batch 32 (OPTIMIZED/ENHANCED):
  - Scaled LR = 2e-5 * sqrt(32/16) = 2e-5 * 1.41 = 2.83e-5
  - Warmup = 10% * 1.41 = 14.1%

Batch 16 (default):
  - Scaled LR = 2e-5 (no change)
  - Warmup = 10%
```

**Key Points:**
- ✅ Square-root scaling is conservative and stable
- ✅ Warmup adjusted proportionally (prevents instability)
- ✅ Capped at 20% warmup maximum
- ✅ Total steps calculation accounts for accumulation

---

### Fix 4: Fix Scheduler Stepping (Per-Step, Inside train_epoch)

**Already fixed in Fix 2 (train_epoch function)**

**Location:** `training/train_transformer.py`, main training loop (~line 728)

**BEFORE (Wrong):**
```python
train_loss = train_epoch(
    model, train_loader, optimizer, criterion,
    device, scaler, args.accumulation_steps
)

val_metrics = evaluate(model, val_loader, device, criterion)

scheduler.step()  # ❌ Wrong: stepping per-epoch
```

**AFTER (Correct):**
```python
# ✅ Pass scheduler to train_epoch
train_loss = train_epoch(
    model, train_loader, optimizer, criterion,
    device, scheduler, scaler, args.accumulation_steps  # ✅ Added scheduler
)

val_metrics = evaluate(model, val_loader, device, criterion)

# ✅ REMOVED: scheduler.step() (now inside train_epoch)
```

**Key Points:**
- ✅ `get_linear_schedule_with_warmup` is per-step scheduler
- ✅ Must step after each optimizer.step(), not per epoch
- ✅ Scheduler passed to train_epoch and stepped there

---

### Fix 5: Fix Checkpoint Save/Load (PyTorch 2.6 Compatible)

**Location:** `training/train_transformer.py`, S3CheckpointManager.save_checkpoint (~line 340)

```python
def save_checkpoint(
    self,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    metrics: Dict[str, Any],
    is_best: bool = False
) -> bool:
    """Save checkpoint (PyTorch 2.6 compatible)."""

    # ✅ Save only state_dicts (avoids pickle issues)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        # ✅ Convert numpy/tensor to native Python types
        'metrics': {
            k: float(v) if isinstance(v, (np.floating, np.integer, torch.Tensor)) else v
            for k, v in metrics.items()
            if k != 'prediction_distribution'  # Skip nested dict
        },
        'prediction_distribution': metrics.get('prediction_distribution', {})
    }

    # Save locally
    local_path = self.local_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, local_path)

    print(f"[+] Saved checkpoint: {local_path}")

    # Save as best if applicable
    if is_best:
        best_path = self.local_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"[+] Saved best model: {best_path}")

    # Upload to S3 if available
    if self.s3_available and self.s3_bucket:
        try:
            s3_key = f"{self.s3_prefix}/checkpoint_epoch_{epoch}.pt"
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key
            )
            if is_best:
                s3_best_key = f"{self.s3_prefix}/best_model.pt"
                self.s3_client.upload_file(
                    str(best_path),
                    self.s3_bucket,
                    s3_best_key
                )
        except Exception as e:
            print(f"[!] S3 upload failed: {e}")

    return True
```

**Location:** `training/train_transformer.py`, S3CheckpointManager.load_checkpoint (~line 360)

```python
def load_checkpoint(self, filename: str) -> Optional[Dict]:
    """Load checkpoint (PyTorch 2.6 compatible)."""
    local_path = self.local_dir / filename

    if not local_path.exists():
        print(f"[!] Checkpoint not found: {local_path}")
        return None

    print(f"[*] Loading checkpoint from {local_path}")

    # ✅ Load with weights_only=True (safe with state_dicts)
    try:
        checkpoint = torch.load(local_path, weights_only=True)
        return checkpoint
    except Exception as e:
        print(f"[!] Failed to load checkpoint: {e}")
        return None
```

**Location:** `training/train_transformer.py`, test evaluation (~line 782)

```python
# Load best model for testing
best_checkpoint = checkpoint_mgr.load_checkpoint('best_model.pt')
if best_checkpoint:
    # ✅ Load state_dict correctly
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"[+] Loaded best model from epoch {best_checkpoint['epoch']}")
else:
    print(f"[!] Could not load best model, using current model")
```

**Key Points:**
- ✅ Save only `state_dict()`, not full model objects
- ✅ Convert numpy/tensors to native Python types
- ✅ Load with `weights_only=True` (safe and compatible)
- ✅ Use `model.load_state_dict()` correctly

---

### Fix 6: Add Prediction Distribution Monitoring

**Location:** `training/train_transformer.py`, evaluate function (~line 420)

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Dict[str, Any]:
    """Evaluate model with prediction distribution monitoring."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1, zero_division=0
    )

    # ✅ Prediction distribution (detect collapse)
    pred_distribution = {
        'predicted_vulnerable': int((all_preds == 1).sum()),
        'predicted_safe': int((all_preds == 0).sum()),
        'actual_vulnerable': int((all_labels == 1).sum()),
        'actual_safe': int((all_labels == 0).sum())
    }

    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'binary_f1_vulnerable': f1,
        'prediction_distribution': pred_distribution  # ✅ Added
    }

    return metrics
```

**Location:** `training/train_transformer.py`, main training loop (~line 734)

```python
# Evaluate
val_metrics = evaluate(model, val_loader, device, criterion)

print(f"Train Loss: {train_loss:.4f}")
print(f"Val Loss: {val_metrics['loss']:.4f}")
print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
print(f"Val F1 (vulnerable): {val_metrics['binary_f1_vulnerable']:.4f}")

# ✅ Show prediction distribution
dist = val_metrics['prediction_distribution']
total_preds = dist['predicted_vulnerable'] + dist['predicted_safe']
vuln_pct = dist['predicted_vulnerable'] / total_preds * 100 if total_preds > 0 else 0
safe_pct = dist['predicted_safe'] / total_preds * 100 if total_preds > 0 else 0

print(f"Predictions: Vuln={dist['predicted_vulnerable']} ({vuln_pct:.1f}%), "
      f"Safe={dist['predicted_safe']} ({safe_pct:.1f}%)")
print(f"Actual: Vuln={dist['actual_vulnerable']}, Safe={dist['actual_safe']}")

# ✅ Collapse detection (immediate warning)
if dist['predicted_vulnerable'] == 0:
    print("[!] ⚠️  CRITICAL: Model collapse detected - predicting ONLY safe class!")
    print("    Training is failing. Check:")
    print("    1. Class weights are applied correctly")
    print("    2. Learning rate is not too low")
    print("    3. Batch size is reasonable")
elif dist['predicted_vulnerable'] < dist['actual_vulnerable'] * 0.3:
    # Less than 30% of actual vulnerable predicted
    print(f"[!] ⚠️  WARNING: Model under-predicting vulnerable class")
    print(f"    Predicted: {dist['predicted_vulnerable']}, "
          f"Actual: {dist['actual_vulnerable']} "
          f"({dist['predicted_vulnerable']/dist['actual_vulnerable']*100:.1f}%)")
```

**Key Points:**
- ✅ Track predictions per class every epoch
- ✅ Detect collapse immediately (predicted_vulnerable == 0)
- ✅ Warn if under-predicting vulnerable class (<30%)
- ✅ Show percentages for easy interpretation

---

### Fix 7: Disable AMP for Initial Debugging

**Location:** `StreamGuard_Complete_Training.ipynb`, Cell 7 training command

```bash
# ✅ REMOVE --mixed-precision flag for debugging
!python training/train_transformer.py \
  --train-data /content/data/processed/codexglue/train.jsonl \
  --val-data /content/data/processed/codexglue/valid.jsonl \
  --test-data /content/data/processed/codexglue/test.jsonl \
  --output-dir /content/models/transformer_phase1 \
  --epochs {t_config['epochs']} \
  --batch-size {t_config['batch_size']} \
  --max-seq-len {t_config['max_seq_len']} \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --dropout 0.1 \
  --early-stopping-patience {t_config['patience']} \
  --seed 42
  # ✅ REMOVED: --mixed-precision (re-enable after stability confirmed)
```

**After confirming stability (3-4 epochs with improving F1), re-enable:**

```bash
  --mixed-precision \  # ✅ Re-enable after stability confirmed
```

**Key Points:**
- ✅ AMP can hide subtle numeric issues
- ✅ Debug without AMP first
- ✅ Re-enable only after F1 stable and improving
- ✅ When re-enabling, ensure proper `autocast()` usage

---

### Fix 8: Add Conservative Early Stopping with Collapse Detection

**Location:** `training/train_transformer.py`, main training loop (~line 750)

```python
# Early stopping logic
is_best = val_metrics['binary_f1_vulnerable'] > best_val_f1

checkpoint_mgr.save_checkpoint(
    epoch + 1, model, optimizer, scheduler, val_metrics, is_best
)

# ✅ Check for catastrophic collapse
dist = val_metrics['prediction_distribution']
if dist['predicted_vulnerable'] == 0:
    print("[!] CRITICAL: Model collapsed - stopping training")
    break

# ✅ Check for severe under-prediction (3 consecutive epochs)
if dist['predicted_vulnerable'] < dist['actual_vulnerable'] * 0.2:
    collapse_counter += 1
    if collapse_counter >= 3:
        print("[!] Model severely under-predicting vulnerable class for 3 epochs")
        print("    Stopping training early")
        break
else:
    collapse_counter = 0  # Reset if prediction improves

# Normal early stopping
if is_best:
    best_val_f1 = val_metrics['binary_f1_vulnerable']
    patience_counter = 0
    print(f"[+] New best model! F1: {best_val_f1:.4f}")
else:
    patience_counter += 1
    print(f"[*] No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

if patience_counter >= args.early_stopping_patience:
    print(f"\n[!] Early stopping triggered at epoch {epoch + 1}")
    break
```

**Add at beginning of training loop:**
```python
best_val_f1 = 0.0
patience_counter = 0
collapse_counter = 0  # ✅ Track consecutive collapse warnings
```

**Key Points:**
- ✅ Immediate stop if complete collapse (predicted_vulnerable == 0)
- ✅ Stop if <20% prediction for 3 consecutive epochs
- ✅ Reset collapse counter if predictions improve
- ✅ Normal early stopping still applies

---

## 📋 Complete Implementation Checklist

### Priority 1: CRITICAL (Must Fix for Training to Work)

- [ ] **Fix 1:** Add class-balanced loss (`weight=class_weights`, `label_smoothing=0.05`, `reduction='mean'`)
- [ ] **Fix 2:** Remove sample-level weighting (simplify train_epoch)
- [ ] **Fix 5:** Fix checkpoint save/load (state_dicts only, PyTorch 2.6 compatible)
- [ ] **Fix 6:** Add prediction distribution monitoring
- [ ] **Fix 7:** Disable AMP temporarily (`--mixed-precision` removed)
- [ ] **Fix 8:** Add collapse detection and early stopping

### Priority 2: Performance & Stability

- [ ] **Fix 3:** Scale learning rate for large batch (with warmup adjustment)
- [ ] **Fix 4:** Fix scheduler stepping (per-step, inside train_epoch)
- [ ] Add gradient clipping (max_norm=1.0) - already in Fix 2

---

## 📝 Files to Modify

### File 1: `training/train_transformer.py`

**Changes Required:**

| Line | Section | Change |
|------|---------|--------|
| ~340 | `save_checkpoint` | Save only state_dicts, convert types |
| ~360 | `load_checkpoint` | Use `weights_only=True` |
| ~420 | `evaluate` | Add prediction_distribution |
| ~470 | `train_epoch` signature | Add `scheduler` parameter |
| ~470-550 | `train_epoch` body | Remove sample weights, add clipping, step scheduler |
| ~670 | main, after loading data | Calculate class weights |
| ~688 | main, before optimizer | Add LR scaling with warmup adjustment |
| ~696 | main, total_steps calc | Account for accumulation: `math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs` |
| ~703 | main, criterion | Add class weights, `label_smoothing=0.05`, `reduction='mean'` |
| ~715 | main, before training loop | Add `collapse_counter = 0` |
| ~728 | main, training loop | Pass scheduler to train_epoch |
| ~734 | main, after metrics | Print prediction distribution |
| ~742 | main, after metrics | **REMOVE** `scheduler.step()` |
| ~750 | main, early stopping | Add collapse detection logic |
| ~782 | main, test eval | Use `model.load_state_dict()` |

### File 2: `StreamGuard_Complete_Training.ipynb`

**Cell 7:** Remove `--mixed-precision` flag (re-enable after stability)

---

## 🎯 Expected Results After Fixes

### Before Fixes (COLLAPSED):
```
Epoch 1: Loss=0.6960, F1=0.4337, Pred: Vuln=~1000, Safe=~1732
Epoch 2: Loss=0.6959, F1=0.1125, Pred: Vuln=~200, Safe=~2532
Epoch 3: Loss=0.6936, F1=0.0000, Pred: Vuln=0, Safe=2732 ❌ COLLAPSE
Epoch 4-6: F1=0.0000 (collapse continues)
```

### After Fixes (STABLE & IMPROVING):
```
Epoch 1: Loss=0.620, F1=0.55-0.60, Pred: Vuln=~1100, Safe=~1632 ✅
Epoch 2: Loss=0.580, F1=0.65-0.70, Pred: Vuln=~1150, Safe=~1582 ✅
Epoch 3: Loss=0.540, F1=0.72-0.77, Pred: Vuln=~1180, Safe=~1552 ✅
Epoch 4: Loss=0.510, F1=0.78-0.82, Pred: Vuln=~1200, Safe=~1532 ✅
Epoch 5: Loss=0.485, F1=0.82-0.86, Pred: Vuln=~1210, Safe=~1522 ✅
...
Final (10-15 epochs): F1=0.85-0.90 ✅ TARGET

Balanced predictions throughout!
No collapse!
Stable improvement!
```

---

## 🚀 Implementation Steps

### Step 1: Apply Critical Fixes (15-20 min)
1. Add class weights calculation and update criterion
2. Simplify train_epoch (remove sample weights)
3. Fix checkpoint save/load
4. Add prediction distribution monitoring
5. Add collapse detection

**Test:** Run 2-3 epochs, verify:
- F1 > 0.5
- Predicted vulnerable > 0
- F1 improving

### Step 2: Apply Performance Fixes (10 min)
6. Add LR scaling with warmup adjustment
7. Fix scheduler placement (move to train_epoch)
8. Verify total_steps calculation accounts for accumulation

**Test:** Run 5 epochs, verify:
- F1 continues improving
- Loss decreasing
- No warnings

### Step 3: Full Training Run
9. Run complete training (10-20 epochs depending on config)
10. Monitor for stability
11. Verify F1 reaches 0.80+

### Step 4: Re-enable AMP (After Stability)
12. Add back `--mixed-precision` flag
13. Verify training still stable
14. Should see ~1.5-2x speedup

---

## 📊 Monitoring During Training

### Healthy Training Signs:
```
✅ F1 starting at 0.5+ (not collapsing)
✅ F1 improving each epoch
✅ Predictions balanced (vulnerable predictions > 0)
✅ Loss decreasing
✅ No warnings about collapse
```

### Warning Signs:
```
⚠️  F1 < 0.3 after epoch 1 (check class weights)
⚠️  F1 decreasing (learning rate too high or model diverging)
⚠️  Predicted vulnerable dropping toward 0 (collapse starting)
⚠️  Loss not decreasing (learning rate too low or optimizer issue)
```

### Critical Issues:
```
❌ F1 = 0.0 (complete collapse - stop immediately)
❌ Predicted vulnerable = 0 (collapse - stop immediately)
❌ Loss increasing (divergence - stop and reduce LR)
❌ NaN loss (gradient explosion - add clipping, reduce LR)
```

---

## 🔬 Post-Fix Hyperparameter Tuning (After Stability)

Once training is stable, consider tuning:

### Learning Rate Sweep:
```
Batch 32: Try [2e-5, 3e-5, 4e-5]
Batch 64: Try [3e-5, 4e-5, 5e-5]
```

### Warmup Ratio:
```
Try: [0.05, 0.10, 0.15, 0.20]
```

### Label Smoothing:
```
Try: [0.0, 0.05, 0.10]
If underfitting, reduce or remove
```

### Weight Decay:
```
Current: 0.01
Try: [0.001, 0.01, 0.1]
```

### Batch Size:
```
Try different sizes with proper LR scaling
Larger batch = faster training but may reduce generalization
```

---

## 💡 Key Takeaways

### What We Learned:

1. **Class imbalance kills training** without proper weighting
2. **Large batches need scaled LR** (and more warmup)
3. **Per-step schedulers** must step per optimizer step, not per epoch
4. **Monitoring is critical** - need to see collapse early
5. **Simpler is better** - `reduction='mean'` beats manual weighting
6. **AMP can hide issues** - debug without it first
7. **PyTorch 2.6 changes** require state_dict-only checkpoints

### Best Practices Applied:

✅ Class-balanced loss for imbalanced data
✅ Conservative label smoothing (0.05)
✅ LR scaling with warmup for large batches
✅ Gradient clipping for stability
✅ Per-class prediction monitoring
✅ Early collapse detection
✅ Safe checkpoint save/load
✅ Accurate loss accounting

---

## 📚 References

- **Class Balancing:** He et al., "Focal Loss for Dense Object Detection" (2017)
- **LR Scaling:** Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
- **Warmup:** Gotmare et al., "A Closer Look at Deep Learning Heuristics" (2019)
- **Label Smoothing:** Szegedy et al., "Rethinking the Inception Architecture" (2016)
- **PyTorch 2.6 Changes:** https://pytorch.org/docs/stable/notes/serialization.html

---

**Status:** 📋 Ready for Implementation
**Priority:** 🔴 CRITICAL
**Expected Fix Time:** 30-40 minutes
**Expected Improvement:** F1: 0.00 → 0.85+

---

**Next Steps:**
1. Review this complete plan
2. Implement fixes in order (Priority 1, then Priority 2)
3. Test with 2-3 epochs after Priority 1 fixes
4. Complete full training after all fixes
5. Document results

