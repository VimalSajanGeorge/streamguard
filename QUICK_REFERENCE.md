# StreamGuard ML Training - Quick Reference Card

**🚀 Get Started in 3 Commands**

---

## Option 1: Download CodeXGLUE (Recommended)

```bash
# Install datasets library
pip install datasets

# Download CodeXGLUE
python << 'EOF'
from datasets import load_dataset
import json
from pathlib import Path

dataset = load_dataset('code_x_glue_cc_defect_detection')
output_dir = Path('data/raw/codexglue')
output_dir.mkdir(parents=True, exist_ok=True)

for split in ['train', 'validation', 'test']:
    split_name = 'valid' if split == 'validation' else split
    with open(output_dir / f'{split_name}.jsonl', 'w') as f:
        for sample in dataset[split]:
            f.write(json.dumps(sample) + '\n')
    print(f"[+] Saved {split_name}: {len(dataset[split])} samples")
EOF

# Preprocess
python training/scripts/data/preprocess_codexglue.py \
  --input-dir data/raw/codexglue \
  --output-dir data/processed/codexglue

# Train (quick test)
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --quick-test \
  --output-dir models/transformer_test
```

---

## Option 2: Use Collector Data

```bash
# Preprocess existing data
python training/scripts/data/preprocess_codexglue.py \
  --input-dir data/raw/github \
  --output-dir data/processed/github

# Train
python training/train_transformer.py \
  --train-data data/processed/github/train.jsonl \
  --val-data data/processed/github/valid.jsonl \
  --quick-test
```

---

## Full Training Commands

### Phase 1: Baseline

```bash
# 1. Preprocess (30-60 min)
python training/scripts/data/preprocess_codexglue.py \
  --input-dir data/raw/codexglue \
  --output-dir data/processed/codexglue

# 2. Train Transformer (2-3 hours)
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 5 --batch-size 16 --lr 2e-5 \
  --mixed-precision \
  --output-dir models/transformer_phase1

# 3. Train GNN (4-6 hours)
python training/train_gnn.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 100 --auto-batch-size \
  --output-dir models/gnn_phase1

# 4. Train Fusion (3-4 hours)
python training/train_fusion.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --transformer-checkpoint models/transformer_phase1/checkpoints/best_model.pt \
  --gnn-checkpoint models/gnn_phase1/checkpoints/best_model.pt \
  --output-dir models/fusion_phase1

# 5. Evaluate (1 hour)
python training/evaluate_models.py \
  --transformer-checkpoint models/transformer_phase1/checkpoints/best_model.pt \
  --gnn-checkpoint models/gnn_phase1/checkpoints/best_model.pt \
  --test-data data/processed/codexglue/test.jsonl \
  --n-runs 5 --compare
```

---

## AWS SageMaker (Cost-Optimized)

```bash
# Setup
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export SAGEMAKER_EXECUTION_ROLE="arn:aws:iam::$AWS_ACCOUNT_ID:role/StreamGuardSageMakerRole"

# Build Docker
cd training/scripts/sagemaker
docker build -t streamguard-training:v1 -f Dockerfile ../..
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag streamguard-training:v1 $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/streamguard-training:v1
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/streamguard-training:v1

# Upload data
aws s3 sync data/processed/codexglue/ s3://streamguard-training-$AWS_ACCOUNT_ID/data/

# Launch training
python launch_transformer_training.py \
  --train-data-s3 s3://streamguard-training-$AWS_ACCOUNT_ID/data/train.jsonl \
  --val-data-s3 s3://streamguard-training-$AWS_ACCOUNT_ID/data/valid.jsonl \
  --use-spot --mixed-precision
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 8 --accumulation-steps 2
```

### AST Parsing Fails
```bash
# Clone tree-sitter-c
git clone https://github.com/tree-sitter/tree-sitter-c.git vendor/tree-sitter-c

# Fallback still works - 100% samples will be processed
```

### Slow Training
```bash
# Enable mixed precision (~2x speedup)
--mixed-precision

# Use SageMaker Spot ($0.20/hour)
```

---

## Expected Results

| Model | Phase 1 F1 | Phase 2 F1 | Training Time | Cost (SageMaker) |
|-------|-----------|-----------|---------------|------------------|
| Transformer | 0.68-0.71 | 0.72-0.76 | 2-3 hours | $0.40 |
| GNN | 0.64-0.67 | 0.68-0.72 | 4-6 hours | $0.80 |
| Fusion | 0.70-0.73 | 0.74-0.78 | 4 hours | $0.20 |

**Total Cost:** ~$1.40 (Phase 1) + ~$3.40 (Phase 2) = **$4.80 / $100 budget**

---

## File Locations

```
streamguard/
├── training/
│   ├── train_transformer.py       # Main transformer training
│   ├── train_gnn.py              # Main GNN training
│   ├── train_fusion.py           # Fusion layer training
│   ├── evaluate_models.py        # Statistical evaluation
│   └── scripts/
│       ├── data/
│       │   └── preprocess_codexglue.py
│       └── sagemaker/
│           ├── launch_transformer_training.py
│           └── Dockerfile
├── tests/
│   └── test_preprocessing.py
├── COMPLETE_ML_TRAINING_GUIDE.md  # 📖 Full guide (25 pages)
├── IMPLEMENTATION_COMPLETE.md     # 📋 Executive summary
└── QUICK_REFERENCE.md            # ⚡ This file
```

---

## Quick Tests

```bash
# Test preprocessing (2 min)
python training/scripts/data/preprocess_codexglue.py --quick-test

# Test transformer training (10 min)
python training/train_transformer.py --quick-test

# Test GNN training (15 min)
python training/train_gnn.py --quick-test

# Run unit tests
python tests/test_preprocessing.py
```

---

## Help Commands

```bash
# Any script has detailed help
python training/train_transformer.py --help
python training/train_gnn.py --help
python training/train_fusion.py --help

# Check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check dependencies
python -c "import torch, torch_geometric, transformers; print('OK')"
```

---

## Documentation Index

| Read First | Then | Advanced |
|------------|------|----------|
| IMPLEMENTATION_COMPLETE.md | COMPLETE_ML_TRAINING_GUIDE.md | PHASE_6_ML_TRAINING_IMPLEMENTATION.md |
| (Executive summary) | (Step-by-step commands) | (Technical architecture) |

---

## Status Checklist

**Prerequisites:**
- [ ] Python 3.10+ installed
- [ ] CUDA 11.8+ (for GPU)
- [ ] Dependencies installed (`pip install ...`)
- [ ] tree-sitter-c cloned

**Phase 1:**
- [ ] Dataset obtained
- [ ] Preprocessing complete
- [ ] Transformer trained (F1 >0.65)
- [ ] GNN trained (F1 >0.60)
- [ ] Fusion trained (F1 >0.70)

**Phase 2:**
- [ ] Collector data preprocessed
- [ ] Noise reduction applied
- [ ] Models retrained
- [ ] Improvement verified

---

**🚀 Ready to start? Run the 3 commands at the top!**

**📖 Need details? See COMPLETE_ML_TRAINING_GUIDE.md**

**❓ Questions? Check the troubleshooting section above**

---

**Last Updated:** October 24, 2025
**Status:** ✅ Production-Ready
