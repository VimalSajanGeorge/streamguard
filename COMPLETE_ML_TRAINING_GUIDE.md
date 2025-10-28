# StreamGuard ML Training - Complete Implementation Guide

**Version:** 1.0
**Last Updated:** October 24, 2025
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Baseline Training (CodeXGLUE)](#phase-1-baseline-training-codexglue)
4. [Phase 2: Enhanced Training (Collectors)](#phase-2-enhanced-training-collectors)
5. [AWS SageMaker Deployment](#aws-sagemaker-deployment)
6. [Troubleshooting](#troubleshooting)
7. [Cost Optimization](#cost-optimization)

---

## Overview

StreamGuard uses a **two-phase training strategy** to build production vulnerability detection models:

**Phase 1: Baseline Training**
- Dataset: CodeXGLUE/Devign (27,318 samples, clean and balanced)
- Models: Enhanced SQL Intent Transformer + Enhanced Taint-Flow GNN + Fusion
- Goal: Establish clean baseline performance

**Phase 2: Enhanced Training**
- Dataset: Phase 1 + Collector data (GitHub, OSV, ExploitDB, Synthetic)
- Models: Same architecture, retrained with noise reduction
- Goal: Improve coverage and real-world performance

###  Architecture

```
Input: C/C++ Code
│
├─→ [Enhanced SQL Intent Transformer]
│   - Base: CodeBERT
│   - Output: Logits [batch, 2]
│
├─→ [Enhanced Taint-Flow GNN]
│   - 4-layer GCN on AST
│   - Output: Logits [batch, 2]
│
└─→ [Fusion Layer]
    - Weighted averaging + MLP
    - Output: Final prediction [batch, 2]
```

---

## Prerequisites

### 1. System Requirements

**Local Development:**
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 50GB+ disk space

**AWS SageMaker (Recommended):**
- AWS Account with $100 credits
- IAM Role with SageMaker permissions
- S3 Bucket for data and models

### 2. Install Dependencies

```bash
# Core dependencies
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Transformers
pip install transformers==4.35.0 tokenizers==0.15.0

# Tree-sitter for AST parsing
pip install tree-sitter==0.20.4

# Additional dependencies
pip install scikit-learn scipy tqdm requests boto3 sagemaker

# For development
pip install pytest black flake8
```

### 3. Clone tree-sitter-c

```bash
cd streamguard
mkdir -p vendor
cd vendor
git clone https://github.com/tree-sitter/tree-sitter-c.git
cd ..
```

---

## Phase 1: Baseline Training (CodeXGLUE)

### Step 1: Get CodeXGLUE Dataset

**Option A: Download from Hugging Face (Recommended)**

Since the direct URLs are outdated, use the Hugging Face `datasets` library:

```bash
pip install datasets

python3 << 'EOF'
from datasets import load_dataset
from pathlib import Path

# Load dataset
dataset = load_dataset("code_x_glue_cc_defect_detection")

# Create output directory
output_dir = Path("data/raw/codexglue")
output_dir.mkdir(parents=True, exist_ok=True)

# Save to JSONL
for split in ['train', 'validation', 'test']:
    split_name = 'valid' if split == 'validation' else split
    output_file = output_dir / f"{split_name}.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dataset[split]:
            import json
            f.write(json.dumps(sample) + '\n')

    print(f"[+] Saved {split_name}: {len(dataset[split])} samples")

print("[+] Download complete!")
EOF
```

**Option B: Use Pre-downloaded Dataset**

If you have a local copy or alternative source:

```bash
# Place files in:
data/raw/codexglue/train.jsonl
data/raw/codexglue/valid.jsonl
data/raw/codexglue/test.jsonl
```

**Option C: Use Synthetic/Collector Data**

If CodeXGLUE is unavailable, you can start with your collector data:

```bash
# Use existing collector data
python training/scripts/collection/run_full_collection.py \
  --collectors synthetic github osv \
  --synthetic-samples 10000 \
  --github-samples 5000 \
  --osv-samples 5000
```

### Step 2: Preprocess Dataset

This step converts raw data to StreamGuard's standardized format with:
- Token offsets for future Integrated Gradients
- AST nodes with fallback strategies
- Vulnerable code-aware trimming
- Graph statistics for GNN batch sizing

**Quick Test (100 samples):**

```bash
python training/scripts/data/preprocess_codexglue.py \
  --input-dir data/raw/codexglue \
  --output-dir data/processed/codexglue \
  --quick-test
```

**Expected output:**
```
[*] Processing train split: data/raw/codexglue/train.jsonl
    Processed 100 samples (AST: 85, Fallback: 15)

Graph Statistics & GNN Batch Size Recommendation
  total_samples: 100
  avg_nodes: 127.3
  p95_nodes: 384
  recommended_batch_size: 32
```

**Full Preprocessing (27K samples, ~30-60 minutes):**

```bash
python training/scripts/data/preprocess_codexglue.py \
  --input-dir data/raw/codexglue \
  --output-dir data/processed/codexglue \
  --tokenizer microsoft/codebert-base \
  --max-seq-len 512
```

**Expected output:**
```
data/processed/codexglue/
├── train.jsonl (21,854 samples)
├── valid.jsonl (2,732 samples)
├── test.jsonl (2,732 samples)
└── preprocessing_metadata.json
```

**Monitor these metrics:**
- **AST Success Rate:** Target >80% (fallback handles the rest)
- **Recommended Batch Size:** Typically 24-32 for 16GB GPU

### Step 3: Train Enhanced SQL Intent Transformer

**Quick Test (100 samples, ~10 minutes):**

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --quick-test \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5 \
  --output-dir models/transformer_test
```

**Full Training (Local GPU, ~2-3 hours):**

```bash
python training/train_transformer.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --max-seq-len 512 \
  --dropout 0.1 \
  --early-stopping-patience 2 \
  --mixed-precision \
  --output-dir models/transformer_phase1 \
  --seed 42
```

**Expected results:**
```
Epoch 1/5
----------
Train Loss: 0.4823
Val Loss: 0.3912
Val Accuracy: 0.7234
Val F1 (vulnerable): 0.6891

...

TRAINING COMPLETE
Best validation F1 (vulnerable): 0.7145

TEST EVALUATION
Test Accuracy: 0.7089
Test F1 (vulnerable): 0.6823
```

**Output files:**
```
models/transformer_phase1/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── best_model.pt
│   └── latest.pt
└── exp_config.json
```

### Step 4: Train Enhanced Taint-Flow GNN

**Quick Test:**

```bash
python training/train_gnn.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --quick-test \
  --epochs 20 \
  --auto-batch-size \
  --output-dir models/gnn_test
```

**Full Training (Local GPU, ~4-6 hours with early stopping):**

```bash
python training/train_gnn.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.3 \
  --early-stopping-patience 10 \
  --auto-batch-size \
  --output-dir models/gnn_phase1 \
  --seed 42
```

**Expected results:**
```
Graph Statistics:
  P95 nodes: 384
  Recommended batch size: 32

Epoch 1/100
----------
Train Loss: 0.6234
Val Accuracy: 0.6891
Val F1 (vulnerable): 0.6512

...

Early stopping at epoch 63

TEST EVALUATION
Test Accuracy: 0.6823
Test F1 (vulnerable): 0.6445
```

### Step 5: Train Fusion Layer

The fusion layer combines Transformer + GNN predictions using out-of-fold (OOF) predictions to prevent data leakage.

**Generate OOF Predictions (5-fold CV, ~3-4 hours):**

```bash
python training/train_fusion.py \
  --train-data data/processed/codexglue/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --transformer-checkpoint models/transformer_phase1/checkpoints/best_model.pt \
  --gnn-checkpoint models/gnn_phase1/checkpoints/best_model.pt \
  --n-folds 5 \
  --epochs 20 \
  --lr 1e-3 \
  --output-dir models/fusion_phase1 \
  --seed 42
```

**Process:**
1. Split training data into 5 folds
2. For each fold:
   - Train Transformer on 4 folds (fine-tune)
   - Train GNN on 4 folds (fine-tune)
   - Generate predictions on held-out fold
3. Train fusion layer on OOF predictions
4. Evaluate on validation and test sets

**Expected results:**
```
Generating OOF predictions with 5-fold CV
  Total samples: 21854

Fold 1/5
  Train: 17483, Val: 4371
  Training Transformer...
  Training GNN...
  Generating predictions for fold 1...

...

Training fusion layer
  Training samples: 21854
  Validation samples: 2732

Fusion training complete. Best Val F1: 0.7423

TEST EVALUATION
Test F1 (vulnerable): 0.7189
```

### Step 6: Evaluate with Statistical Testing

Compare all models with multiple runs and statistical significance testing:

```bash
python training/evaluate_models.py \
  --transformer-checkpoint models/transformer_phase1/checkpoints/best_model.pt \
  --gnn-checkpoint models/gnn_phase1/checkpoints/best_model.pt \
  --test-data data/processed/codexglue/test.jsonl \
  --n-runs 5 \
  --compare \
  --output evaluation_phase1.json
```

**Expected output:**
```
Evaluating transformer with 5 runs
  Run 1/5 (seed=42)...
  Run 2/5 (seed=43)...
  ...

Evaluating gnn with 5 runs
  ...

Statistical Comparison: Transformer vs GNN
======================================================================

F1:
  Transformer: 0.6823 ± 0.0089 (95% CI: [0.6745, 0.6901])
  GNN: 0.6445 ± 0.0112 (95% CI: [0.6334, 0.6556])
  Improvement: -0.0378 (-5.5%)
  p-value: 0.0234 ✓ SIGNIFICANT

F1_VULNERABLE:
  Transformer: 0.7145 ± 0.0095 (95% CI: [0.7055, 0.7235])
  GNN: 0.6734 ± 0.0108 (95% CI: [0.6626, 0.6842])
  Improvement: -0.0411 (-5.8%)
  p-value: 0.0198 ✓ SIGNIFICANT

Results saved to: evaluation_phase1.json
```

---

## Phase 2: Enhanced Training (Collectors)

### Step 1: Collect Data from All Sources

Run full data collection (this was done in previous sessions):

```bash
# Check existing collector data
ls -lh data/raw/github/
ls -lh data/raw/opensource/
ls -lh data/raw/cves/

# If needed, run additional collection
python training/scripts/collection/run_full_collection.py \
  --collectors github osv exploitdb synthetic \
  --github-samples 5000 \
  --osv-samples 10000 \
  --exploitdb-samples 3000 \
  --synthetic-samples 5000
```

### Step 2: Preprocess Collector Data

Apply the same preprocessing pipeline to collector data:

```bash
# Preprocess each source
for source in github osv exploitdb synthetic; do
  python training/scripts/data/preprocess_codexglue.py \
    --input-dir data/raw/$source \
    --output-dir data/processed/$source \
    --tokenizer microsoft/codebert-base
done
```

### Step 3: Noise Reduction

Use Phase 1 models to filter low-quality samples with weighted sampling:

```bash
python training/scripts/data/noise_reduction.py \
  --phase1-transformer models/transformer_phase1/checkpoints/best_model.pt \
  --phase1-gnn models/gnn_phase1/checkpoints/best_model.pt \
  --input-dirs data/processed/github data/processed/osv data/processed/exploitdb \
  --output-dir data/processed/phase2_filtered \
  --confidence-threshold 0.8 \
  --hard-set-retention 0.1
```

**This creates weighted samples:**
- High confidence (>0.8): weight = 1.0
- Medium confidence (0.6-0.8): weight = 0.3
- Low confidence (<0.6): 10% retained with weight = 0.1

### Step 4: Merge Datasets

```bash
python training/scripts/data/merge_datasets.py \
  --base-data data/processed/codexglue \
  --additional-data data/processed/phase2_filtered \
  --output-dir data/processed/phase2_merged \
  --verify-distribution
```

**Expected output:**
```
Merging datasets:
  Base (CodeXGLUE): 27,318 samples
  Additional (Collectors): 18,452 samples (after filtering)
  Total: 45,770 samples

Label distribution:
  Vulnerable: 22,893 (50.0%)
  Safe: 22,877 (50.0%)
  ✓ Distribution maintained
```

### Step 5: Retrain Models with Weights

**Retrain Transformer:**

```bash
python training/train_transformer.py \
  --train-data data/processed/phase2_merged/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 5 \
  --use-weights \
  --output-dir models/transformer_phase2
```

**Retrain GNN:**

```bash
python training/train_gnn.py \
  --train-data data/processed/phase2_merged/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --epochs 100 \
  --use-weights \
  --auto-batch-size \
  --output-dir models/gnn_phase2
```

**Retrain Fusion:**

```bash
python training/train_fusion.py \
  --train-data data/processed/phase2_merged/train.jsonl \
  --val-data data/processed/codexglue/valid.jsonl \
  --test-data data/processed/codexglue/test.jsonl \
  --transformer-checkpoint models/transformer_phase2/checkpoints/best_model.pt \
  --gnn-checkpoint models/gnn_phase2/checkpoints/best_model.pt \
  --output-dir models/fusion_phase2
```

### Step 6: Compare Phase 1 vs Phase 2

```bash
python training/evaluate_models.py \
  --transformer-checkpoint models/transformer_phase2/checkpoints/best_model.pt \
  --gnn-checkpoint models/gnn_phase2/checkpoints/best_model.pt \
  --test-data data/processed/codexglue/test.jsonl \
  --n-runs 5 \
  --output evaluation_phase2.json

# Compare results
python training/scripts/compare_phases.py \
  --phase1-results evaluation_phase1.json \
  --phase2-results evaluation_phase2.json \
  --output phase_comparison.json
```

---

## AWS SageMaker Deployment

### Step 1: Setup AWS Environment

**Configure AWS CLI:**

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter:
#   AWS Access Key ID
#   AWS Secret Access Key
#   Default region: us-east-1
#   Default output format: json
```

**Create IAM Role for SageMaker:**

```bash
# Save this as trust-policy.json
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name StreamGuardSageMakerRole \
  --assume-role-policy-document file://trust-policy.json

# Attach managed policies
aws iam attach-role-policy \
  --role-name StreamGuardSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name StreamGuardSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get role ARN (save this!)
aws iam get-role --role-name StreamGuardSageMakerRole --query 'Role.Arn' --output text
```

**Set environment variable:**

```bash
export SAGEMAKER_EXECUTION_ROLE="arn:aws:iam::YOUR_ACCOUNT_ID:role/StreamGuardSageMakerRole"
```

### Step 2: Build and Push Custom Docker Image

**Build Docker image:**

```bash
cd training/scripts/sagemaker

# Build image
docker build -t streamguard-training:v1 -f Dockerfile ../..

# Test locally
docker run --rm streamguard-training:v1 python3 -c "import torch; import torch_geometric; import transformers; print('OK')"
```

**Push to Amazon ECR:**

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"

# Create ECR repository
aws ecr create-repository --repository-name streamguard-training --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag image
docker tag streamguard-training:v1 \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/streamguard-training:v1

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/streamguard-training:v1

# Save image URI
export CUSTOM_IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/streamguard-training:v1"
echo "Image URI: $CUSTOM_IMAGE_URI"
```

### Step 3: Upload Data to S3

```bash
# Create S3 bucket
aws s3 mb s3://streamguard-training-$AWS_ACCOUNT_ID

# Upload processed data
aws s3 sync data/processed/codexglue/ \
  s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/data/ \
  --exclude "*.md"

# Verify upload
aws s3 ls s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/data/
```

### Step 4: Launch Training on SageMaker

**Train Transformer on SageMaker:**

```bash
python training/scripts/sagemaker/launch_transformer_training.py \
  --train-data-s3 s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/data/train.jsonl \
  --val-data-s3 s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/data/valid.jsonl \
  --test-data-s3 s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/data/test.jsonl \
  --output-s3 s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/output/transformer \
  --role $SAGEMAKER_EXECUTION_ROLE \
  --custom-image $CUSTOM_IMAGE_URI \
  --instance-type ml.g4dn.xlarge \
  --use-spot \
  --max-run-hours 4 \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5 \
  --mixed-precision
```

**Monitor training:**

```bash
# Via AWS Console:
# https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs

# Or via CLI:
aws sagemaker list-training-jobs --max-results 5
```

**Download trained model:**

```bash
# Find job name
JOB_NAME=$(aws sagemaker list-training-jobs \
  --name-contains streamguard-transformer \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 1 \
  --query 'TrainingJobSummaries[0].TrainingJobName' \
  --output text)

# Get model artifact location
MODEL_ARTIFACT=$(aws sagemaker describe-training-job \
  --training-job-name $JOB_NAME \
  --query 'ModelArtifacts.S3ModelArtifacts' \
  --output text)

# Download
aws s3 cp $MODEL_ARTIFACT models/transformer_sagemaker.tar.gz

# Extract
tar -xzf models/transformer_sagemaker.tar.gz -C models/transformer_sagemaker/
```

### Step 5: Cost Monitoring

**Check current spending:**

```bash
# Get cost for last 7 days
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter file://<(cat <<EOF
{
  "Dimensions": {
    "Key": "SERVICE",
    "Values": ["Amazon SageMaker"]
  }
}
EOF
)
```

**Expected costs (Phase 1):**
- Transformer: ~$0.40 (2 hours @ $0.20/hour Spot)
- GNN: ~$0.80 (4 hours @ $0.20/hour Spot)
- Fusion: ~$0.20 (1 hour @ $0.20/hour Spot)
- **Total: ~$1.40**

---

## Troubleshooting

### Issue 1: AST Parsing Fails

**Symptoms:**
```
[!] tree-sitter C language library not found
AST parsing will use fallback mode
```

**Solution:**

```bash
# Ensure tree-sitter-c is cloned
cd vendor
git clone https://github.com/tree-sitter/tree-sitter-c.git
cd ..

# Build library manually
python3 << 'EOF'
from tree_sitter import Language
from pathlib import Path

build_dir = Path('build')
build_dir.mkdir(exist_ok=True)

Language.build_library(
    'build/my-languages.so',
    ['vendor/tree-sitter-c']
)
print("[+] Built successfully")
EOF
```

### Issue 2: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

```bash
# For Transformer: Reduce batch size
python training/train_transformer.py \
  --batch-size 8 \  # Instead of 16
  --accumulation-steps 2  # Maintain effective batch size

# For GNN: Use auto-batch-size
python training/train_gnn.py \
  --auto-batch-size  # Automatically adjusts based on graph size
```

### Issue 3: Slow Training

**Solutions:**

1. **Enable mixed precision:**
```bash
--mixed-precision  # ~2x speedup
```

2. **Use data parallel (multi-GPU):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python training/train_transformer.py \
  --batch-size 32  # Larger batch with 2 GPUs
```

3. **Use SageMaker instead of local:**
```bash
# ml.g4dn.xlarge is faster than most local GPUs
```

### Issue 4: SageMaker Job Fails

**Check logs:**

```bash
# Via AWS Console:
# SageMaker → Training Jobs → [job name] → View logs

# Or download CloudWatch logs
aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name $JOB_NAME/algo-1-1234567890
```

**Common issues:**

1. **IAM permissions:** Ensure role has S3 and SageMaker access
2. **Spot interruption:** Check if Spot was interrupted, job should auto-resume
3. **Data format:** Verify S3 paths and file formats

### Issue 5: Low Model Performance

**Debugging steps:**

1. **Check data quality:**
```bash
python training/scripts/data/verify_dataset.py \
  --dataset data/processed/codexglue/train.jsonl
```

2. **Review preprocessing metadata:**
```bash
cat data/processed/codexglue/preprocessing_metadata.json
```

3. **Visualize predictions:**
```bash
python training/scripts/analyze_predictions.py \
  --model models/transformer_phase1/checkpoints/best_model.pt \
  --data data/processed/codexglue/test.jsonl \
  --output analysis/predictions.html
```

4. **Check for data leakage:**
```bash
# Ensure OOF predictions were used for fusion
grep "OOF" training/train_fusion.py
```

---

## Cost Optimization

### 1. Use Spot Instances (62% savings)

```bash
# Always enable Spot for training
--use-spot
```

### 2. Checkpoint Frequently

```bash
# Already implemented in all scripts
# S3 checkpointing enables resume after interruptions
```

### 3. Use `--quick-test` During Development

```bash
# Test with 100 samples first
--quick-test

# Only run full training when ready
```

### 4. Optimize Instance Types

```bash
# For testing: ml.g4dn.xlarge ($0.20/hour Spot)
# For production: ml.p3.2xlarge ($1.23/hour Spot)

# CPU-only (cheaper): ml.c5.4xlarge ($0.34/hour Spot)
```

### 5. Stop Idle Resources

```bash
# List running jobs
aws sagemaker list-training-jobs --status-equals InProgress

# Stop job if needed
aws sagemaker stop-training-job --training-job-name <job-name>
```

### 6. Use S3 Lifecycle Policies

```bash
# Archive old checkpoints to Glacier after 30 days
aws s3api put-bucket-lifecycle-configuration \
  --bucket streamguard-training-$AWS_ACCOUNT_ID \
  --lifecycle-configuration file://lifecycle.json
```

---

## Summary Checklist

### Phase 1 (Baseline)
- [ ] Install dependencies
- [ ] Download/obtain CodeXGLUE dataset
- [ ] Preprocess data (quick test first)
- [ ] Train Transformer (local or SageMaker)
- [ ] Train GNN (local or SageMaker)
- [ ] Train Fusion with OOF predictions
- [ ] Evaluate with statistical testing
- [ ] Verify F1 (vulnerable) > 0.65

### Phase 2 (Enhanced)
- [ ] Collect data from all sources
- [ ] Preprocess collector data
- [ ] Run noise reduction
- [ ] Merge datasets
- [ ] Retrain all models with weights
- [ ] Compare Phase 1 vs Phase 2
- [ ] Verify improvement is significant

### AWS SageMaker (Optional)
- [ ] Setup AWS credentials
- [ ] Create IAM role
- [ ] Build and push Docker image
- [ ] Upload data to S3
- [ ] Launch training jobs
- [ ] Monitor costs (<$10 target)

---

## Expected Timeline

| Task | Duration | Cost (SageMaker) |
|------|----------|------------------|
| **Phase 1** | | |
| Data preprocessing | 1-2 hours | $0 (local) |
| Transformer training | 2-3 hours | $0.40 |
| GNN training | 4-6 hours | $0.80 |
| Fusion training | 3-4 hours | $0.20 |
| Evaluation | 1 hour | $0 (local) |
| **Phase 1 Subtotal** | **11-16 hours** | **$1.40** |
| | | |
| **Phase 2** | | |
| Data collection | 2-4 hours | $0 (local) |
| Noise reduction | 1 hour | $0.20 |
| Retraining (all models) | 10-12 hours | $2.80 |
| Comparison | 2 hours | $0.40 |
| **Phase 2 Subtotal** | **15-19 hours** | **$3.40** |
| | | |
| **TOTAL** | **26-35 hours** | **$4.80** |

**Remaining budget from $100:** ~$95

---

## Next Steps After Training

1. **Model Deployment:**
   - Create SageMaker endpoint
   - Integrate with StreamGuard API
   - Setup monitoring and alerts

2. **Continuous Retraining:**
   - Schedule monthly retraining
   - Incorporate new vulnerability data
   - A/B test model versions

3. **Explainability (Deferred):**
   - Implement Integrated Gradients
   - Visualize token attributions
   - Generate vulnerability reports

4. **Documentation:**
   - API documentation
   - Model performance benchmarks
   - Deployment guide

---

## Support and Resources

**Internal Documentation:**
- `PHASE_6_ML_TRAINING_IMPLEMENTATION.md` - Implementation details
- `docs/ml_training_completion.md` - Architecture overview
- `tests/test_preprocessing.py` - Unit tests

**External Resources:**
- CodeXGLUE Paper: https://arxiv.org/abs/2102.04664
- PyTorch Geometric Docs: https://pytorch-geometric.readthedocs.io/
- SageMaker Python SDK: https://sagemaker.readthedocs.io/

**Contact:**
- For technical issues: Check `TROUBLESHOOTING.md`
- For AWS cost concerns: Monitor via Cost Explorer

---

**Last Updated:** October 24, 2025
**Version:** 1.0
**Status:** ✅ Production-Ready
