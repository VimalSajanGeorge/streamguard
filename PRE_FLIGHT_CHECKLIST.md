# StreamGuard Pre-Flight Checklist - Before Training

**Date:** October 24, 2025
**Python Version:** 3.12.2 ✅
**PyTorch Version:** 2.7.1+cu118
**CUDA Available:** ❌ No (CPU-only training)

---

## ⚠️ IMPORTANT: What You Need to Know

### 1. **SageMaker is OPTIONAL** - You Can Train Locally

The training scripts have **optional** SageMaker support, but it's NOT required:

- **boto3** is imported but gracefully skips if not available
- All scripts work perfectly **without AWS/SageMaker**
- S3 checkpointing is only enabled if you explicitly configure it

**ACTION NEEDED:** ✅ **NONE** - You can train locally without any SageMaker setup!

### 2. **CPU-Only Training Detected**

Your system shows:
- PyTorch: 2.7.1+cu118 ✅
- CUDA Available: **False** ❌

**What this means:**
- Training will be **MUCH SLOWER** (5-10x longer)
- Expected timeline:
  - Preprocessing: 1-2 hours (same)
  - Transformer: **10-15 hours** instead of 2-3 hours
  - GNN: **20-30 hours** instead of 4-6 hours
  - Fusion: **15-20 hours** instead of 3-4 hours
  - **Total: 46-67 hours** instead of 10-14 hours

**OPTIONS:**
1. **Continue with CPU** (slower but works)
2. **Use Google Colab Free GPU** (recommended)
3. **Use AWS SageMaker** (costs ~$1.40 for Phase 1)

### 3. **Tree-sitter C Library MUST Be Built**

**Status:** ❌ NOT BUILT YET

**Required for:** AST (Abstract Syntax Tree) parsing during preprocessing

**ACTION NEEDED:** Build the tree-sitter library BEFORE preprocessing

---

## 📋 Complete Pre-Flight Checklist

### ✅ Already Complete

- [x] Python 3.12.2 installed
- [x] PyTorch 2.7.1 installed
- [x] Transformers 4.57.0 installed
- [x] PyTorch Geometric 2.4.0 installed
- [x] tree-sitter package installed
- [x] scikit-learn 1.7.2 installed
- [x] vendor/tree-sitter-c cloned
- [x] CodeXGLUE dataset downloaded (27,318 samples)

### ⚠️ Action Required Before Training

#### **1. Build tree-sitter C Library** (REQUIRED)

This is needed for preprocessing to parse C/C++ code into AST.

**Command:**
```bash
python -c "
from tree_sitter import Language
from pathlib import Path

# Create build directory
build_dir = Path('build')
build_dir.mkdir(exist_ok=True)

# Build library
Language.build_library(
    'build/my-languages.dll',  # Windows uses .dll
    ['vendor/tree-sitter-c']
)
print('[+] tree-sitter C library built successfully!')
"
```

**Expected output:**
```
[+] tree-sitter C library built successfully!
```

**Verify:**
```bash
powershell -Command "if (Test-Path 'build\my-languages.dll') { Write-Host 'SUCCESS: Library built' } else { Write-Host 'FAILED: Library not found' }"
```

---

#### **2. Verify All Dependencies** (RECOMMENDED)

Run this to check all packages:

```bash
python -c "
import sys
import importlib

required = {
    'torch': '2.0+',
    'transformers': '4.0+',
    'torch_geometric': '2.0+',
    'tree_sitter': 'any',
    'sklearn': 'any',
    'numpy': 'any',
    'scipy': 'any'
}

print('Checking dependencies...')
print('-' * 50)
missing = []
for pkg, version in required.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', 'installed')
        print(f'✓ {pkg:20s} {ver}')
    except ImportError:
        print(f'✗ {pkg:20s} MISSING')
        missing.append(pkg)

print('-' * 50)
if missing:
    print(f'❌ Missing: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
else:
    print('✅ All dependencies installed!')
"
```

---

#### **3. Check Disk Space** (RECOMMENDED)

```bash
# Check available disk space
powershell -Command "Get-PSDrive C | Select-Object @{Name='Free(GB)';Expression={[math]::Round($_.Free/1GB,2)}}"
```

**Required space:**
- Preprocessing output: ~200 MB
- Model checkpoints: ~2-5 GB
- **Total needed: ~10 GB free recommended**

---

## 🔧 Optional Configurations

### A. **For Local CPU Training** (Default)

**No changes needed!** Just run the scripts as documented.

**Tips to speed up:**
- Close other applications
- Use `--quick-test` first to verify setup
- Run overnight for full training

---

### B. **For Google Colab GPU Training** (Recommended Alternative)

If you want faster training without AWS costs:

**Setup:**
1. Go to https://colab.research.google.com/
2. Upload your preprocessing script
3. Upload your preprocessed data (or run preprocessing in Colab)
4. Run training with free GPU (T4)

**Speed:** ~3-4x faster than CPU, completely free!

**Colab GPU detection:**
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

### C. **For AWS SageMaker Training** (Optional)

**Only if you want to use AWS SageMaker for faster training**

#### Prerequisites:
1. AWS Account with credits
2. AWS CLI configured
3. IAM role created
4. S3 bucket created

#### Configuration Steps:

**1. Install AWS tools:**
```bash
pip install boto3 sagemaker awscli
```

**2. Configure AWS credentials:**
```bash
aws configure
# Enter:
#   AWS Access Key ID: [your key]
#   AWS Secret Access Key: [your secret]
#   Default region: us-east-1
#   Default output format: json
```

**3. Create IAM role for SageMaker:**

Save this as `trust-policy.json`:
```json
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
```

Run:
```bash
aws iam create-role --role-name StreamGuardSageMakerRole --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --role-name StreamGuardSageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name StreamGuardSageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Get role ARN (save this!)
aws iam get-role --role-name StreamGuardSageMakerRole --query 'Role.Arn' --output text
```

**4. Create S3 bucket:**
```bash
# Get your AWS account ID
$AWS_ACCOUNT_ID = aws sts get-caller-identity --query Account --output text

# Create bucket
aws s3 mb s3://streamguard-training-$AWS_ACCOUNT_ID

# Upload data
aws s3 sync data/processed/codexglue/ s3://streamguard-training-$AWS_ACCOUNT_ID/phase1/data/
```

**5. Modify training script to use S3:**

When running training, add these flags:
```bash
python training/train_transformer.py \
  --s3-checkpoint-uri s3://streamguard-training-YOUR_ACCOUNT_ID/checkpoints/ \
  --use-spot \
  ... (other flags)
```

**Cost estimate:** ~$1.40 for full Phase 1 with Spot instances

---

## 🎯 Recommended Path for You

Based on your current setup (CPU-only, no GPU), here's what I recommend:

### **Option 1: Quick Local Test + Colab Full Training** (RECOMMENDED)

1. **Build tree-sitter library** (5 min)
2. **Run quick preprocessing test locally** (5 min)
3. **Run full preprocessing locally** (1-2 hours)
4. **Upload to Google Colab** (10 min)
5. **Train on Colab free GPU** (6-10 hours total)

**Benefits:**
- Free
- Fast (GPU acceleration)
- No AWS complexity

---

### **Option 2: Full Local CPU Training** (SIMPLEST)

1. **Build tree-sitter library** (5 min)
2. **Run preprocessing** (1-2 hours)
3. **Train locally** (46-67 hours)

**Benefits:**
- No cloud setup needed
- Complete control
- Can run overnight/weekend

**Drawbacks:**
- Very slow
- Ties up your machine

---

### **Option 3: AWS SageMaker** (FASTEST, SMALL COST)

1. **Build tree-sitter library** (5 min)
2. **Run preprocessing locally** (1-2 hours)
3. **Setup AWS** (30 min - follow section C above)
4. **Train on SageMaker** (10-14 hours, ~$1.40 cost)

**Benefits:**
- Fast GPU training
- Spot instances save money
- Production-ready setup

**Drawbacks:**
- AWS setup required
- Small cost (~$1.40)

---

## 📝 Pre-Training Test Commands

Before committing to full training, run these quick tests:

### Test 1: Build tree-sitter (REQUIRED)
```bash
python -c "from tree_sitter import Language; from pathlib import Path; Path('build').mkdir(exist_ok=True); Language.build_library('build/my-languages.dll', ['vendor/tree-sitter-c']); print('[+] Built!')"
```

### Test 2: Quick preprocessing (5 min)
```bash
python training/scripts/data/preprocess_codexglue.py --input-dir data/raw/codexglue --output-dir data/processed/codexglue_test --quick-test
```

**Expected output:**
```
[*] Loading tokenizer: microsoft/codebert-base
[+] Tokenizer validated: fast=True, supports_offsets=True
[*] Building tree-sitter parser
[+] Parser ready
[*] Processing train split: data/raw/codexglue/train.jsonl
    Processed 100 samples (AST: 85, Fallback: 15)

Graph Statistics & GNN Batch Size Recommendation
  total_samples: 100
  avg_nodes: 127.3
  recommended_batch_size: 32
```

### Test 3: Quick training test (10 min)
```bash
python training/train_transformer.py --train-data data/processed/codexglue_test/train.jsonl --val-data data/processed/codexglue_test/valid.jsonl --quick-test --epochs 3 --batch-size 8 --lr 2e-5 --output-dir models/transformer_test
```

**If all 3 tests pass, you're ready for full training!**

---

## ⚡ Summary: What You MUST Do Before Training

### **Mandatory Steps:**

1. ✅ **Build tree-sitter C library**
   ```bash
   python -c "from tree_sitter import Language; from pathlib import Path; Path('build').mkdir(exist_ok=True); Language.build_library('build/my-languages.dll', ['vendor/tree-sitter-c']); print('[+] Built!')"
   ```

2. ✅ **Run quick preprocessing test**
   ```bash
   python training/scripts/data/preprocess_codexglue.py --input-dir data/raw/codexglue --output-dir data/processed/codexglue_test --quick-test
   ```

3. ✅ **Decide training environment:**
   - Local CPU (slow but simple)
   - Google Colab (fast and free)
   - AWS SageMaker (fastest, ~$1.40)

### **SageMaker-Specific Steps (ONLY if using AWS):**

1. ⚠️ Install AWS tools: `pip install boto3 sagemaker awscli`
2. ⚠️ Configure AWS CLI: `aws configure`
3. ⚠️ Create IAM role (see Section C above)
4. ⚠️ Create S3 bucket
5. ⚠️ Add S3 flags to training commands

### **For Local Training (Default):**

✅ **NO additional setup needed!** Just build tree-sitter and run.

---

## 🚦 Ready Check

Run this final check before starting:

```bash
python -c "
from pathlib import Path
import sys

print('StreamGuard Pre-Flight Ready Check')
print('=' * 50)

checks = {
    'CodeXGLUE data exists': Path('data/raw/codexglue/train.jsonl').exists(),
    'tree-sitter-c cloned': Path('vendor/tree-sitter-c').exists(),
    'tree-sitter library built': Path('build/my-languages.dll').exists(),
}

all_pass = True
for name, status in checks.items():
    symbol = '✓' if status else '✗'
    print(f'{symbol} {name}')
    if not status:
        all_pass = False

print('=' * 50)
if all_pass:
    print('✅ ALL CHECKS PASSED! Ready to train.')
else:
    print('❌ Some checks failed. Complete missing steps above.')
    sys.exit(1)
"
```

**Expected output:**
```
StreamGuard Pre-Flight Ready Check
==================================================
✓ CodeXGLUE data exists
✓ tree-sitter-c cloned
✓ tree-sitter library built
==================================================
✅ ALL CHECKS PASSED! Ready to train.
```

---

## 📞 Need Help?

- **tree-sitter build fails:** Check `vendor/tree-sitter-c` exists, try rebuilding
- **CUDA not available:** This is OK, training will be slower on CPU
- **Dependencies missing:** Run `pip install -r requirements.txt`
- **Memory errors:** Use `--quick-test` or reduce `--batch-size`

---

**Last Updated:** October 24, 2025
**Status:** Pre-Flight Checklist Complete
**Next Step:** Build tree-sitter library → Quick test → Full training
