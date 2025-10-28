# StreamGuard - Complete Data Collection Guide

**Date:** October 24, 2025
**Strategy:** Hybrid Approach (Public Datasets + Custom Collectors)
**Target:** 35,000-45,000 total training samples

---

## Table of Contents

1. [Overview](#overview)
2. [Step 1: Download Public Datasets](#step-1-download-public-datasets)
3. [Step 2: Run Custom Collectors](#step-2-run-custom-collectors)
4. [Step 3: Merge All Datasets](#step-3-merge-all-datasets)
5. [Step 4: Verify and Proceed to Training](#step-4-verify-and-proceed-to-training)
6. [Timeline Estimates](#timeline-estimates)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Hybrid Data Collection Strategy

**Why Hybrid?**
- **Public datasets:** High-quality, well-labeled, immediately available (27K-48K samples)
- **Custom collectors:** Fresh, diverse, domain-specific data (8K-9K samples)
- **Combined strength:** Balanced dataset with broad coverage

### Dataset Composition

| Source | Samples | Quality | Speed | Purpose |
|--------|---------|---------|-------|---------|
| **Devign** | ~27,000 | ⭐⭐⭐⭐⭐ | Instant | C/C++ vulnerabilities |
| **CodeXGLUE** | ~21,000 | ⭐⭐⭐⭐⭐ | Instant | Defect detection benchmark |
| **Synthetic** | 3,000 | ⭐⭐⭐⭐ | Fast | Controlled vulnerability patterns |
| **OSV** | 4,000 | ⭐⭐⭐⭐ | Slow | Real-world multi-language vulns |
| **ExploitDB** | 2,000 | ⭐⭐⭐⭐ | Medium | Exploit code samples |
| **TOTAL** | **~57,000** | High | Mixed | Comprehensive training set |

---

## Step 1: Download Public Datasets

### Quick Start (Recommended)

```bash
# Navigate to project root
cd "c:\Users\Vimal Sajan\streamguard"

# Install dependencies
pip install tqdm requests

# Download Devign and CodeXGLUE (auto-download, ~2-5 minutes)
python training/scripts/collection/download_public_datasets.py --datasets devign codexglue
```

### Expected Output

```
======================================================================
StreamGuard - Public Dataset Downloader
======================================================================

Output directory: data/public
Datasets to download: devign, codexglue

======================================================================
Downloading Devign Dataset
======================================================================

Downloading from: https://raw.githubusercontent.com/saikat107/Devign/master/Devign.json
Saving to: data\public\devign\Devign.json
[████████████████████████████████████████] 100%

Processing Devign dataset...
[+] Processed 27,000 samples:
    Vulnerable: 13,500
    Safe: 13,500
    Output: data\public\devign\devign_processed.jsonl

======================================================================
Downloading CodeXGLUE Defect Detection Dataset
======================================================================

Downloading from: https://raw.githubusercontent.com/microsoft/CodeXGLUE/.../train.jsonl
[████████████████████████████████████████] 100%

Processing CodeXGLUE dataset...
[+] Processed 21,854 total samples:
    Vulnerable: 10,927
    Safe: 10,927
    Output: data\public\codexglue\codexglue_processed.jsonl

======================================================================
[+] Summary report saved: data\public\DOWNLOAD_SUMMARY.md
[+] Total samples downloaded: 48,854
======================================================================
```

### Verify Downloaded Data

```bash
# Count samples
powershell -Command "Get-Content 'data\public\devign\devign_processed.jsonl' | Measure-Object -Line"
powershell -Command "Get-Content 'data\public\codexglue\codexglue_processed.jsonl' | Measure-Object -Line"

# View sample data
python -c "import json; print(json.dumps(json.loads(open('data/public/devign/devign_processed.jsonl').readline()), indent=2)[:500])"
```

### Download All Datasets (Including Info for Juliet/SARD)

```bash
# Download all (includes Juliet and SARD information files)
python training/scripts/collection/download_public_datasets.py --all
```

**Note:** Juliet and SARD require manual download due to size. See generated README.md files for instructions.

---

## Step 2: Run Custom Collectors

### Overview

Now collect 8,000-9,000 fresh samples from our custom collectors to complement the public datasets.

### Recommended Collection Plan

**Target:** 9,000 samples
**Distribution:**
- Synthetic: 3,000 samples (~5 minutes)
- OSV: 4,000 samples (~5 hours with API calls)
- ExploitDB: 2,000 samples (~1.2 hours)

**Total Time:** ~6-7 hours (run overnight recommended)

### Complete Collection Command

```bash
# Navigate to project root
cd "c:\Users\Vimal Sajan\streamguard"

# Run all three collectors sequentially
python training/scripts/collection/run_full_collection.py \
  --collectors synthetic osv exploitdb \
  --synthetic-samples 3000 \
  --osv-samples 4000 \
  --exploitdb-samples 2000 \
  --sequential \
  --no-dashboard
```

### Step-by-Step (Alternative)

If you prefer to run collectors individually:

#### 2.1 Synthetic Collector (~5 minutes)

```bash
python training/scripts/collection/run_full_collection.py \
  --collectors synthetic \
  --synthetic-samples 3000 \
  --sequential \
  --no-dashboard
```

**Expected Output:**
```
Generating 3000 synthetic samples...
Generated 3000 total samples
Vulnerable: 1500
Safe: 1500

Saved 3000 samples to: data\raw\synthetic\synthetic_data.jsonl

[+] Collection Complete!
Total Samples: 3000
Duration: ~5 minutes
```

#### 2.2 OSV Collector (~5 hours)

```bash
python training/scripts/collection/run_full_collection.py \
  --collectors osv \
  --osv-samples 4000 \
  --sequential \
  --no-dashboard
```

**Expected Output:**
```
Target per ecosystem: 400

============================================================
Collecting: PyPI
============================================================
Downloading vulnerability list from GCS...
Found 17,060 vulnerabilities for PyPI
Will collect up to 400 samples
  Processed 10/400 - Success: 10 (100.0%)
  Processed 20/400 - Success: 20 (100.0%)
  ...

[Progress continues for all 10 ecosystems]

Total unique samples collected: 4000
Saved to: data\raw\osv\osv_vulnerabilities.jsonl

[+] Collection Complete!
Duration: ~5 hours
```

**Note:** OSV is slow due to API calls. This is normal. Consider running overnight.

#### 2.3 ExploitDB Collector (~1.2 hours)

```bash
python training/scripts/collection/run_full_collection.py \
  --collectors exploitdb \
  --exploitdb-samples 2000 \
  --sequential \
  --no-dashboard
```

**Expected Output:**
```
============================================================
Fetching ExploitDB CSV database...
============================================================
Downloaded from GitLab
Found 46,920 exploit entries
Filtered to 14,916 code-based exploits

Final: 2000/2000 successful (100.0%)
Saved to: data\raw\exploitdb\exploitdb_exploits.jsonl

[+] Collection Complete!
Duration: ~1.2 hours
```

### Monitor Progress During Collection

While collection is running, monitor in another terminal:

```bash
# Watch file sizes grow
dir data\raw\synthetic\*.jsonl
dir data\raw\osv\*.jsonl
dir data\raw\exploitdb\*.jsonl

# Count samples collected so far
powershell -Command "if (Test-Path 'data\raw\osv\osv_vulnerabilities.jsonl') { (Get-Content 'data\raw\osv\osv_vulnerabilities.jsonl' | Measure-Object -Line).Lines } else { 'Not started yet' }"
```

---

## Step 3: Merge All Datasets

### Combine Public + Collector Data

Once all collections are complete, merge everything into a unified training dataset:

```bash
python training/scripts/collection/merge_datasets.py \
  --public data/public/devign/devign_processed.jsonl \
  --public data/public/codexglue/codexglue_processed.jsonl \
  --collectors data/raw/synthetic/synthetic_data.jsonl \
  --collectors data/raw/osv/osv_vulnerabilities.jsonl \
  --collectors data/raw/exploitdb/exploitdb_exploits.jsonl \
  --output data/training/merged_dataset.jsonl \
  --balance \
  --test-split 0.2 \
  --seed 42
```

### Expected Output

```
======================================================================
StreamGuard - Dataset Merger
======================================================================

======================================================================
Loading Public Datasets
======================================================================

Loading: data/public/devign/devign_processed.jsonl
  [+] Loaded 27,000 samples

Loading: data/public/codexglue/codexglue_processed.jsonl
  [+] Loaded 21,854 samples

======================================================================
Loading Collector Datasets
======================================================================

Loading: data/raw/synthetic/synthetic_data.jsonl
  [+] Loaded 3,000 samples

Loading: data/raw/osv/osv_vulnerabilities.jsonl
  [+] Loaded 4,000 samples

Loading: data/raw/exploitdb/exploitdb_exploits.jsonl
  [+] Loaded 2,000 samples

Balancing dataset...
  Before balancing:
    Vulnerable: 28,927
    Safe: 28,927
  After balancing:
    Vulnerable: 28,927
    Safe: 28,927
    Total: 57,854

Dataset split:
  Train: 46,283 samples (80%)
  Test: 11,571 samples (20%)

[+] Saved 46,283 samples to: data\training\merged_dataset_train.jsonl
[+] Saved 11,571 samples to: data\training\merged_dataset_test.jsonl

======================================================================
Dataset Statistics
======================================================================

Total Samples: 57,854

By Source:
  public_devign            27,000 (46.7%)
  public_codexglue         21,854 (37.8%)
  collector_osv             4,000 ( 6.9%)
  collector_synthetic       3,000 ( 5.2%)
  collector_exploitdb       2,000 ( 3.5%)

By Severity:
  HIGH                     28,927 (50.0%)
  SAFE                     28,927 (50.0%)

By Ecosystem:
  C/C++                    48,854 (84.4%)
  PyPI                        400 ( 0.7%)
  npm                         400 ( 0.7%)
  [... more ecosystems ...]

======================================================================

[+] Merge complete!
```

### Merge Options Explained

| Option | Purpose | Example |
|--------|---------|---------|
| `--public` | Public dataset files | `--public devign.jsonl codexglue.jsonl` |
| `--collectors` | Collector dataset files | `--collectors synthetic.jsonl osv.jsonl` |
| `--output` | Output file path | `--output merged.jsonl` |
| `--balance` | Balance vulnerable/safe | Ensures 50/50 split |
| `--test-split` | Create train/test split | `--test-split 0.2` = 80/20 split |
| `--seed` | Random seed | `--seed 42` for reproducibility |

---

## Step 4: Verify and Proceed to Training

### Verify Merged Dataset

```bash
# Count total samples
powershell -Command "(Get-Content 'data\training\merged_dataset_train.jsonl' | Measure-Object -Line).Lines"
powershell -Command "(Get-Content 'data\training\merged_dataset_test.jsonl' | Measure-Object -Line).Lines"

# View sample structure
python -c "import json; d=json.loads(open('data/training/merged_dataset_train.jsonl').readline()); print(json.dumps(d, indent=2)[:800])"

# Check data quality
python training/scripts/collection/show_examples.py --dataset data/training/merged_dataset_train.jsonl
```

### Expected Dataset Structure

Each sample should have:

```json
{
  "vulnerability_id": "DEVIGN-12345",
  "description": "Buffer overflow in function...",
  "vulnerable_code": "void foo() { char buf[10]; strcpy(buf, input); }",
  "fixed_code": "void foo() { char buf[10]; strncpy(buf, input, 9); buf[9] = 0; }",
  "ecosystem": "C/C++",
  "severity": "HIGH",
  "source": "devign",
  "metadata": {
    "dataset_source": "public_devign",
    "project": "FFmpeg",
    "cwe": "CWE-120"
  }
}
```

### Proceed to Model Training

Once verification is complete:

```bash
# Navigate to training directory
cd training

# Start training with merged dataset
python train_model.py \
  --train-data ../data/training/merged_dataset_train.jsonl \
  --test-data ../data/training/merged_dataset_test.jsonl \
  --model-name streamguard-v1 \
  --epochs 10 \
  --batch-size 16
```

---

## Timeline Estimates

### Quick Collection (Minimal - 10 minutes)

**For testing only**

```bash
# Public datasets only
python download_public_datasets.py --datasets devign codexglue
# Time: ~5 minutes
# Samples: 48,854
```

### Recommended Collection (6-7 hours)

**Best for initial production training**

```bash
# 1. Download public datasets (~5 minutes)
python download_public_datasets.py --datasets devign codexglue

# 2. Run collectors (~6 hours)
python run_full_collection.py \
  --collectors synthetic osv exploitdb \
  --synthetic-samples 3000 \
  --osv-samples 4000 \
  --exploitdb-samples 2000 \
  --sequential --no-dashboard

# 3. Merge datasets (~1 minute)
python merge_datasets.py \
  --public data/public/*/\*_processed.jsonl \
  --collectors data/raw/*/\*.jsonl \
  --output data/training/merged_dataset.jsonl \
  --balance --test-split 0.2
```

**Total Time:** ~6-7 hours
**Total Samples:** ~57,000
**Recommendation:** Run overnight

### Full Collection (12-15 hours)

**For maximum dataset size**

```bash
# Increase collector samples
python run_full_collection.py \
  --collectors synthetic osv exploitdb \
  --synthetic-samples 5000 \
  --osv-samples 8000 \
  --exploitdb-samples 3000 \
  --sequential --no-dashboard
```

**Total Time:** ~12-15 hours
**Total Samples:** ~65,000

---

## Complete Command Reference

### All-in-One Script (Recommended)

Create a batch file for easy execution:

**File:** `collect_all_data.bat`

```batch
@echo off
echo ======================================================================
echo StreamGuard - Complete Data Collection
echo ======================================================================
echo.

echo Step 1: Downloading public datasets...
python training\scripts\collection\download_public_datasets.py --datasets devign codexglue

echo.
echo Step 2: Running custom collectors (this will take 6-7 hours)...
python training\scripts\collection\run_full_collection.py ^
  --collectors synthetic osv exploitdb ^
  --synthetic-samples 3000 ^
  --osv-samples 4000 ^
  --exploitdb-samples 2000 ^
  --sequential ^
  --no-dashboard

echo.
echo Step 3: Merging all datasets...
python training\scripts\collection\merge_datasets.py ^
  --public data\public\devign\devign_processed.jsonl ^
  --public data\public\codexglue\codexglue_processed.jsonl ^
  --collectors data\raw\synthetic\synthetic_data.jsonl ^
  --collectors data\raw\osv\osv_vulnerabilities.jsonl ^
  --collectors data\raw\exploitdb\exploitdb_exploits.jsonl ^
  --output data\training\merged_dataset.jsonl ^
  --balance ^
  --test-split 0.2 ^
  --seed 42

echo.
echo ======================================================================
echo [+] Data collection complete!
echo ======================================================================
echo.
echo Train dataset: data\training\merged_dataset_train.jsonl
echo Test dataset: data\training\merged_dataset_test.jsonl
echo.
echo Next step: cd training && python train_model.py
echo ======================================================================
pause
```

**Run with:**
```bash
collect_all_data.bat
```

---

## Troubleshooting

### Issue 1: Public Dataset Download Fails

**Symptoms:** HTTP errors, timeout, connection refused

**Solutions:**
```bash
# Retry with longer timeout
python download_public_datasets.py --datasets devign --timeout 300

# Manual download alternative
# Visit: https://github.com/saikat107/Devign
# Download Devign.json manually and place in data/public/devign/
```

### Issue 2: OSV Collector Hangs or Timeouts

**Symptoms:** Stuck at 0%, no progress for minutes

**Solutions:**
```bash
# Reduce sample count
python run_full_collection.py --collectors osv --osv-samples 1000

# Run with resume capability
python run_full_collection.py --collectors osv --osv-samples 4000 --resume

# Check network connectivity
ping osv.dev
```

### Issue 3: ExploitDB Collection Slow

**Symptoms:** Very slow progress, GitLab rate limiting

**Solutions:**
```bash
# Reduce sample count
python run_full_collection.py --collectors exploitdb --exploitdb-samples 1000

# Run at different time (avoid peak hours)

# Use cached database if available
# ExploitDB caches the CSV database after first download
```

### Issue 4: Merge Fails - File Not Found

**Symptoms:** "File not found" errors during merge

**Solutions:**
```bash
# Verify all files exist
dir data\public\devign\devign_processed.jsonl
dir data\public\codexglue\codexglue_processed.jsonl
dir data\raw\synthetic\synthetic_data.jsonl
dir data\raw\osv\osv_vulnerabilities.jsonl
dir data\raw\exploitdb\exploitdb_exploits.jsonl

# Re-run missing collectors
python run_full_collection.py --collectors <missing_collector> --resume
```

### Issue 5: Out of Memory During Merge

**Symptoms:** Python crashes, "MemoryError"

**Solutions:**
```bash
# Process in smaller batches
python merge_datasets.py --public data/public/devign/devign_processed.jsonl --output batch1.jsonl
python merge_datasets.py --public data/public/codexglue/codexglue_processed.jsonl --output batch2.jsonl
# Then merge batch1.jsonl and batch2.jsonl

# Or use --max-per-category to limit size
python merge_datasets.py --balance --max-per-category 20000 [other options]
```

---

## Quick Reference Card

### One-Command Data Collection

```bash
# RECOMMENDED: Complete data collection (6-7 hours)
python training/scripts/collection/download_public_datasets.py --datasets devign codexglue && \
python training/scripts/collection/run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 3000 --osv-samples 4000 --exploitdb-samples 2000 --sequential --no-dashboard && \
python training/scripts/collection/merge_datasets.py \
  --public data/public/devign/devign_processed.jsonl data/public/codexglue/codexglue_processed.jsonl \
  --collectors data/raw/synthetic/synthetic_data.jsonl data/raw/osv/osv_vulnerabilities.jsonl data/raw/exploitdb/exploitdb_exploits.jsonl \
  --output data/training/merged_dataset.jsonl --balance --test-split 0.2
```

### Expected Final Output

```
data/
├── public/
│   ├── devign/
│   │   └── devign_processed.jsonl          (27,000 samples)
│   ├── codexglue/
│   │   └── codexglue_processed.jsonl       (21,854 samples)
│   └── DOWNLOAD_SUMMARY.md
├── raw/
│   ├── synthetic/
│   │   └── synthetic_data.jsonl            (3,000 samples)
│   ├── osv/
│   │   └── osv_vulnerabilities.jsonl       (4,000 samples)
│   └── exploitdb/
│       └── exploitdb_exploits.jsonl        (2,000 samples)
└── training/
    ├── merged_dataset_train.jsonl          (46,283 samples - 80%)
    └── merged_dataset_test.jsonl           (11,571 samples - 20%)

TOTAL: ~57,854 samples ready for training!
```

---

## Next Steps After Data Collection

1. ✅ **Verify datasets** are complete and valid
2. ✅ **Review statistics** in merge output
3. ✅ **Proceed to training:**
   ```bash
   cd training
   python train_model.py --train-data ../data/training/merged_dataset_train.jsonl
   ```

---

**Document Version:** 1.0
**Last Updated:** October 24, 2025
**Status:** Ready for production use ✅
