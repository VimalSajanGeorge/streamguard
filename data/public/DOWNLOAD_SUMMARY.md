# Public Datasets Download Summary

**Date:** 2025-10-24 09:25:10
**Total Samples Downloaded:** 0

---

## Downloaded Datasets

### ❌ Devign

**Status:** FAILED
**Samples:** 0
**Description:** Graph-based vulnerability detection dataset (C projects)

---

### ✅ CodeXGLUE Defect Detection

**Status:** SUCCESS
**Samples:** 0
**Description:** Microsoft's defect detection benchmark (C projects)
**Output File:** `data\public\codexglue\codexglue_processed.jsonl`

---

## Next Steps

### 1. Verify Downloaded Data

```bash
# Count samples in each dataset
wc -l data/public/devign/devign_processed.jsonl
wc -l data/public/codexglue/codexglue_processed.jsonl
```

### 2. Combine with Collector Data

```bash
# Run collectors to gather 8000-9000 additional samples
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 3000 --osv-samples 4000 --exploitdb-samples 2000 \
  --sequential --no-dashboard
```

### 3. Merge All Datasets

```bash
python merge_datasets.py \
  --public data/public/devign/devign_processed.jsonl \
  --public data/public/codexglue/codexglue_processed.jsonl \
  --collectors data/raw/synthetic/synthetic_data.jsonl \
  --collectors data/raw/osv/osv_vulnerabilities.jsonl \
  --collectors data/raw/exploitdb/exploitdb_exploits.jsonl \
  --output data/training/merged_dataset.jsonl
```

### 4. Start Training

```bash
cd training
python train_model.py --dataset ../data/training/merged_dataset.jsonl
```

---

## Dataset Licenses

- **Devign:** MIT License
- **CodeXGLUE:** MIT License
- **Juliet:** Public Domain (NIST)
- **SARD:** Public Domain (NIST)

All datasets are approved for research and commercial use.
