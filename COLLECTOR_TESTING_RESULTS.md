# StreamGuard Data Collector Testing Results

**Date:** October 24, 2025
**Status:** All Collectors Verified Working ✅

---

## Summary

All three working collectors have been tested and verified with small sample counts. The OSV collector has been reverted to the original API-based approach per user request.

---

## Test Configuration

**Testing Approach:**
- Sequential mode (one collector at a time)
- Small sample counts (50 samples each)
- No dashboard (cleaner output)
- Fresh caches (all caches cleared before testing)

**Command Pattern:**
```bash
python run_full_collection.py --collectors <name> --<name>-samples 50 --sequential --no-dashboard
```

---

## Test Results

### 1. Synthetic Collector ✅

**Command:**
```bash
python run_full_collection.py --collectors synthetic --synthetic-samples 50 --sequential --no-dashboard
```

**Results:**
- **Status:** ✅ SUCCESS
- **Samples Collected:** 50/50 (100%)
- **Duration:** 4.8 seconds
- **Rate:** ~10.4 samples/second
- **Errors:** 0

**Sample Distribution:**
- 25 vulnerable samples
- 25 safe samples
- 5 vulnerability types (5 pairs each):
  - SQL Injection (concat)
  - XSS (output)
  - Command Injection
  - Path Traversal
  - SSRF

**Output File:** `data/raw/synthetic/synthetic_data.jsonl`

---

### 2. OSV Collector ✅ (API Mode - Reverted)

**Command:**
```bash
python run_full_collection.py --collectors osv --osv-samples 50 --sequential --no-dashboard
```

**Results:**
- **Status:** ✅ SUCCESS
- **Samples Collected:** 50/50 (100%)
- **Duration:** 216.8 seconds (3.6 minutes)
- **Rate:** ~0.23 samples/second
- **Errors:** 0

**Collection Method:**
- Downloads vulnerability IDs from GCS ZIP files (fast)
- Makes individual API calls to fetch details for each vulnerability (slow)
- 5 samples per ecosystem across 10 ecosystems

**Ecosystem Distribution:**
| Ecosystem | Samples | Total Available |
|-----------|---------|-----------------|
| PyPI      | 5       | 17,060          |
| npm       | 5       | 69,806          |
| Maven     | 5       | 6,053           |
| Go        | 5       | 4,854           |
| crates.io | 5       | 1,895           |
| RubyGems  | 5       | 1,856           |
| Packagist | 5       | 4,935           |
| NuGet     | 5       | 1,501           |
| Hex       | 5       | 40              |
| Pub       | 5       | 12              |

**Severity Distribution:**
- HIGH: 36 (72%)
- UNKNOWN: 14 (28%)

**Output File:** `data/raw/osv/osv_vulnerabilities.jsonl`

---

### 3. ExploitDB Collector ✅

**Command:**
```bash
python run_full_collection.py --collectors exploitdb --exploitdb-samples 50 --sequential --no-dashboard
```

**Results:**
- **Status:** ✅ SUCCESS
- **Samples Collected:** 50/50 (100%)
- **Duration:** 110.1 seconds (1.8 minutes)
- **Rate:** ~0.45 samples/second
- **Errors:** 0

**Collection Method:**
- Downloads CSV database from GitLab (46,920 total exploits)
- Filters to code-based exploits (14,916 available)
- Fetches code via GitLab API for selected exploits

**Platform Distribution:**
- All 50 samples from AIX platform

**Language Distribution:**
- C: 24 (48%)
- Bash: 14 (28%)
- Perl: 7 (14%)
- Ruby: 3 (6%)
- Python: 2 (4%)

**Type Distribution:**
- Local: 45 (90%)
- DoS: 3 (6%)
- Remote: 2 (4%)

**Output File:** `data/raw/exploitdb/exploitdb_exploits.jsonl`

---

## Performance Comparison

| Collector | Samples | Duration | Rate (samples/sec) | Scalability for 1000 samples |
|-----------|---------|----------|--------------------|-----------------------------|
| **Synthetic** | 50 | 4.8s | 10.4 | ~1.6 minutes |
| **OSV** | 50 | 216.8s | 0.23 | ~72 minutes (1.2 hours) |
| **ExploitDB** | 50 | 110.1s | 0.45 | ~37 minutes |

---

## Key Changes Made

### OSV Collector Reverted to API Mode

**File:** `training/scripts/collection/osv_collector.py`

**Changes:**

1. **`collect_by_ecosystem()` method (lines 175-233):**
   - Reverted to fetch vulnerability IDs from GCS
   - Restored API calls to `_fetch_vulnerability_details(vuln_id)` for each sample
   - Progress reporting every 10 samples instead of 100

2. **`_download_ecosystem_vulns_from_gcs()` method (lines 235-305):**
   - Changed return type from `List[Dict]` → `List[str]` (IDs only)
   - Changed storage from `vuln_data_list` → `vuln_ids`
   - Extracts only ID: `vuln_ids.append(vuln_data.get("id"))`
   - Changed cache key from "gcs_vuln_data" → "gcs_vuln_list"
   - Changed cache structure from `{"vuln_data": ...}` → `{"vuln_ids": ...}`

**Why Reverted:**
- User requested reverting the bulk data optimization
- Returned to proven, stable API-based approach
- Slower but allows for incremental testing with small batches

---

## Production Collection Estimates

Based on test results, here are estimates for larger production runs:

### Scenario 1: Small Batch (1,000 samples total)
```bash
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 500 --osv-samples 300 --exploitdb-samples 200 \
  --sequential --no-dashboard
```

**Estimated Duration:**
- Synthetic: 500 samples ÷ 10.4/s = ~48 seconds
- OSV: 300 samples ÷ 0.23/s = ~1,304 seconds (~22 minutes)
- ExploitDB: 200 samples ÷ 0.45/s = ~444 seconds (~7 minutes)
- **Total: ~30 minutes**

---

### Scenario 2: Medium Batch (5,000 samples total)
```bash
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 2000 --osv-samples 2000 --exploitdb-samples 1000 \
  --sequential --no-dashboard
```

**Estimated Duration:**
- Synthetic: 2000 samples ÷ 10.4/s = ~3.2 minutes
- OSV: 2000 samples ÷ 0.23/s = ~145 minutes (~2.4 hours)
- ExploitDB: 1000 samples ÷ 0.45/s = ~37 minutes
- **Total: ~3 hours**

---

### Scenario 3: Large Batch (10,000 samples total)
```bash
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 5000 --osv-samples 3000 --exploitdb-samples 2000 \
  --sequential --no-dashboard
```

**Estimated Duration:**
- Synthetic: 5000 samples ÷ 10.4/s = ~8 minutes
- OSV: 3000 samples ÷ 0.23/s = ~217 minutes (~3.6 hours)
- ExploitDB: 2000 samples ÷ 0.45/s = ~74 minutes (~1.2 hours)
- **Total: ~5 hours**

---

## Recommendations

### For Quick Testing (15-30 minutes)
```bash
# Test run - verify everything works
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 100 --osv-samples 50 --exploitdb-samples 50 \
  --sequential --no-dashboard
```
- **Total samples:** 200
- **Estimated time:** ~25 minutes
- **Purpose:** Verify all collectors work, check data quality

---

### For Development/Training (2-3 hours)
```bash
# Development dataset
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 2000 --osv-samples 1000 --exploitdb-samples 1000 \
  --sequential --no-dashboard
```
- **Total samples:** 4,000
- **Estimated time:** ~2.5 hours
- **Purpose:** Sufficient for model training experiments

---

### For Production Dataset (4-6 hours)
```bash
# Production dataset
python run_full_collection.py --collectors synthetic osv exploitdb \
  --synthetic-samples 5000 --osv-samples 3000 --exploitdb-samples 2000 \
  --sequential --no-dashboard
```
- **Total samples:** 10,000
- **Estimated time:** ~5 hours
- **Purpose:** Full production dataset for model training

---

## Tips for Large Collections

### 1. Run Overnight or During Low-Activity Periods
OSV collector is slow with API calls (~0.23 samples/sec). Plan accordingly.

### 2. Use Sequential Mode for Visibility
```bash
--sequential --no-dashboard
```
- Clearer output showing progress
- Easier to debug if issues occur
- Better for monitoring in terminal

### 3. Monitor Progress
The collectors show progress every 10-100 samples:
```
  Processed 10/50 - Success: 10 (100.0%), Failed: 0
  Processed 20/50 - Success: 20 (100.0%), Failed: 0
```

### 4. Check Output Files During Collection
```bash
# In another terminal, monitor sample counts
dir data\raw\synthetic\*.jsonl
dir data\raw\osv\*.jsonl
dir data\raw\exploitdb\*.jsonl

# Count lines (samples) in real-time
powershell -Command "Get-Content 'data\raw\osv\osv_vulnerabilities.jsonl' | Measure-Object -Line"
```

### 5. Checkpoint System
OSV collector has built-in checkpoints - if interrupted, you can resume:
```bash
python run_full_collection.py --collectors osv --osv-samples 3000 --resume
```

---

## Data Quality Verification

### Check Sample Structure
```python
import json

# Read first sample from each collector
with open('data/raw/synthetic/synthetic_data.jsonl') as f:
    print("Synthetic:", json.loads(f.readline()).keys())

with open('data/raw/osv/osv_vulnerabilities.jsonl') as f:
    print("OSV:", json.loads(f.readline()).keys())

with open('data/raw/exploitdb/exploitdb_exploits.jsonl') as f:
    print("ExploitDB:", json.loads(f.readline()).keys())
```

### View Sample Data
```bash
python training/scripts/collection/show_examples.py
```

---

## Next Steps

1. **Choose a collection size** based on your needs (see recommendations above)
2. **Run the collection** with sequential mode for visibility
3. **Monitor progress** periodically to ensure everything is working
4. **Verify data quality** after collection completes
5. **Proceed to model training** once data collection is complete

---

## Status

✅ **All collectors tested and working**
✅ **OSV reverted to API mode (slower but stable)**
✅ **Ready for production data collection**

---

**Last Updated:** October 24, 2025
**Tested By:** Individual collector runs with 50 samples each
**Next:** User to choose collection size and run full data collection
