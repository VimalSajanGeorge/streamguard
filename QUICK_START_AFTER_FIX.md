# Quick Start Guide - After Fix

## 🎯 Your Immediate Action Required

### Step 1: Generate New GitHub Token (5 minutes)

1. Go to: https://github.com/settings/tokens
2. Click: **"Generate new token (classic)"**
3. Settings:
   - Name: `StreamGuard Data Collection`
   - Expiration: `90 days`
   - Scopes: ✓ `public_repo` ✓ `read:org` ✓ `security_events`
4. Click **"Generate token"**
5. **COPY THE TOKEN** (you won't see it again!)

### Step 2: Update .env File

Open: `c:\Users\Vimal Sajan\streamguard\.env`

Replace line 35:
```env
# OLD (expired):
GITHUB_TOKEN=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

# NEW:
GITHUB_TOKEN=your_new_token_here
```

### Step 3: Test Token (1 minute)

```bash
cd "c:\Users\Vimal Sajan\streamguard"
python test_github_token.py
```

✅ **Success:** `Status code: 200`
❌ **Failed:** `Status code: 401` → Check token is correct

---

## 🚀 Quick Testing (10 minutes)

### Test All Collectors:

```bash
# 1. GitHub (needs token)
python training/scripts/collection/test_github_collector.py

# 2. OSV (no auth required)
python training/scripts/collection/osv_collector.py --target-samples 10

# 3. ExploitDB (no auth required)
python training/scripts/collection/exploitdb_collector.py --target-samples 10
```

---

## 📊 What You Now Have

### 6 Data Sources (was 4)

| Source | Samples | Auth Required | Status |
|--------|---------|---------------|--------|
| CVE | 15,000 | ❌ No | ✅ Working |
| GitHub | 10,000 | ✅ Yes | ⚠️ Needs token |
| Repos | 20,000 | ❌ No | ✅ Working |
| Synthetic | 5,000 | ❌ No | ✅ Working |
| **OSV** | **20,000** | ❌ **No** | ✅ **NEW!** |
| **ExploitDB** | **10,000** | ❌ **No** | ✅ **NEW!** |
| **TOTAL** | **80,000** | | |

### New Collectors

#### OSV Collector
- **What:** Aggregates 100K+ vulnerabilities from multiple sources
- **Benefit:** More comprehensive than single sources
- **No auth needed:** Free unlimited access
- **File:** `training/scripts/collection/osv_collector.py`

#### ExploitDB Collector
- **What:** 50K+ real exploit code samples
- **Benefit:** Actual attack code (not just descriptions)
- **No auth needed:** Public GitLab repository
- **File:** `training/scripts/collection/exploitdb_collector.py`

---

## ⚡ Quick Collection Commands

### Small Test (100 samples each, ~15 minutes)

```bash
# GitHub (after token fix)
python training/scripts/collection/test_github_collector.py

# OSV
python training/scripts/collection/osv_collector.py --target-samples 100

# ExploitDB
python training/scripts/collection/exploitdb_collector.py --target-samples 100
```

### Medium Collection (2,000 samples each, ~2 hours)

```bash
# OSV
python training/scripts/collection/osv_collector.py --target-samples 2000

# ExploitDB
python training/scripts/collection/exploitdb_collector.py --target-samples 2000

# GitHub (after token fix)
python training/scripts/collection/run_full_collection.py --collectors github --github-samples 2000
```

### Full Collection (80K total, ~12 hours)

```bash
# Run in separate terminals (parallel)

# Terminal 1
python training/scripts/collection/cve_collector_enhanced.py

# Terminal 2 (needs token)
python training/scripts/collection/run_full_collection.py --collectors github

# Terminal 3
python training/scripts/collection/osv_collector.py

# Terminal 4
python training/scripts/collection/exploitdb_collector.py

# Terminal 5
python training/scripts/collection/repo_miner_enhanced.py

# Terminal 6
python training/scripts/collection/synthetic_generator.py
```

---

## 📁 Output Files

After collection, you'll have:

```
data/raw/
├── cves/
│   └── cve_data.jsonl                    # 15,000 samples
├── github/
│   └── github_advisories.jsonl           # 10,000 samples
├── opensource/
│   └── repo_data.jsonl                   # 20,000 samples
├── synthetic/
│   └── synthetic_data.jsonl              # 5,000 samples
├── osv/                                   # NEW!
│   └── osv_vulnerabilities.jsonl         # 20,000 samples
└── exploitdb/                             # NEW!
    └── exploitdb_exploits.jsonl          # 10,000 samples

TOTAL: 80,000 samples
```

---

## 🔍 Troubleshooting

### GitHub 401 Error
**Problem:** `Status code: 401`
**Solution:** Generate new token (see Step 1 above)

### OSV No Data
**Problem:** `No vulnerabilities found`
**Solution:** Check internet connection, OSV API may be down

### ExploitDB Slow
**Problem:** Taking too long
**Solution:** Normal - fetching code from GitLab is slower. Use `--target-samples` to limit

---

## 📚 Full Documentation

- [GITHUB_TOKEN_ISSUE.md](GITHUB_TOKEN_ISSUE.md) - Detailed token fix guide
- [DATASET_COLLECTION_FIX_AND_ENHANCEMENT.md](DATASET_COLLECTION_FIX_AND_ENHANCEMENT.md) - Complete technical details
- [QUICK_START_AFTER_FIX.md](QUICK_START_AFTER_FIX.md) - This file

---

## ✅ Your Checklist

- [ ] Generate new GitHub token
- [ ] Update `.env` file
- [ ] Test token: `python test_github_token.py`
- [ ] Test GitHub collector: `python training/scripts/collection/test_github_collector.py`
- [ ] Test OSV: `python training/scripts/collection/osv_collector.py --target-samples 10`
- [ ] Test ExploitDB: `python training/scripts/collection/exploitdb_collector.py --target-samples 10`
- [ ] Run full collection (optional)

---

## 🎉 Success!

Once the token is fixed:
- ✅ All 6 collectors working
- ✅ 80,000 total samples available
- ✅ Ready for model training

**Questions?** Check the detailed docs above or run with `--help` flag.

---

*Last Updated: October 16, 2025*
