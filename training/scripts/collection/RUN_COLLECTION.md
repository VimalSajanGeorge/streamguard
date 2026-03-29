# StreamGuard Data Collection Execution Guide

## Prerequisites

Before running the collection, ensure you have completed:

1. ✓ Created `.env` file with API keys (see `SETUP_GUIDE.md`)
2. ✓ Installed dependencies: `pip install gitpython requests python-dotenv rich tqdm`
3. ✓ Verified setup: `python pre_collection_check.py`

## Quick Start

### Option 1: Full Parallel Collection (Recommended)

Run all 6 collectors in parallel (~12-24 hours):

```bash
cd training/scripts/collection
python master_orchestrator.py \
    --collectors cve github repo synthetic osv exploitdb \
    --output-dir ../../data/raw \
    --parallel
```

### Option 2: Sequential Collection (Safer)

Run collectors one by one (~24-48 hours):

```bash
python master_orchestrator.py \
    --collectors synthetic osv exploitdb github cve repo \
    --output-dir ../../data/raw \
    --sequential
```

### Option 3: Individual Collectors (For Testing)

Test each collector individually:

```bash
# 1. Synthetic (fastest, ~5 minutes)
python synthetic_generator.py \
    --output-dir ../../data/raw/synthetic \
    --samples 5000

# 2. OSV (~2-4 hours)
python osv_collector.py \
    --output-dir ../../data/raw/osv \
    --samples 20000

# 3. ExploitDB (~2-4 hours)
python exploitdb_collector.py \
    --output-dir ../../data/raw/exploitdb \
    --samples 10000

# 4. GitHub Advisories (~4-6 hours)
python github_advisory_collector_enhanced.py \
    --output-dir ../../data/raw/github \
    --samples 10000

# 5. CVE/NVD (~6-8 hours)
python cve_collector_enhanced.py \
    --output-dir ../../data/raw/cves \
    --samples 15000

# 6. Repository Mining (~8-12 hours)
python repo_miner_enhanced.py \
    --output-dir ../../data/raw/opensource
```

## Monitoring Progress

### Real-Time Dashboard

The master orchestrator includes a Rich-based progress dashboard that shows:
- Per-collector progress bars
- Samples collected vs target
- Error counts
- Estimated time remaining

### Manual Progress Check

Check collection results at any time:

```bash
# View overall results
cat ../../data/raw/collection_results.json

# Check individual collector stats
cat ../../data/raw/cves/cve_stats.json
cat ../../data/raw/github/github_stats.json
# ... etc
```

### Sample Counts

```bash
# Count samples in each source
wc -l ../../data/raw/*/*.jsonl
```

## Handling Interruptions

### Resume from Checkpoint

If collection is interrupted, resume with:

```bash
python master_orchestrator.py \
    --resume \
    --collectors cve github repo osv exploitdb \
    --parallel
```

### Partial Results

If interrupted with Ctrl+C, partial results are automatically saved to:
- `../../data/raw/collection_partial_TIMESTAMP.json`

## Expected Output

After successful collection, you should have:

```
data/raw/
├── cves/
│   ├── cve_samples.jsonl          # ~15,000 samples
│   └── cve_stats.json
├── github/
│   ├── github_advisories.jsonl    # ~10,000 samples
│   └── github_stats.json
├── opensource/
│   ├── repos/                     # Cached clones (~5-10 GB)
│   ├── mined_samples.jsonl        # ~20,000 samples
│   └── mined_stats.json
├── osv/
│   ├── osv_samples.jsonl          # ~20,000 samples
│   └── osv_stats.json
├── exploitdb/
│   ├── exploitdb_samples.jsonl    # ~10,000 samples
│   └── exploitdb_stats.json
├── synthetic/
│   ├── synthetic_samples.jsonl    # ~5,000 samples
│   └── synthetic_stats.json
└── collection_results.json        # Overall summary
```

**Total: ~80,000 samples**

## Troubleshooting

### Rate Limit Errors

If you hit rate limits:

**NVD API:**
- Without key: 5 requests per 30 seconds
- With key: 50 requests per 30 seconds
- Solution: Get API key from https://nvd.nist.gov/developers/request-an-api-key

**GitHub API:**
- Without token: 60 requests per hour
- With token: 5000 GraphQL points per hour
- Solution: Create token at https://github.com/settings/tokens

### Network Errors

If collectors fail due to network issues:
1. Check internet connectivity
2. Verify API endpoints are accessible
3. Use `--resume` flag to continue from checkpoint

### Disk Space Issues

If running out of disk space:
1. Free up at least 20 GB
2. Delete cached repos after mining: `rm -rf ../../data/raw/opensource/repos/`
3. Compress JSONL files: `gzip ../../data/raw/*/*.jsonl`

### Memory Issues

If running out of memory:
1. Run collectors sequentially instead of parallel
2. Reduce target samples per collector
3. Close other applications

## Validation

After collection completes, validate the data:

```bash
python validate_collection.py --base-dir ../../data/raw --save-report
```

This checks:
- Sample counts vs targets
- Language distribution (Python/JS ratio)
- Code length ranges
- Duplicate rates
- Label balance

## Next Steps

After successful collection and validation:

1. **Merge datasets:**
   ```bash
   python merge_datasets.py \
       --input-dirs ../../data/raw/cves ../../data/raw/github ../../data/raw/opensource ../../data/raw/osv ../../data/raw/exploitdb ../../data/raw/synthetic \
       --output ../../data/raw/merged/all_samples.jsonl \
       --deduplicate \
       --validate
   ```

2. **Run preprocessing:**
   ```bash
   python ../../preprocessing/enhanced_preprocessing.py \
       --input ../../data/raw/merged/all_samples.jsonl \
       --output ../../data/processed/streamguard/ \
       --tokenizer microsoft/codebert-base \
       --max-seq-len 512
   ```

3. **Start training:**
   - Use the Colab notebook: `StreamGuard_Production_Training.ipynb`
   - Or run locally: `python ../../train_transformer.py`, `python ../../train_gnn.py`, `python ../../train_fusion.py`

## Estimated Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Synthetic | 5 min | No API calls |
| OSV | 2-4 hours | Bulk download |
| ExploitDB | 2-4 hours | CSV + files |
| GitHub | 4-6 hours | Rate limited |
| CVE/NVD | 6-8 hours | Most rate limited |
| Repo Mining | 8-12 hours | Cloning + analysis |
| **Parallel Total** | **12-24 hours** | All collectors at once |
| **Sequential Total** | **24-48 hours** | One at a time |

## Support

If you encounter issues:
1. Check error logs in `collection_results.json`
2. Review collector-specific stats files
3. Run pre-collection check: `python pre_collection_check.py`
4. Consult documentation in `SETUP_GUIDE.md`
