# CVE Collector Quick Start Guide

## Quick Start

### 1. Basic Collection (No Token)

```bash
cd training/scripts/collection
python cve_collector_enhanced.py
```

This will:
- Collect 15,000 CVE samples
- Store data in `data/raw/cves/cve_data.jsonl`
- Use caching to resume interrupted runs
- Take approximately 8-10 hours

### 2. Collection with GitHub Token (Recommended)

First, get a GitHub personal access token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select "public_repo" scope
4. Copy the token

Then run:

```bash
python cve_collector_enhanced.py --github-token YOUR_TOKEN_HERE
```

This reduces collection time to 4-6 hours.

### 3. Test Run (Small Sample)

To test the collector with a small sample:

```python
from cve_collector_enhanced import CVECollectorEnhanced

collector = CVECollectorEnhanced(
    output_dir='C:\\Users\\Vimal Sajan\\streamguard\\data\\raw\\cves',
    cache_enabled=True
)

# Override target for testing
collector.TARGET_SAMPLES = 50
collector.KEYWORDS = ["SQL injection", "XSS"]  # Just 2 keywords

samples = collector.collect()
print(f"Collected {len(samples)} samples")
```

### 4. Run Examples

```bash
python example_cve_usage.py
```

Then select from the menu:
1. Basic collection
2. Collection with GitHub token
3. Collection for specific keywords
4. Analyze existing data
5. Incremental collection

### 5. Check Progress

While collection is running, you can check:

```bash
# Count collected samples
cd data/raw/cves
find . -name "cve_data.jsonl" -exec wc -l {} \;

# Check cache size
ls -lh .cache/
```

### 6. Resume Interrupted Collection

The collector automatically resumes from cache:

```bash
# Just run again - it will use cached data
python cve_collector_enhanced.py
```

## Common Commands

### Check Collection Status

```python
import json
from pathlib import Path

data_file = Path('data/raw/cves/cve_data.jsonl')
if data_file.exists():
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    with_code = sum(1 for s in samples if s.get('vulnerable_code'))
    print(f"Total: {len(samples)}")
    print(f"With code: {with_code}")
```

### Analyze Collected Data

```bash
python example_cve_usage.py
# Select option 4 (Analyze existing collection)
```

### Clear Cache

```bash
# Windows
rmdir /s data\raw\cves\.cache

# Linux/Mac
rm -rf data/raw/cves/.cache
```

## Expected Output

### Console Output

```
================================================================================
Enhanced CVE Collection Started
================================================================================
Target samples: 15000
Time range: 2020-2025
Keywords: 12
Cache enabled: True
================================================================================

Starting parallel collection with 4 processes...

--- Processing keyword: 'SQL injection' ---
  Fetched 150 CVEs from API (index: 0)
  Fetched 150 CVEs from API (index: 150)
  Rate limit reached, waiting 28.3s...
  Fetched 150 CVEs from API (index: 300)
  ...
Collected 1250 samples for 'SQL injection'
Total samples so far: 1250

--- Processing keyword: 'XSS' ---
  ...
```

### Output File Format

Each line in `cve_data.jsonl`:

```json
{
  "cve_id": "CVE-2023-12345",
  "description": "SQL injection vulnerability...",
  "vulnerable_code": "query = 'SELECT * FROM users WHERE id = ' + user_id",
  "fixed_code": "query = 'SELECT * FROM users WHERE id = ?'\ncursor.execute(query, (user_id,))",
  "vulnerability_type": "sql_injection",
  "severity": "HIGH",
  "cvss_score": 8.5,
  "cwes": ["CWE-89"],
  "published_date": "2023-06-15T10:00:00.000",
  "source": "github:owner/repo:abc123",
  "collected_at": "2023-06-20T14:30:00.000"
}
```

## Troubleshooting

### Rate Limit Issues

**Problem**: Collection very slow

**Solution**: Use GitHub token for 80x faster GitHub API access

```bash
python cve_collector_enhanced.py --github-token YOUR_TOKEN
```

### Out of Memory

**Problem**: Process crashes with memory error

**Solution**: Reduce target samples or collect in batches

```python
collector.TARGET_SAMPLES = 5000  # Smaller batch
```

### No Code Samples

**Problem**: Many CVEs don't have code

**Solution**: This is normal. Only 20-30% of CVEs have GitHub references with extractable code.

### Connection Errors

**Problem**: Network timeouts or connection errors

**Solution**: The collector has automatic retry. Just let it run. Check your internet connection.

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'base_collector'`

**Solution**: Run from the correct directory:

```bash
cd training/scripts/collection
python cve_collector_enhanced.py
```

## Performance Tips

1. **Use GitHub Token**: 4-6 hours vs 8-10 hours
2. **Enable Caching**: Default, allows resume
3. **Run Overnight**: Long collection times are normal
4. **Check Network**: Stable connection required
5. **Monitor Progress**: Check output file size periodically

## Next Steps

After collection:

1. **Verify Data Quality**
   ```bash
   python example_cve_usage.py  # Option 4: Analyze
   ```

2. **Prepare for Training**
   - Run data preparation scripts
   - Filter samples by quality
   - Split into train/val/test

3. **Integration**
   - Feed data into StreamGuard training pipeline
   - Combine with other data sources
   - Balance dataset by vulnerability type

## Support

For issues:
1. Check error messages in console output
2. Review `README_CVE_COLLECTOR.md` for details
3. Check StreamGuard documentation
4. Verify API endpoints are accessible

## Useful Links

- NVD API: https://nvd.nist.gov/developers/vulnerabilities
- GitHub API: https://docs.github.com/en/rest
- CVE Database: https://cve.mitre.org/
