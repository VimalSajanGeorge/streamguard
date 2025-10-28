# GitHub Advisory Collector - Quick Reference

## Setup (One-Time)

```bash
# Set GitHub token
export GITHUB_TOKEN="your_github_token_here"  # Linux/Mac
$env:GITHUB_TOKEN="your_github_token_here"    # Windows PowerShell

# Verify setup
python training/scripts/collection/github_advisory_collector_enhanced.py --help
```

## Quick Commands

### Interactive Menu
```bash
python scripts/run_github_collector.py
```

### Quick Test (100 samples, ~5 min)
```bash
python scripts/run_github_collector.py
# Select option 1
```

### Small Collection (1,000 samples, ~30 min)
```bash
python training/scripts/collection/github_advisory_collector_enhanced.py \
    --target-samples 1000
```

### Full Collection (10,000 samples, ~4-6 hours)
```bash
python training/scripts/collection/github_advisory_collector_enhanced.py
```

### Custom Collection
```bash
python training/scripts/collection/github_advisory_collector_enhanced.py \
    --target-samples 5000 \
    --output-dir data/custom/github \
    --no-cache
```

## Programmatic Usage

### Basic Collection
```python
from github_advisory_collector_enhanced import GitHubAdvisoryCollectorEnhanced

collector = GitHubAdvisoryCollectorEnhanced()
samples = collector.collect()  # 10,000 samples
```

### Ecosystem-Specific
```python
collector = GitHubAdvisoryCollectorEnhanced()
samples = collector.collect_by_ecosystem_severity(
    ecosystem="PIP",
    severity="HIGH",
    max_samples=1000
)
```

### Code Extraction
```python
collector = GitHubAdvisoryCollectorEnhanced()
vuln_code, fixed_code = collector.extract_code_with_diff(
    package_name="package-name",
    ecosystem="PIP",
    vulnerable_range="< 1.2.3",
    patched_version="1.2.3",
    references=["https://github.com/owner/repo"]
)
```

## Supported Ecosystems

- **PIP** - Python packages
- **NPM** - JavaScript/Node.js packages
- **MAVEN** - Java packages
- **RUBYGEMS** - Ruby packages
- **GO** - Go modules
- **COMPOSER** - PHP packages
- **NUGET** - .NET packages
- **CARGO** - Rust crates

## Severity Levels

- **LOW** - Minor vulnerabilities
- **MODERATE** - Medium-impact vulnerabilities
- **HIGH** - High-impact vulnerabilities
- **CRITICAL** - Critical security issues

## Output Files

- **Main**: `data/raw/github/github_advisories.jsonl`
- **Intermediate**: `data/raw/github/github_advisories_intermediate_<timestamp>.jsonl`
- **Cache**: `data/raw/github/.cache/`

## Common Issues

### Missing Token
```bash
# Error: GITHUB_TOKEN environment variable is required
export GITHUB_TOKEN="your_token"  # Fix
```

### Rate Limit
```
# Wait time displayed in output
# Or enable caching to reduce API calls
--no-cache  # Disable (increases API calls)
# Default: caching enabled
```

### Import Error
```python
# Add to path
import sys
sys.path.insert(0, 'training/scripts/collection')
from github_advisory_collector_enhanced import GitHubAdvisoryCollectorEnhanced
```

## Tips

1. **Start small**: Test with 100 samples first
2. **Use cache**: Dramatically reduces API calls on reruns
3. **Monitor progress**: Check intermediate files during long runs
4. **Interruption**: Safe to interrupt (Ctrl+C) - cache allows resumption
5. **Quality**: 30-40% code extraction success rate is normal

## Examples

See complete examples:
```bash
python training/scripts/collection/example_github_usage.py
```

7 examples available:
1. Basic usage
2. Small collection
3. Ecosystem-specific
4. High-severity only
5. Custom filtering
6. Code extraction
7. With statistics

## Documentation

- **Full Guide**: `docs/github_advisory_collector_guide.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Scripts**: `scripts/README.md`
- **Tests**: `tests/test_github_advisory_collector.py`

## Statistics

After collection, view:
- Total samples collected
- Success/failure rates
- Distribution by ecosystem
- Distribution by severity
- Error count and details

## Testing

```bash
# Run all tests
pytest tests/test_github_advisory_collector.py -v

# Run specific test
pytest tests/test_github_advisory_collector.py::test_initialization -v
```

All 20 tests should pass.

## Performance

- **Time**: ~4-6 hours for 10K samples (rate limited)
- **API calls**: ~1-2 GraphQL points per advisory
- **Memory**: ~100MB active
- **Output size**: ~10MB for 10K samples
- **Cache benefit**: 70-90% reduction in API calls

## Get Token

1. Go to GitHub Settings
2. Developer settings > Personal access tokens
3. Generate new token (classic)
4. Select scopes: `public_repo`, `read:packages`
5. Copy token and set as environment variable

Never commit tokens to version control!

---

**Quick Start**: `python scripts/run_github_collector.py`

**Help**: `python training/scripts/collection/github_advisory_collector_enhanced.py --help`

**Status**: ✅ Production Ready
