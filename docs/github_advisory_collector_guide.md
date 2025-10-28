# Enhanced GitHub Security Advisory Collector Guide

## Overview

The Enhanced GitHub Security Advisory Collector is a sophisticated data collection tool for StreamGuard that harvests vulnerability data from GitHub's Security Advisory Database using the GraphQL API. It extracts before/after code pairs from version diffs to create high-quality training data for vulnerability detection models.

## Features

- **Comprehensive Coverage**: Collects from 8 package ecosystems (PIP, NPM, MAVEN, RUBYGEMS, GO, COMPOSER, NUGET, CARGO)
- **Severity Filtering**: Collects across all severity levels (LOW, MODERATE, HIGH, CRITICAL)
- **Code Extraction**: Automatically extracts vulnerable and fixed code from:
  - Commit diffs
  - Version tag comparisons
  - Pull request patches
- **Smart Repository Discovery**: Automatically finds source repositories for packages
- **Rate Limit Management**: Respects GitHub API rate limits (5000 points/hour)
- **Caching**: Intelligent caching to avoid redundant API calls
- **Error Recovery**: Robust error handling with detailed logging

## Prerequisites

### 1. GitHub Personal Access Token

You need a GitHub Personal Access Token with the following scopes:

- `public_repo` - Access public repositories
- `read:packages` - Read package data
- `read:org` - Read organization data (optional)

**To create a token:**

1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Click "Generate new token (classic)"
3. Select the required scopes
4. Copy the token (you won't see it again!)

### 2. Environment Setup

Set the token as an environment variable:

**Linux/Mac:**
```bash
export GITHUB_TOKEN="your_token_here"
```

**Windows (PowerShell):**
```powershell
$env:GITHUB_TOKEN="your_token_here"
```

**Windows (CMD):**
```cmd
set GITHUB_TOKEN=your_token_here
```

For permanent setup, add to your `.env` file:
```
GITHUB_TOKEN=your_token_here
```

### 3. Dependencies

All required dependencies are in `requirements.txt`:
- `requests>=2.31.0` - HTTP client
- `python-dotenv>=1.0.1` - Environment variable management

## Usage

### Basic Usage

Run the collector with default settings (10,000 samples):

```bash
python training/scripts/collection/github_advisory_collector_enhanced.py
```

### Custom Configuration

```bash
python training/scripts/collection/github_advisory_collector_enhanced.py \
    --output-dir data/raw/github \
    --target-samples 5000 \
    --no-cache
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `data/raw/github` | Output directory for collected data |
| `--target-samples` | `10000` | Target number of samples to collect |
| `--no-cache` | `False` | Disable caching (useful for fresh data) |

### Programmatic Usage

```python
from training.scripts.collection.github_advisory_collector_enhanced import (
    GitHubAdvisoryCollectorEnhanced
)

# Initialize collector
collector = GitHubAdvisoryCollectorEnhanced(
    output_dir="data/raw/github",
    cache_enabled=True
)

# Collect all advisories (default: 10,000 samples)
samples = collector.collect_all_advisories(target_samples=10000)

# Or collect for specific ecosystem and severity
pip_high_samples = collector.collect_by_ecosystem_severity(
    ecosystem="PIP",
    severity="HIGH",
    max_samples=1000
)

# Extract code for specific package
vulnerable_code, fixed_code = collector.extract_code_with_diff(
    package_name="django",
    ecosystem="PIP",
    vulnerable_range="< 3.2.5",
    patched_version="3.2.5",
    references=["https://github.com/django/django/commit/abc123"]
)
```

## Output Format

The collector saves data in JSONL (JSON Lines) format: `github_advisories.jsonl`

Each line is a JSON object with the following structure:

```json
{
  "advisory_id": "GHSA-xxxx-yyyy-zzzz",
  "description": "Detailed vulnerability description",
  "vulnerable_code": "# Code before the fix",
  "fixed_code": "# Code after the fix",
  "ecosystem": "PIP",
  "severity": "HIGH",
  "published_at": "2024-01-15T10:00:00Z",
  "source": "github_advisory",
  "metadata": {
    "package_name": "example-package",
    "vulnerable_range": "< 1.2.3",
    "patched_version": "1.2.3",
    "references": [
      "https://github.com/owner/repo/commit/abc123"
    ],
    "vulnerability_type": "sql_injection"
  }
}
```

## How It Works

### 1. Advisory Collection

```
GraphQL Query → GitHub API
    ↓
Filter by ecosystem + severity
    ↓
Filter by date (last 3 years)
    ↓
Extract advisory metadata
```

### 2. Repository Discovery

For each advisory:
```
Check references for GitHub repo
    ↓ (if not found)
Query package registry (PyPI, npm, etc.)
    ↓
Extract repository URL from package metadata
    ↓
Validate GitHub repository
```

### 3. Code Extraction

```
Check references for commit SHA
    ↓ (if found)
Fetch commit diff → Parse patch → Extract before/after
    ↓ (if not found)
Find version tags → Compare versions → Extract diff
    ↓
Validate code quality (length, content)
    ↓
Return (vulnerable_code, fixed_code)
```

## Rate Limiting

The collector respects GitHub's rate limits:

- **GraphQL API**: 5,000 points per hour
- **REST API**: 5,000 requests per hour

**Points per query:**
- Advisory query: ~1-2 points
- Commit fetch: 1 request
- Tag comparison: 2 requests

**Estimated collection time:**
- 10,000 samples: ~4-6 hours (with rate limiting)
- Progress is saved periodically for interruption recovery

## Caching

The collector uses intelligent caching to speed up repeated runs:

- **Advisory queries**: Cached by ecosystem/severity/cursor
- **Repository lookups**: Cached by package name
- **Commit diffs**: Cached by commit SHA
- **Cache location**: `data/raw/github/.cache/`

**Cache benefits:**
- Reduces API calls by 70-90% on repeated runs
- Speeds up development and testing
- Preserves rate limit quota

**Clear cache:**
```bash
rm -rf data/raw/github/.cache/
```

## Ecosystem-Specific Notes

### Python (PIP)
- Source: PyPI (pypi.org)
- Repository usually in `project_urls.Source` or `home_page`
- Best coverage due to strong GitHub integration

### JavaScript (NPM)
- Source: npm registry
- Repository in `repository.url`
- Excellent coverage, most packages on GitHub

### Java (MAVEN)
- Source: Maven Central
- Uses `groupId:artifactId` format
- Repository info often in POM files

### Ruby (RUBYGEMS)
- Source: RubyGems.org
- Repository in `source_code_uri`
- Good GitHub integration

### Go (GO)
- Source: pkg.go.dev
- Repository is part of import path
- Direct GitHub references

### PHP (COMPOSER)
- Source: Packagist
- Repository in package metadata
- Good coverage

### .NET (NUGET)
- Source: NuGet.org
- Repository info in package metadata
- Varies by package

### Rust (CARGO)
- Source: crates.io
- Repository field required for crates
- Excellent coverage

## Troubleshooting

### "GITHUB_TOKEN environment variable is required"

**Solution:**
```bash
export GITHUB_TOKEN="your_token_here"
```

### Rate Limit Exceeded

**Symptoms:**
- "Rate limit approaching" messages
- Long wait times

**Solutions:**
- Wait for rate limit reset (displayed in output)
- Use caching to reduce API calls
- Reduce `--target-samples` for smaller collections

### No Code Extracted

**Reasons:**
1. Repository not on GitHub
2. No commit references in advisory
3. Version tags not found
4. Code changes too small/large

**Solution:**
- Check cache for partial results
- Review error logs in output
- Some advisories naturally lack code (package-only issues)

### Connection Errors

**Solutions:**
- Check internet connection
- Verify GitHub API status (status.github.com)
- Check firewall/proxy settings
- Ensure token is valid and not expired

## Best Practices

1. **Start Small**: Test with `--target-samples 100` first
2. **Use Caching**: Keep cache enabled during development
3. **Monitor Progress**: Watch for "Saved intermediate results" messages
4. **Check Statistics**: Review final statistics to assess quality
5. **Version Control**: Add `.cache/` to `.gitignore`
6. **Backup Data**: Save collected data before clearing cache
7. **Token Security**: Never commit tokens to version control

## Advanced Usage

### Custom Ecosystem Collection

```python
collector = GitHubAdvisoryCollectorEnhanced()

# Collect only Python high-severity vulnerabilities
python_vulns = []
for severity in ["HIGH", "CRITICAL"]:
    samples = collector.collect_by_ecosystem_severity(
        ecosystem="PIP",
        severity=severity,
        max_samples=2000
    )
    python_vulns.extend(samples)
```

### Parallel Collection

```python
from concurrent.futures import ThreadPoolExecutor

def collect_combo(args):
    ecosystem, severity = args
    collector = GitHubAdvisoryCollectorEnhanced(cache_enabled=True)
    return collector.collect_by_ecosystem_severity(ecosystem, severity, 500)

# Collect in parallel (be careful with rate limits!)
combos = [
    ("PIP", "HIGH"),
    ("NPM", "HIGH"),
    ("MAVEN", "CRITICAL")
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(collect_combo, combos)
```

### Custom Filtering

```python
collector = GitHubAdvisoryCollectorEnhanced()

# Override to add custom filtering
original_process = collector._process_advisory

def custom_process(advisory, ecosystem, severity):
    sample = original_process(advisory, ecosystem, severity)

    # Only keep SQL injection vulnerabilities
    if sample and "sql" in sample["description"].lower():
        return sample
    return None

collector._process_advisory = custom_process
samples = collector.collect()
```

## Output Statistics

After collection, you'll see detailed statistics:

```
COLLECTION STATISTICS
============================================================
Total Advisories Processed: 10245
Successful Code Extractions: 3891
Failed Code Extractions: 6354

By Ecosystem:
  PIP: 1823
  NPM: 2104
  MAVEN: 981
  RUBYGEMS: 543
  GO: 1234
  COMPOSER: 876
  NUGET: 1234
  CARGO: 450

By Severity:
  LOW: 1234
  MODERATE: 2345
  HIGH: 4567
  CRITICAL: 2099
```

**Success rate**: ~30-40% for code extraction is normal
- Many advisories don't have accessible code
- Some are configuration-only vulnerabilities
- Repository access issues

## Integration with StreamGuard

The collected data feeds into StreamGuard's training pipeline:

```
GitHub Advisory Collector
    ↓
data/raw/github/github_advisories.jsonl
    ↓
Data Preprocessing
    ↓
Vulnerability Detection Model Training
    ↓
StreamGuard Agent
```

## Contributing

To improve the collector:

1. **Add ecosystem support**: Update `PACKAGE_REGISTRIES`
2. **Improve code extraction**: Enhance `_parse_patch()`
3. **Add filters**: Extend `_process_advisory()`
4. **Better version parsing**: Implement per-ecosystem version comparison

## References

- [GitHub GraphQL API](https://docs.github.com/en/graphql)
- [GitHub Security Advisories](https://github.com/advisories)
- [GitHub REST API](https://docs.github.com/en/rest)
- [Rate Limiting](https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting)

## Support

For issues or questions:
1. Check the error logs in collection output
2. Review GitHub API status
3. Verify token permissions
4. Check cache integrity

## License

Part of StreamGuard - MIT License
