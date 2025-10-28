# StreamGuard Scripts

Collection of utility scripts for running and managing StreamGuard components.

## GitHub Advisory Collector

### Quick Start Script

The `run_github_collector.py` script provides an interactive interface to run the GitHub Security Advisory Collector.

#### Prerequisites

1. **Set GitHub Token**:
   ```bash
   # Linux/Mac
   export GITHUB_TOKEN="your_github_token_here"

   # Windows (PowerShell)
   $env:GITHUB_TOKEN="your_github_token_here"

   # Windows (CMD)
   set GITHUB_TOKEN=your_github_token_here
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Usage

Run the interactive menu:

```bash
python scripts/run_github_collector.py
```

This will present you with options:

1. **Quick Test (100 samples, ~5 minutes)** - Test the setup with a small collection
2. **Small Collection (1,000 samples, ~30 minutes)** - Small production collection
3. **Full Collection (10,000 samples, ~4-6 hours)** - Full production collection
4. **Custom Collection** - Specify your own parameters
5. **Ecosystem-Specific Collection** - Collect for specific languages/ecosystems

#### Direct CLI Usage

You can also run the collector directly:

```bash
# Basic usage (10,000 samples)
python training/scripts/collection/github_advisory_collector_enhanced.py

# Custom sample count
python training/scripts/collection/github_advisory_collector_enhanced.py \
    --target-samples 1000

# Disable caching
python training/scripts/collection/github_advisory_collector_enhanced.py \
    --no-cache

# Custom output directory
python training/scripts/collection/github_advisory_collector_enhanced.py \
    --output-dir data/custom/github
```

#### Output

The collector saves data to:
- **Primary output**: `data/raw/github/github_advisories.jsonl`
- **Intermediate results**: `data/raw/github/github_advisories_intermediate_<timestamp>.jsonl`
- **Cache**: `data/raw/github/.cache/` (if enabled)

Each line in the output file is a JSON object with:

```json
{
  "advisory_id": "GHSA-xxxx-yyyy-zzzz",
  "description": "Vulnerability description",
  "vulnerable_code": "Code before fix",
  "fixed_code": "Code after fix",
  "ecosystem": "PIP",
  "severity": "HIGH",
  "published_at": "2024-01-15T10:00:00Z",
  "source": "github_advisory",
  "metadata": {
    "package_name": "package-name",
    "vulnerable_range": "< 1.2.3",
    "patched_version": "1.2.3",
    "references": ["https://github.com/..."],
    "vulnerability_type": "sql_injection"
  }
}
```

#### Rate Limiting

The GitHub API has rate limits:
- **GraphQL API**: 5,000 points per hour
- **REST API**: 5,000 requests per hour

The collector automatically handles rate limiting and will pause when approaching limits.

#### Troubleshooting

**"GITHUB_TOKEN environment variable is required"**
- Make sure you've set the GITHUB_TOKEN environment variable
- Verify the token is valid and not expired

**Rate limit exceeded**
- Wait for the rate limit to reset (displayed in output)
- Enable caching to reduce API calls
- Reduce the target sample count

**No code extracted**
- This is normal - many advisories don't have accessible code
- Check that the repository is on GitHub
- Some vulnerabilities are package-level only (no code changes)

#### Examples

**Collect Python-only vulnerabilities**:
```bash
python scripts/run_github_collector.py
# Select option 5 (Ecosystem-Specific Collection)
# Choose PIP
# Choose HIGH severity
```

**Quick test before full run**:
```bash
python scripts/run_github_collector.py
# Select option 1 (Quick Test)
```

**Resume interrupted collection**:
```bash
# The collector uses caching, so simply re-run:
python training/scripts/collection/github_advisory_collector_enhanced.py
```

## Other Scripts

Additional scripts will be documented here as they are added.

## Documentation

For detailed information, see:
- [GitHub Advisory Collector Guide](../docs/github_advisory_collector_guide.md)
- [Project Documentation](../docs/)

## Support

For issues or questions:
- Check the documentation
- Review error logs in the console output
- Open an issue on GitHub
