# Enhanced CVE Collector for StreamGuard

## Overview

The Enhanced CVE Collector is a comprehensive data collection tool that gathers vulnerability samples from the National Vulnerability Database (NVD) API 2.0 and extracts actual vulnerable and fixed code from GitHub commit references.

## Features

- **Parallel Collection**: Collects CVEs using keyword-based parallel processing
- **GitHub Code Extraction**: Automatically extracts before/after code from GitHub commits
- **Rate Limiting**: Respects NVD API rate limits (5 requests per 30 seconds)
- **Caching**: Caches API responses to avoid redundant requests
- **Quality Validation**: Validates code samples for length and content quality
- **Comprehensive Metadata**: Extracts vulnerability type, severity, CWEs, and more

## Target Collection

- **Sample Count**: 15,000 CVEs
- **Time Range**: 2020-2025 (5 years)
- **Keywords**: SQL injection, XSS, command injection, path traversal, SSRF, XXE, CSRF, deserialization, and more

## Usage

### Basic Usage

```bash
python cve_collector_enhanced.py
```

### With Custom Output Directory

```bash
python cve_collector_enhanced.py --output-dir /path/to/output
```

### With GitHub Token (Recommended)

For better GitHub API rate limits, provide a GitHub token:

```bash
python cve_collector_enhanced.py --github-token YOUR_GITHUB_TOKEN
```

### Disable Caching

```bash
python cve_collector_enhanced.py --no-cache
```

## Output Format

Each collected sample is saved in JSONL format with the following structure:

```json
{
  "cve_id": "CVE-2023-12345",
  "description": "SQL injection vulnerability in login form...",
  "vulnerable_code": "query = 'SELECT * FROM users WHERE name = ' + user_input",
  "fixed_code": "query = 'SELECT * FROM users WHERE name = ?', (user_input,)",
  "vulnerability_type": "sql_injection",
  "severity": "HIGH",
  "cvss_score": 8.5,
  "cwes": ["CWE-89"],
  "published_date": "2023-06-15T10:00:00.000",
  "source": "github:owner/repo:commit_hash",
  "collected_at": "2023-06-20T14:30:00.000"
}
```

## Output Files

- **Main Output**: `data/raw/cves/cve_data.jsonl`
- **Cache Directory**: `data/raw/cves/.cache/`

## Key Components

### CVECollectorEnhanced Class

Main collector class that inherits from `BaseCollector`.

#### Methods

1. **collect()** - Main collection method that orchestrates the entire process
2. **collect_cves_parallel()** - Implements parallel collection by keyword
3. **collect_by_keyword()** - Collects CVEs for a specific vulnerability keyword
4. **extract_cve_data()** - Parses CVE record and extracts relevant data
5. **find_github_references()** - Finds GitHub URLs in CVE references
6. **fetch_code_from_github()** - Fetches before/after code from GitHub commits

### Vulnerability Keywords

The collector searches for CVEs using the following keywords:

- SQL injection
- XSS / Cross-site scripting
- Command injection / Remote code execution
- Path traversal / Directory traversal
- SSRF (Server-Side Request Forgery)
- XXE (XML External Entity)
- CSRF (Cross-Site Request Forgery)
- Deserialization / Insecure deserialization

### Rate Limiting

The collector implements strict rate limiting to comply with API usage policies:

- **NVD API**: 5 requests per 30 seconds (public API)
- **GitHub API**: Standard GitHub rate limits apply (60/hour without token, 5000/hour with token)

### Code Quality Validation

All extracted code samples must meet quality criteria:

- **Minimum Length**: 50 characters
- **Maximum Length**: 10,000 characters
- **Content Check**: Must not be just whitespace or comments
- **Difference Check**: Before and after code must be different

## Error Handling

The collector implements comprehensive error handling:

- Retries failed requests with exponential backoff
- Logs all errors with context information
- Continues collection even if individual CVEs fail
- Provides detailed error statistics at the end

## Caching Strategy

To minimize API requests and speed up repeated runs:

- **NVD API responses** are cached by keyword and pagination parameters
- **GitHub commit data** is cached by repository and commit hash
- Cache files are stored in `.cache/` subdirectory
- Cache keys are MD5 hashes of request parameters

## Performance Considerations

### Expected Runtime

With rate limiting and GitHub code extraction:
- **Without GitHub token**: ~8-10 hours for 15,000 samples
- **With GitHub token**: ~4-6 hours for 15,000 samples

### Optimization Tips

1. **Use GitHub Token**: Significantly increases GitHub API rate limits
2. **Enable Caching**: Reuses previous API responses
3. **Adjust Target Samples**: Reduce target for faster testing
4. **Resume Collection**: Cache allows resuming interrupted collections

## Troubleshooting

### Common Issues

#### Rate Limit Errors

```
Error: API rate limit exceeded
```

**Solution**: The collector automatically waits when rate limits are hit. Be patient or provide a GitHub token.

#### No Code Found

If many CVEs lack code samples:

**Reason**: Not all CVEs have GitHub references with code changes.

**Expected**: ~20-30% of CVEs will have extractable code samples.

#### Connection Timeouts

```
Error: Connection timeout
```

**Solution**: Check internet connection. The collector will retry automatically.

## Example Session

```bash
$ python cve_collector_enhanced.py --github-token ghp_xxxxxxxxxxxx

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
  ...
Collected 1250 samples for 'SQL injection'
Total samples so far: 1250

--- Processing keyword: 'XSS' ---
  Loaded 150 CVEs from cache (index: 0)
  ...

================================================================================
Collection Complete
================================================================================
Total samples collected: 15243
Samples with code: 4521
Errors encountered: 23
Output file: C:\Users\Vimal Sajan\streamguard\data\raw\cves\cve_data.jsonl
================================================================================
```

## Integration with StreamGuard

The collected CVE data integrates with StreamGuard's training pipeline:

1. **Raw Data**: Stored in `data/raw/cves/cve_data.jsonl`
2. **Processing**: Processed by data preparation scripts
3. **Training**: Used to train vulnerability detection models
4. **Evaluation**: Used for model evaluation and testing

## API References

- **NVD API Documentation**: https://nvd.nist.gov/developers/vulnerabilities
- **GitHub API Documentation**: https://docs.github.com/en/rest

## License

This collector is part of the StreamGuard project and follows the project's license terms.

## Support

For issues or questions about the CVE collector:
1. Check the error logs in the output
2. Review the statistics output
3. Examine cached responses in `.cache/` directory
4. Refer to StreamGuard documentation
