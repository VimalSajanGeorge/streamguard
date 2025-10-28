# Enhanced Repository Miner for StreamGuard

## Overview

The Enhanced Repository Miner is a sophisticated tool for mining security-related commits from open-source repositories to extract vulnerable/fixed code pairs for training data. It targets 20,000 high-quality samples from 12 major open-source projects.

## Features

- **Multi-Repository Mining**: Mines 6 Python and 6 JavaScript repositories in parallel
- **Intelligent Commit Detection**: Identifies security-related commits using keyword matching
- **Code Pair Extraction**: Extracts vulnerable and fixed code pairs from commit diffs
- **Quality Validation**: Validates code samples for length and content quality
- **Vulnerability Classification**: Automatically categorizes vulnerabilities by type
- **Multiprocessing Support**: Uses parallel processing for efficient mining
- **Repository Caching**: Caches cloned repositories for faster subsequent runs
- **Comprehensive Statistics**: Generates detailed statistics about collected samples

## Target Repositories

### Python Repositories (16,500 samples)
- **django/django** - Target: 3,500 samples
- **pallets/flask** - Target: 3,000 samples
- **sqlalchemy/sqlalchemy** - Target: 3,000 samples
- **psf/requests** - Target: 2,500 samples
- **tiangolo/fastapi** - Target: 2,500 samples
- **Pylons/pyramid** - Target: 2,000 samples

### JavaScript Repositories (16,500 samples)
- **expressjs/express** - Target: 3,500 samples
- **nodejs/node** - Target: 3,500 samples
- **koajs/koa** - Target: 2,500 samples
- **fastify/fastify** - Target: 2,500 samples
- **nestjs/nest** - Target: 2,500 samples
- **hapijs/hapi** - Target: 2,000 samples

## Security Keywords

The miner identifies commits containing these security-related keywords:

- security
- vulnerability
- CVE
- SQL injection
- XSS
- CSRF
- command injection
- RCE (Remote Code Execution)
- path traversal
- SSRF (Server-Side Request Forgery)
- XXE (XML External Entity)
- fix security
- authentication bypass
- deserialization
- unsafe
- sanitize
- exploit
- malicious
- injection
- buffer overflow
- privilege escalation
- information disclosure

## Output Format

Each sample contains the following fields:

```json
{
  "vulnerable_code": "def authenticate(username, password):\n    query = \"SELECT * FROM users WHERE username='%s'\" % username",
  "fixed_code": "def authenticate(username, password):\n    query = \"SELECT * FROM users WHERE username=?\"\n    cursor.execute(query, (username,))",
  "commit_sha": "abc123def456...",
  "commit_message": "Fix SQL injection vulnerability in authentication",
  "repository": "django/django",
  "file_path": "django/contrib/auth/models.py",
  "vulnerability_type": "sql_injection",
  "committed_date": "2024-03-15T10:30:00+00:00",
  "source": "opensource_repo",
  "language": "python"
}
```

## Usage

### Basic Usage

```bash
cd training/scripts/collection
python repo_miner_enhanced.py
```

### Advanced Usage

```bash
# Specify output directory
python repo_miner_enhanced.py --output-dir /path/to/output

# Disable repository caching
python repo_miner_enhanced.py --no-cache

# Combine options
python repo_miner_enhanced.py --output-dir custom/path --no-cache
```

### Programmatic Usage

```python
from training.scripts.collection.repo_miner_enhanced import EnhancedRepoMiner

# Initialize miner
miner = EnhancedRepoMiner(
    output_dir="data/raw/opensource",
    cache_enabled=True
)

# Collect samples
samples = miner.collect()

# Save samples
miner.save_samples_to_file(samples)

# Get statistics
stats = miner.get_stats()
print(f"Collected {stats['total_samples']} samples")
```

## Output Files

The miner generates the following files in the output directory:

1. **mined_samples.jsonl** - Main output file with all samples (JSONL format)
2. **mined_samples_stats.json** - Statistics about the mining process
3. **repos/** - Directory containing cloned repositories (cached)

## Statistics

The statistics file includes:

- Total samples collected
- Number of errors encountered
- Samples per repository
- Samples per vulnerability type
- Recent errors (last 10)

Example statistics output:

```json
{
  "samples_collected": 20145,
  "errors_count": 3,
  "total_samples": 20145,
  "repositories": 12,
  "samples_per_repo": {
    "django/django": 3489,
    "pallets/flask": 2998,
    "expressjs/express": 3501,
    ...
  },
  "samples_per_vulnerability": {
    "sql_injection": 2341,
    "xss": 1876,
    "command_injection": 1543,
    ...
  }
}
```

## Architecture

### Class: EnhancedRepoMiner

Inherits from `BaseCollector` and implements:

#### Key Methods

1. **mine_all_repositories()** - Mines all configured repositories using multiprocessing
2. **mine_repository(repo_name, config)** - Mines a single repository
3. **find_security_commits(repo)** - Finds commits with security keywords
4. **is_security_commit(commit)** - Checks if a commit is security-related
5. **extract_from_commit(repo, commit, repo_name, language)** - Extracts vulnerable/fixed code pairs

#### Internal Methods

- **_get_repository(repo_name)** - Clones or retrieves cached repository
- **_is_relevant_file(file_path, language)** - Checks file relevance for language
- **_extract_code_from_diff(diff)** - Extracts code from git diff
- **_build_code_snippet(...)** - Builds code snippet with context

## Requirements

### Python Packages

- GitPython >= 3.1.43
- requests >= 2.31.0

These are already included in `requirements.txt`.

### System Requirements

- Git installed and available in PATH
- Internet connection for cloning repositories
- Sufficient disk space (~5-10 GB for all repositories)
- Multicore CPU recommended for parallel processing

## Performance

### Expected Runtime

- First run (cloning repos): 2-4 hours
- Subsequent runs (cached repos): 1-2 hours
- Processing speed: ~150-200 samples per minute

### Optimization Tips

1. Enable caching to avoid re-cloning repositories
2. Use SSD for faster git operations
3. Increase `max_workers` in `mine_all_repositories()` for more parallelism
4. Run on a machine with good internet connection for faster cloning

## Troubleshooting

### Common Issues

**Issue**: Slow cloning
- **Solution**: Check internet connection; consider using a mirror

**Issue**: Out of disk space
- **Solution**: Clear cache directory or use external drive

**Issue**: Git errors
- **Solution**: Ensure git is installed and in PATH

**Issue**: Import errors
- **Solution**: Run from project root; check PYTHONPATH

**Issue**: Memory errors
- **Solution**: Reduce `max_workers` in multiprocessing

## Testing

Run the test suite:

```bash
cd training/scripts/collection
python test_repo_miner.py
```

Tests cover:
- Miner initialization
- Security commit detection
- File relevance checking
- Vulnerability type extraction
- Code validation
- Repository configurations

## Contributing

When modifying the miner:

1. Update security keywords if new vulnerability types are identified
2. Adjust target counts if repository quality changes
3. Update tests when adding new features
4. Document any new configuration options

## License

Part of the StreamGuard project. See main project LICENSE file.
