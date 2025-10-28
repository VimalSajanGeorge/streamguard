# Enhanced Repository Miner - Quick Start Guide

## Quick Start

### 1. Install Dependencies

Ensure GitPython is installed:

```bash
pip install GitPython>=3.1.43
```

### 2. Run the Miner

```bash
cd training/scripts/collection
python repo_miner_enhanced.py
```

### 3. Check Output

Output files are saved to `data/raw/opensource/`:
- `mined_samples.jsonl` - All collected samples
- `mined_samples_stats.json` - Collection statistics
- `repos/` - Cached repositories

## Usage Examples

### Basic Command

```bash
# Run with default settings
python repo_miner_enhanced.py
```

### Custom Output Directory

```bash
# Specify output directory
python repo_miner_enhanced.py --output-dir /path/to/output
```

### Disable Caching

```bash
# Don't cache cloned repositories
python repo_miner_enhanced.py --no-cache
```

## What It Does

1. **Clones** 12 open-source repositories (6 Python, 6 JavaScript)
2. **Searches** last 3 years of commits for security keywords
3. **Extracts** vulnerable/fixed code pairs from commit diffs
4. **Validates** code quality and relevance
5. **Saves** ~20,000 samples to JSONL file

## Target Repositories

### Python (16,500 samples)
- django/django (3,500)
- pallets/flask (3,000)
- sqlalchemy/sqlalchemy (3,000)
- psf/requests (2,500)
- tiangolo/fastapi (2,500)
- Pylons/pyramid (2,000)

### JavaScript (16,500 samples)
- expressjs/express (3,500)
- nodejs/node (3,500)
- koajs/koa (2,500)
- fastify/fastify (2,500)
- nestjs/nest (2,500)
- hapijs/hapi (2,000)

## Expected Output

Each sample includes:

```json
{
  "vulnerable_code": "...",
  "fixed_code": "...",
  "commit_sha": "abc123...",
  "commit_message": "Fix SQL injection...",
  "repository": "django/django",
  "file_path": "auth/models.py",
  "vulnerability_type": "sql_injection",
  "committed_date": "2024-03-15T10:30:00+00:00",
  "source": "opensource_repo",
  "language": "python"
}
```

## Performance

- **First run**: 2-4 hours (clones all repositories)
- **Subsequent runs**: 1-2 hours (uses cached repos)
- **Disk space**: ~5-10 GB
- **Output size**: ~50-100 MB

## Common Issues

### Slow Cloning
- Check internet connection
- Consider running overnight

### Out of Space
- Clear cache: `rm -rf data/raw/opensource/repos/`
- Use external drive for output

### Git Errors
- Ensure git is installed: `git --version`
- Check git credentials if accessing private repos

## Testing

Run tests to verify setup:

```bash
python test_repo_miner.py
```

All 6 tests should pass.

## Next Steps

After collection:

1. **Check Statistics**
   ```bash
   cat data/raw/opensource/mined_samples_stats.json
   ```

2. **Preview Samples**
   ```bash
   head -n 5 data/raw/opensource/mined_samples.jsonl
   ```

3. **Count Samples**
   ```bash
   wc -l data/raw/opensource/mined_samples.jsonl
   ```

4. **Analyze by Type**
   ```bash
   grep "sql_injection" data/raw/opensource/mined_samples.jsonl | wc -l
   ```

## Help

For detailed documentation, see `README_REPO_MINER.md`.

For usage examples, see `example_usage.py`.

For issues, check the error log in the stats file.
