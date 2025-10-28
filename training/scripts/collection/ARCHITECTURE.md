# Enhanced Repository Miner - Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Repository Miner                     │
│                     (EnhancedRepoMiner)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ inherits from
                       ▼
            ┌──────────────────────┐
            │   BaseCollector      │
            │   (Abstract Base)    │
            └──────────────────────┘
```

## Data Flow

```
┌──────────────┐
│   User       │
│   Request    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  1. INITIALIZATION                                        │
│  - Configure repositories (12 repos)                      │
│  - Set security keywords (22 keywords)                    │
│  - Create output directories                              │
│  - Initialize cache                                       │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  2. REPOSITORY MANAGEMENT                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Check      │→ │ Clone or   │→ │ Update     │        │
│  │ Cache      │  │ Load Repo  │  │ (git pull) │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  3. PARALLEL MINING (ProcessPoolExecutor: 4 workers)     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Worker 1   │  │ Worker 2   │  │ Worker 3   │  ...   │
│  │ (Repo 1-3) │  │ (Repo 4-6) │  │ (Repo 7-9) │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  4. COMMIT ANALYSIS (per repository)                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Get all    │→ │ Filter by  │→ │ Check      │        │
│  │ commits    │  │ date       │  │ keywords   │        │
│  │ (3 years)  │  │ (since)    │  │ (security) │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  5. CODE EXTRACTION (per commit)                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Get diff   │→ │ Parse      │→ │ Extract    │        │
│  │ from       │  │ additions  │  │ code pairs │        │
│  │ parent     │  │ & removals │  │            │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  6. QUALITY VALIDATION                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Check      │→ │ Validate   │→ │ Filter     │        │
│  │ file type  │  │ code       │  │ by         │        │
│  │ (.py/.js)  │  │ length     │  │ language   │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  7. CLASSIFICATION                                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Extract    │→ │ Classify   │→ │ Add        │        │
│  │ vuln type  │  │ based on   │  │ metadata   │        │
│  │ from msg   │  │ keywords   │  │            │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  8. AGGREGATION                                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Collect    │→ │ Deduplicate│→ │ Generate   │        │
│  │ all        │  │ by code    │  │ statistics │        │
│  │ samples    │  │ hash       │  │            │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  9. OUTPUT GENERATION                                     │
│  ┌────────────┐  ┌────────────┐                         │
│  │ Save       │  │ Save       │                         │
│  │ samples    │  │ statistics │                         │
│  │ (.jsonl)   │  │ (.json)    │                         │
│  └────────────┘  └────────────┘                         │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Output     │
│   Files      │
└──────────────┘
```

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    EnhancedRepoMiner                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Configuration                                          │    │
│  │  - REPOSITORIES: Dict[str, Dict]                       │    │
│  │  - SECURITY_KEYWORDS: List[str]                        │    │
│  │  - since_date: datetime                                │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Public Methods                                         │    │
│  │  + collect() → List[Dict]                              │    │
│  │  + mine_all_repositories() → List[Dict]                │    │
│  │  + mine_repository(name, config) → List[Dict]          │    │
│  │  + find_security_commits(repo) → List[Commit]          │    │
│  │  + is_security_commit(commit) → bool                   │    │
│  │  + extract_from_commit(...) → List[Dict]               │    │
│  │  + save_samples_to_file(samples, filename)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Private Methods                                        │    │
│  │  - _get_repository(name) → Repo                        │    │
│  │  - _is_relevant_file(path, lang) → bool                │    │
│  │  - _extract_code_from_diff(diff) → Tuple[str, str]     │    │
│  │  - _build_code_snippet(...) → str                      │    │
│  │  - _mine_repository_wrapper(name, config) → List[Dict] │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Inherited from BaseCollector                           │    │
│  │  + validate_code(code) → bool                          │    │
│  │  + extract_vulnerability_type(text) → str              │    │
│  │  + deduplicate_samples(samples) → List[Dict]           │    │
│  │  + save_samples(samples, filename) → Path              │    │
│  │  + get_stats() → Dict                                  │    │
│  │  + log_error(error, context)                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
training/scripts/collection/
│
├── base_collector.py              # Abstract base class
│   └── BaseCollector
│       ├── validate_code()
│       ├── extract_vulnerability_type()
│       ├── deduplicate_samples()
│       ├── save_samples()
│       ├── get_stats()
│       └── log_error()
│
├── repo_miner_enhanced.py         # Main implementation
│   └── EnhancedRepoMiner
│       ├── REPOSITORIES (config)
│       ├── SECURITY_KEYWORDS (config)
│       ├── collect()
│       ├── mine_all_repositories()
│       ├── mine_repository()
│       ├── find_security_commits()
│       ├── is_security_commit()
│       ├── extract_from_commit()
│       └── save_samples_to_file()
│
├── test_repo_miner.py             # Test suite
│   ├── test_miner_initialization()
│   ├── test_security_commit_detection()
│   ├── test_file_relevance()
│   ├── test_vulnerability_extraction()
│   ├── test_code_validation()
│   └── test_repository_config()
│
├── example_usage.py               # Usage examples
│   ├── example_basic_usage()
│   ├── example_custom_output()
│   ├── example_analyze_single_repo()
│   ├── example_statistics()
│   ├── example_filtering()
│   └── example_quality_check()
│
├── README_REPO_MINER.md          # Full documentation
├── QUICKSTART_REPO_MINER.md      # Quick reference
├── ARCHITECTURE.md                # This file
└── __init__.py                    # Package exports
```

## Data Model

### Sample Object

```python
{
    # Code Content (required)
    "vulnerable_code": str,      # Vulnerable version (50-10,000 chars)
    "fixed_code": str,           # Fixed version (50-10,000 chars)

    # Commit Metadata (required)
    "commit_sha": str,           # Full SHA-1 hash (40 chars)
    "commit_message": str,       # Commit message text
    "committed_date": str,       # ISO 8601 datetime

    # File Metadata (required)
    "repository": str,           # org/repo format
    "file_path": str,            # Relative path in repo

    # Classification (required)
    "vulnerability_type": str,   # Classified type
    "language": str,             # "python" or "javascript"

    # Source (required)
    "source": str,               # Always "opensource_repo"
}
```

### Statistics Object

```python
{
    # Collection Metrics
    "samples_collected": int,
    "errors_count": int,
    "total_samples": int,
    "repositories": int,

    # Distributions
    "samples_per_repo": {
        "repo_name": int,
        ...
    },
    "samples_per_vulnerability": {
        "vuln_type": int,
        ...
    },

    # Error Log
    "errors": [
        {
            "timestamp": str,
            "error": str,
            "context": dict
        },
        ...
    ]
}
```

## Processing Pipeline

### 1. Repository Cloning

```
Input: Repository name (e.g., "django/django")
       ↓
Check cache: repos/django_django/
       ↓
If not cached:
    Clone from GitHub: https://github.com/django/django.git
If cached:
    Pull latest changes
       ↓
Output: Git Repo object
```

### 2. Commit Filtering

```
Input: Git Repo object
       ↓
Get all commits since: datetime.now() - 3 years
       ↓
For each commit:
    Check message for security keywords
       ↓
Output: List of security commits
```

### 3. Diff Parsing

```
Input: Security commit
       ↓
Get parent commit
       ↓
Generate diff: parent.diff(commit)
       ↓
For each file in diff:
    Skip if not .py/.js/.ts/.jsx/.tsx
    Parse removed lines (vulnerable code)
    Parse added lines (fixed code)
    Extract context lines
       ↓
Output: List of code pairs
```

### 4. Quality Validation

```
Input: Code pair (vulnerable, fixed)
       ↓
Check vulnerable code:
    Length: 50-10,000 chars
    Non-whitespace: >50% of length
       ↓
Check fixed code:
    Same validations
       ↓
If both pass:
    Output: Valid code pair
Else:
    Discard
```

### 5. Classification

```
Input: Commit message, code pair
       ↓
Check commit message for vulnerability keywords:
    "sql injection" → "sql_injection"
    "xss" → "xss"
    "csrf" → "csrf"
    etc.
       ↓
Output: Classified sample with metadata
```

## Parallel Processing

```
Main Process
    ↓
Create ProcessPoolExecutor(max_workers=4)
    ↓
    ├─→ Worker 1: Mine repos 1-3
    ├─→ Worker 2: Mine repos 4-6
    ├─→ Worker 3: Mine repos 7-9
    └─→ Worker 4: Mine repos 10-12
         ↓
    Collect results as completed
         ↓
    Aggregate all samples
         ↓
    Deduplicate
         ↓
    Generate statistics
         ↓
    Save output
```

## Error Handling

```
Try each repository:
    ├─ Repository clone fails
    │  └─→ Log error, skip repo, continue
    │
    ├─ Git operations fail
    │  └─→ Log error, skip commit, continue
    │
    ├─ Diff parsing fails
    │  └─→ Log error, skip file, continue
    │
    ├─ Code extraction fails
    │  └─→ Log error, skip sample, continue
    │
    └─ Validation fails
       └─→ Silently discard (expected)

All errors logged to:
    - self.errors list
    - Logger output
    - Statistics file
```

## File System Layout

```
data/raw/opensource/
├── repos/                        # Cached repositories
│   ├── django_django/           # Cloned repo
│   ├── pallets_flask/
│   ├── expressjs_express/
│   └── ...
│
├── mined_samples.jsonl          # Output samples (JSONL)
├── mined_samples_stats.json     # Statistics (JSON)
│
└── .cache/                      # BaseCollector cache
    └── (various cache files)
```

## Dependencies

```
External:
    GitPython (git operations)
    requests (HTTP, inherited from BaseCollector)

Python Standard Library:
    json (serialization)
    re (regex, minimal usage)
    os, pathlib (file operations)
    typing (type hints)
    datetime, timedelta (date filtering)
    concurrent.futures (parallel processing)
    logging (output)
    hashlib (deduplication, inherited)

Project:
    BaseCollector (abstract base class)
```

## Performance Characteristics

### Time Complexity

- Repository cloning: O(repo_size)
- Commit iteration: O(commits)
- Diff parsing: O(commits × files × lines)
- Deduplication: O(n) where n = samples
- Overall: O(repos × commits × files × lines)

### Space Complexity

- Repository cache: O(total_repo_size) ≈ 5-10 GB
- Samples in memory: O(samples × avg_size) ≈ 100-200 MB
- Output file: O(samples × avg_size) ≈ 50-100 MB

### Optimization Strategies

1. **Parallel Processing**: 4 workers mine repos simultaneously
2. **Caching**: Repositories cached to avoid re-cloning
3. **Early Filtering**: Invalid files/commits skipped early
4. **Streaming**: Samples saved incrementally (not all at once)
5. **Lazy Loading**: Commits loaded as needed

## Integration Points

### Inputs

- Configuration: REPOSITORIES dict
- Configuration: SECURITY_KEYWORDS list
- User arguments: output_dir, cache_enabled

### Outputs

- File: mined_samples.jsonl (primary output)
- File: mined_samples_stats.json (statistics)
- Directory: repos/ (cached repositories)

### Integration with StreamGuard

```
StreamGuard Training Pipeline
    ↓
Data Collection Phase
    ├─→ CVE Collector (5,000 samples)
    ├─→ GitHub Advisory Collector (5,000 samples)
    ├─→ Enhanced Repository Miner (20,000 samples) ← This component
    └─→ Synthetic Generator (20,000 samples)
         ↓
Data Preprocessing
    ↓
Model Training
```

## Scalability

### Horizontal Scaling

- Increase `max_workers` for more parallelism
- Distribute repositories across multiple machines
- Use shared cache for repositories

### Vertical Scaling

- More memory: Cache more data
- Faster CPU: Process commits faster
- Faster disk: Speed up git operations
- Better network: Faster cloning

### Future Enhancements

1. **Distributed Processing**: Use Celery/Dask for multi-machine mining
2. **Incremental Updates**: Only process new commits since last run
3. **Database Backend**: Store samples in DB instead of files
4. **API Interface**: RESTful API for remote mining
5. **Real-time Monitoring**: Dashboard for mining progress

---

**Architecture Version**: 1.0
**Last Updated**: 2025-10-14
**Status**: Production Ready
