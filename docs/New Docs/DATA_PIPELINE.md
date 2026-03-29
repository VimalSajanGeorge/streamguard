# docs/DATA_PIPELINE.md — 6-Collector Data Pipeline Specification

> Read this before implementing: any file in training/scripts/collection/
> Also see: STREAMGUARD_IMPLEMENTATION.md for complete collector code specs

---

## Canonical Sample Schema (ENFORCED)

Every sample saved by any collector MUST pass `validate_sample()` in `schema.py`.

```python
# training/scripts/collection/schema.py

from dataclasses import dataclass, field
from typing import Optional
import uuid, re
from datetime import datetime

VALID_CWE = {
    "CWE-89", "CWE-78", "CWE-79", "CWE-119", "CWE-120",
    "CWE-121", "CWE-122", "CWE-125", "CWE-134", "CWE-190",
    "CWE-416", "CWE-476"
}
VALID_SOURCES = {
    "sard", "exploitdb", "cve", "github_advisory",
    "osv", "repo", "manual",
    "sard_cfa", "exploitdb_cfa", "cve_cfa",        # CFA variants
    "github_advisory_cfa", "osv_cfa", "repo_cfa",
}

# Optional fields with default values. validate_sample() does NOT require these.
# They are populated by Stage 3 CFA generation and carried through all downstream stages.
OPTIONAL_FIELD_DEFAULTS = {
    "cfa_tier":           0,      # int 1-5: which CFA generation tier produced this sample
                                   # 0 = not a CFA sample (original or SARD native pair)
                                   # 1 = AST rule  2 = zero-shot  3 = CoT
                                   # 4 = few-shot  5 = critique-refine
    "cfa_quality_score":  1.0,    # float 0.0-1.0 from 7-gate validation
                                   # 1.0 = all gates passed  0.6 = Gate 5 soft fail
    "severity_score":     0.0,    # CVSS float, used by L_severity head in training
    "commit_sha":         "",     # 40-char SHA for cross-source dedup and Stage 7 split
    "cve_id":             "",     # for cross-source dedup and Stage 7 split grouping
    "cfa_type":           "",     # "native" (SARD pairs) | "llm_generated" | "ast_generated"
    "aliases":            {},     # {cve: "...", ghsa: "...", osv: "..."}
    "metadata":           {},     # source-specific extra fields
}

def validate_sample(s: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_errors). All errors must be empty to save.

    Only REQUIRED fields are enforced. Optional fields (cfa_tier, cfa_quality_score,
    severity_score, etc.) are never rejected — they default to 0 or "" if absent.
    """
    errors = []

    # Required fields
    if not s.get("id"):           errors.append("missing id")
    if not s.get("source"):       errors.append("missing source")
    if not s.get("code"):         errors.append("missing code")
    if s.get("label") not in [0, 1]: errors.append("label must be 0 or 1")
    if not s.get("cwe"):          errors.append("missing cwe")
    if not s.get("language"):     errors.append("missing language")
    if not s.get("collected_at"): errors.append("missing collected_at")

    # Value validation
    if s.get("source") and s["source"] not in VALID_SOURCES:
        errors.append(f"invalid source: {s['source']}")
    if s.get("cwe") and s["cwe"] not in VALID_CWE:
        errors.append(f"invalid CWE: {s['cwe']}")
    if s.get("language") and s["language"] != "c":
        errors.append("language must be 'c'")

    # Code sanity
    code = s.get("code", "")
    lines = code.splitlines()
    if len(lines) < 5:   errors.append(f"code too short: {len(lines)} lines")
    if len(lines) > 500: errors.append(f"code too long: {len(lines)} lines")
    tokens = code.split()
    if len(tokens) < 10: errors.append(f"too few tokens: {len(tokens)}")

    return len(errors) == 0, errors

def make_sample_id() -> str:
    return str(uuid.uuid4())

def make_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
```

---

## Base Collector Class

```python
# training/scripts/collection/base_collector.py

import json, os, time, hashlib, logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from loguru import logger

class BaseCollector:
    """
    Base class for all StreamGuard data collectors.
    Provides: checkpoint/resume, HTTP retry, deduplication, validation.
    
    Every collector MUST inherit from this.
    """
    
    def __init__(self, output_dir: str, checkpoint_dir: str, source_name: str):
        self.output_dir     = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.source_name    = source_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._seen_hashes: set = set()  # in-memory dedup
        self._samples_saved = 0
        self._samples_failed = 0
        
        self._load_seen_hashes()
    
    # ── HTTP ─────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                        requests.exceptions.Timeout)),
        reraise=True
    )
    def safe_get(self, url: str, **kwargs) -> requests.Response:
        """GET with automatic retry on connection errors. Handles 429/503."""
        kwargs.setdefault("timeout", 45)
        resp = requests.get(url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited. Sleeping {retry_after}s")
            time.sleep(retry_after + 1)
            resp = requests.get(url, **kwargs)
        if resp.status_code == 401:
            raise ValueError(f"401 Unauthorized. Check API key for {url}")
        return resp
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True
    )
    def safe_post(self, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 45)
        return requests.post(url, **kwargs)
    
    # ── Checkpoint ───────────────────────────────────────────────
    def load_checkpoint(self) -> dict:
        cp_path = self.checkpoint_dir / f"{self.source_name}_checkpoint.json"
        if cp_path.exists():
            return json.loads(cp_path.read_text())
        return {}
    
    def save_checkpoint(self, state: dict):
        """Atomic checkpoint write: .tmp → os.replace"""
        cp_path = self.checkpoint_dir / f"{self.source_name}_checkpoint.json"
        tmp_path = cp_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2))
        os.replace(tmp_path, cp_path)
    
    # ── Sample saving ────────────────────────────────────────────
    def save_sample(self, sample: dict) -> bool:
        """Validate + dedup + save. Returns True if saved."""
        from schema import validate_sample
        
        is_valid, errors = validate_sample(sample)
        if not is_valid:
            logger.debug(f"Invalid sample {sample.get('id','?')}: {errors}")
            return False
        
        code_hash = hashlib.md5(" ".join(sample["code"].split()).encode()).hexdigest()
        if code_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(code_hash)
        
        out_path = self.output_dir / f"{self.source_name}_samples.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(sample) + "\n")
        
        self._samples_saved += 1
        if self._samples_saved % 100 == 0:
            logger.info(f"[{self.source_name}] Saved {self._samples_saved} samples")
            self._check_disk_space()
        return True
    
    def save_failed_item(self, item: dict, reason: str):
        """Save failed items for retry."""
        failed_path = self.output_dir / f"{self.source_name}_failed.jsonl"
        with open(failed_path, "a") as f:
            f.write(json.dumps({"item": item, "reason": reason}) + "\n")
        self._samples_failed += 1
    
    def _check_disk_space(self):
        import shutil
        free_gb = shutil.disk_usage(self.output_dir).free / (1024**3)
        if free_gb < 2.0:
            raise RuntimeError(f"Disk space critical: {free_gb:.1f}GB remaining")
    
    def _load_seen_hashes(self):
        out_path = self.output_dir / f"{self.source_name}_samples.jsonl"
        if out_path.exists():
            import hashlib
            for line in open(out_path):
                try:
                    s = json.loads(line)
                    h = hashlib.md5(" ".join(s["code"].split()).encode()).hexdigest()
                    self._seen_hashes.add(h)
                except: pass
            logger.info(f"[{self.source_name}] Loaded {len(self._seen_hashes)} existing hashes")
```

---

## Collector 1: SARD / Juliet Suite (Start Here — No API)

**File**: `training/scripts/collection/process_sard.py`  
**Priority**: P0 — Process first. 8K–12K samples, fully local.

```
Actual Juliet Suite v1.3.1 on-disk structure:
  data/raw/sard/2022-08-11-juliet-c-cplusplus-.../
    ├── 100000-v1.0.0/src/testcases/CWE121_Stack_Based_.../
    │   ├── CWE121_..._01.c    ← contains BOTH bad() and good() functions
    │   ├── CWE121_..._02.c
    │   └── ...
    ├── 100001-v1.0.0/src/testcases/CWE122_Heap_.../
    └── ...  (thousands of numbered directories)

Key points:
- Label determined by FUNCTION NAME (not filename): bad() → 1, good() → 0
- Each .c file contains both bad and good functions → native CFA pairs
- pair_id assigned per FILE when both labels present
- 7 M1 target CWEs: CWE-78, CWE-121, CWE-122, CWE-134, CWE-190, CWE-416, CWE-476
- tree-sitter walks full AST (handles #ifndef OMITBAD/#endif guards)
- Helper functions (main, printLine, etc.) are skipped via SKIP_NAMES
- Non-UTF-8 files handled via chardet fallback
- No compile-check — SARD uses custom headers from testcasesupport/
- Actual output: 53,666 samples (53K extracted; Stage 1 length filter drops thin wrappers already rejected by collector)
```

---

## Collector 2: ExploitDB

**File**: `training/scripts/collection/exploitdb_collector.py`  
**Priority**: P0 — Process second. Fully local after one-time clone.

```bash
# One-time setup:
git clone --depth=1 https://gitlab.com/exploit-database/exploitdb.git data/raw/exploitdb_repo

# Then process locally — no API calls needed
# Filter: platform in {linux, unix, freebsd, multiple} AND type in {local, remote, dos}
# Language filter: "c" in files_exploits.csv type column
# Generate safe counterpart via rule-based mutation (safer than LLM for ExploitDB)
```

---

## Collector 3: CVE/NVD (Two-Phase)

**File**: `training/scripts/collection/cve_collector_enhanced.py`

**Phase 1**: Query NVD API → build index of CVEs with GitHub commit URLs  
**Phase 2**: Fetch actual C code diffs from GitHub Commits API

```
CRITICAL RULES:
1. Phase 1 saves an INDEX (cve_id + github_commit_url). NOT training samples.
2. Phase 2 saves training samples (before/after C code from commit diffs).
3. NEVER save NVD JSON metadata as a training sample.
4. Filter: only .c and .h files. Status=modified. 5–150 changed lines.
5. Rate limits: NVD = 50 req/30s. GitHub = 5000 pts/hr.

12 CWE filters for NVD query:
CWE-89, CWE-78, CWE-79, CWE-119, CWE-120, CWE-121, CWE-122,
CWE-125, CWE-134, CWE-190, CWE-416, CWE-476
```

---

## Collector 4: GitHub Advisory

**File**: `training/scripts/collection/github_advisory_collector.py`

```
USE GraphQL API (NOT REST). Endpoint: https://api.github.com/graphql
C/C++ projects appear under ecosystem:OTHER — apply C_PROJECT_KEYWORDS filter AFTER fetch.

C_PROJECT_KEYWORDS = {
    "linux", "kernel", "glibc", "openssl", "curl", "ffmpeg",
    "nginx", "apache", "sqlite", "zlib", "libpng", "libjpeg",
    "libxml", "pcre", "openssh", "php", "tcpdump", "wireshark",
}

Monitor rate limits EVERY 50 pages: if remaining < 400 points, sleep until reset+30s.
```

---

## Collector 5: OSV

**File**: `training/scripts/collection/osv_collector.py`

```
Query only C ecosystems: ["Linux", "OSS-Fuzz"]
OSV has direct commit SHA in ranges[].events[].fixed — use this directly.
Cross-dedup against CVE collector output to avoid duplicates.
```

---

## Collector 6: Repo Miner (Highest Volume)

**File**: `training/scripts/collection/repo_miner_enhanced.py`

```
15 TARGET REPOS (process in this order — fastest first):
1. openssl/openssl          (~400 security commits)
2. curl/curl                (~300)
3. nginx/nginx              (~200)
4. FFmpeg/FFmpeg            (~500)
5. php/php-src              (~300)
6. sqlite/sqlite            (~150)
7. libpng/libpng            (~80)
8. madler/zlib              (~60)
9. git/git                  (~150)
10. redis/redis             (~120)
11. libuv/libuv             (~80)
12. libevent/libevent       (~60)
13. openldap/openldap       (~100)
14. torvalds/linux          (~3000, SLOW — do last)
15. antirez/redis           (~50)

ABSOLUTE RULE: NEVER git clone ANY of these repos.
Use GitHub Commits API: GET /repos/{owner}/{repo}/commits?per_page=100
Checkpoint per repo: status = pending | in_progress | complete | error
```

---

## Expected Data Volumes

| Collector | Expected Samples | Days to Collect | Notes |
|-----------|-----------------|-----------------|-------|
| SARD | 53,666 (collected) | 0.5 (local) | Start here |
| ExploitDB | 800–1,200 | 0.5 (local) | Start same day |
| CVE Phase 1 | 18,000–25,000 index entries | 4–6 hours | Index only |
| CVE Phase 2 | 3,000–5,000 pairs | 4–6 days | API limited |
| GitHub Advisory | 2,000–3,500 pairs | 4–6 days | GraphQL |
| OSV | 1,000–1,800 pairs | 3–4 days | Fastest of the API ones |
| Repo Miner | 18,000–28,000 pairs | 7–10 days | Highest volume |
| **Total** | **33,600–52,300** | | Run all in parallel |

---

*docs/DATA_PIPELINE.md | StreamGuard v1.0 | March 2026*
