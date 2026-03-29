#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage3_cfa.py

Stage 3: CFA (CounterFactual Augmentation) Generation for StreamGuard pipeline.

Input:  training/data/processed/deduped/samples.jsonl
Output: training/data/processed/with_cfa/samples.jsonl

Architecture: 5-tier CFA generation dispatched by CWE difficulty.
  Tier 1: Rule-based AST mutation  (CWE-134, CWE-120, CWE-476)
  Tier 2: Zero-shot LLM            (CWE-121, CWE-122, CWE-125)
  Tier 3: Chain-of-thought LLM     (CWE-89, CWE-78, CWE-190, CWE-79)
  Tier 4: Few-shot exemplar + CoT  (CWE-416, CWE-119)
  Tier 5: Hybrid validation-guided regeneration (fallback for any tier)

This file implements the FOUNDATION layer:
  - CWE_TIER_MAP, TIER_GENERATORS registry, FIX_SIGNATURES
  - validate_cfa_v2() with 7-gate validation
  - CfaQualityTracker for per-CWE metrics
  - run_stage3_tiered() main orchestration loop

Tier generator functions are stubs (None) until each tier story is built.

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2, §3.2, §5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from loguru import logger

try:
    from tree_sitter import Language, Parser
    import tree_sitter_c as tsc

    _C_LANGUAGE = Language(tsc.language())
    _TS_PARSER = Parser(_C_LANGUAGE)
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False
    _TS_PARSER = None

# ── STEP 1: CWE_TIER_MAP ────────────────────────────────────────
CWE_TIER_MAP: dict[str, int] = {
    "CWE-134": 1, "CWE-120": 1, "CWE-476": 1,       # Tier 1: AST
    "CWE-121": 2, "CWE-122": 2, "CWE-125": 2,       # Tier 2: Zero-shot
    "CWE-89":  3, "CWE-78":  3, "CWE-190": 3,       # Tier 3: CoT
    "CWE-79":  3,                                      # Tier 3: CoT
    "CWE-416": 4,                                        # Tier 4: Few-shot
    "CWE-119": 3,                                        # Tier 3: CoT (no exemplars for Tier 4)
}

# ── STEP 2: TIER_GENERATORS registry ────────────────────────────
# Each tier story will replace None with the actual generator function.
# Signature: (code: str, cwe: str, config: CfaConfig) -> list[str]
from training.scripts.preprocessing.cfa_tier1 import tier1_generate
from training.scripts.preprocessing.cfa_tier2 import tier2_generate
from training.scripts.preprocessing.cfa_tier3 import tier3_generate
from training.scripts.preprocessing.cfa_tier4 import tier4_generate
from training.scripts.preprocessing.cfa_tier5 import _tier5_registry_wrapper

TIER_GENERATORS: dict[int, Callable | None] = {
    1: tier1_generate,
    2: tier2_generate,
    3: tier3_generate,
    4: tier4_generate,
    5: _tier5_registry_wrapper,
}

# ── STEP 3: FIX_SIGNATURES ──────────────────────────────────────
# Gate 5: expected fix patterns per CWE. At least one must be present.
FIX_SIGNATURES: dict[str, list[str]] = {
    "CWE-134": [r'"%%s"', r'"%%d"'],
    "CWE-120": [r'\bstrncpy\s*\(', r'\bstrncat\s*\(', r'\bsnprintf\s*\(', r'\bfgets\s*\('],
    "CWE-89":  [r'sqlite3_prepare', r'mysql_stmt', r'mysql_real_escape'],
    "CWE-78":  [r'\bexecve\s*\(', r'whitelist', r'\bstrncmp\s*\('],
    "CWE-190": [r'INT_MAX\s*-', r'INT_MAX\s*/', r'__builtin_.*overflow'],
    "CWE-416": [r'NULL\s*;', r'==\s*NULL', r'!=\s*NULL'],
    "CWE-476": [r'==\s*NULL\s*\)', r'!=\s*NULL'],
    "CWE-121": [r'\bsizeof\s*\(', r'strncpy', r'snprintf'],
    "CWE-122": [r'\bsizeof\s*\(', r'NULL\s*\)', r'strncpy'],
    "CWE-125": [r'>=\s*0', r'<\s*\w+_\w*(size|len|count)', r'!=\s*NULL'],
}

# ── Vulnerability patterns (Gate 4) ─────────────────────────────
# If ANY of these still appear in CFA code, the vuln was NOT removed.
VULN_PATTERNS: dict[str, list[str]] = {
    "CWE-134": [r'\b(printf|fprintf|sprintf)\s*\(\s*\w+\s*\)'],
    "CWE-120": [r'\b(strcpy|strcat|gets|sprintf)\s*\('],
    "CWE-476": [],  # NULL-deref fix adds guards, not removes derefs — regex too coarse
    "CWE-121": [r'\bstrcpy\s*\(', r'\bgets\s*\('],
    "CWE-122": [r'\bstrcpy\s*\(', r'\bgets\s*\('],
    "CWE-125": [],  # too context-dependent for regex
    "CWE-89":  [r'sprintf\s*\([^)]*query.*%s'],
    "CWE-78":  [r'\b(system|popen)\s*\([^)]*\buser\w*'],
    "CWE-190": [],  # overflow patterns are arithmetic, hard to regex
    "CWE-79":  [r'sprintf\s*\([^)]*%s'],
    "CWE-416": [],  # UAF needs control flow, not regex
    "CWE-119": [],
}

# ── Taint analysis patterns (Gate 6) ────────────────────────────
_TAINT_SOURCES = r'(scanf|gets|fgets|getenv|argv|recv|read)\s*\('
_TAINT_SINKS: dict[str, str] = {
    "CWE-89":  r'(sprintf|mysql_query|sqlite3_exec|strcat)\s*\([^)]*query',
    "CWE-78":  r'(system|popen|execl|execv)\s*\(',
    "CWE-134": r'(printf|fprintf)\s*\(\s*\w+\s*\)',
    "CWE-79":  r'(printf|fprintf|sprintf)\s*\([^)]*%s',
}

# Gate 6 applies only to these injection CWEs
_INJECTION_CWES = {"CWE-89", "CWE-78", "CWE-134", "CWE-79"}

# SARD sources: skip CFA generation (they already have native pairs)
_SARD_SOURCES = {"sard", "sard_cfa"}


# ── Configuration ────────────────────────────────────────────────
@dataclass
class CfaConfig:
    """Configuration for Stage 3 CFA generation."""
    enable_cpg_diff: bool = False           # Gate 7 toggle
    checkpoint_interval: int = 200          # samples between checkpoints
    compiler: str = "/c/mingw64/bin/gcc.exe"  # C compiler for Gate 3
    use_tree_sitter_fallback: bool = True   # fallback if compiler unavailable
    similarity_upper: float = 0.99          # Gate 2 upper bound (all tiers)
    similarity_lower_tier1: float = 0.70    # Gate 2 lower bound for Tier 1
    similarity_lower_tier1_cwe476: float = 0.45  # CWE-476 inserts multi-line guards → more diff
    similarity_lower_default: float = 0.20  # Gate 2 lower bound for Tier 2-5 (real-world snippets are short, LLM rewrites differ more)
    # ── LLM (Tier 2-4) settings ──────────────────────────────────
    max_attempts_tier2: int = 6             # max LLM calls per sample (Tier 2)
    max_attempts_tier3: int = 7             # max LLM calls per sample (Tier 3)
    max_attempts_tier4: int = 10            # max LLM calls per sample (Tier 4)
    target_ratio: int = 1                   # target CFAs per vulnerable sample
    openrouter_api_key: str | None = None   # overrides OPENROUTER_API_KEY env var
    llm_timeout: float = 60.0              # seconds per API call


# ── Gate helpers ─────────────────────────────────────────────────
def _compiles_gcc(code: str, compiler: str = "gcc") -> bool:
    """Check if C code compiles (syntax-only) using an external compiler."""
    # Wrap in dummy main if no main present, add common headers
    wrapped = (
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "#include <limits.h>\n"
        + code
    )
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".c", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapped)
            f.flush()
            r = subprocess.run(
                [compiler, "-fsyntax-only", "-x", "c", f.name],
                capture_output=True,
                timeout=10,
            )
            return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    finally:
        try:
            os.unlink(f.name)
        except OSError:
            pass


def _compiles_tree_sitter(code: str) -> bool:
    """Fallback compile check: parse with tree-sitter, reject if ERROR nodes."""
    if not _HAS_TREE_SITTER or _TS_PARSER is None:
        return True  # can't check, assume OK
    tree = _TS_PARSER.parse(code.encode("utf-8", errors="replace"))
    return not tree.root_node.has_error


def compiles(code: str, config: CfaConfig | None = None) -> bool:
    """Check if code compiles. Uses tree-sitter (syntax-only) as primary check.

    Real-world CFA samples are code fragments (missing headers, kernel types,
    project macros) so gcc -fsyntax-only rejects them even when syntactically
    valid. Tree-sitter checks parse structure which is what we actually need.
    """
    if _HAS_TREE_SITTER:
        return _compiles_tree_sitter(code)
    # No tree-sitter available — try gcc as last resort
    cfg = config or CfaConfig()
    return _compiles_gcc(code, cfg.compiler)


def _has_taint_path(code: str, cwe: str) -> bool:
    """Lightweight regex-based taint check for injection CWEs."""
    sink_pattern = _TAINT_SINKS.get(cwe, "")
    if not sink_pattern:
        return False
    has_source = bool(re.search(_TAINT_SOURCES, code))
    has_sink = bool(re.search(sink_pattern, code))
    return has_source and has_sink


def extract_c_code(raw: str) -> str:
    """Extract C code from LLM output (strip markdown fences, explanation)."""
    # Try to extract from ```c ... ``` or ``` ... ``` blocks
    m = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If no fences, return as-is (already code)
    return raw.strip()


# ── STEP 4: validate_cfa_v2 ─────────────────────────────────────
def validate_cfa_v2(
    original: str,
    cfa_raw: str,
    cwe: str,
    tier: int,
    config: CfaConfig | None = None,
) -> tuple[bool, str, str, float]:
    """
    7-gate CFA validation.

    Returns: (is_valid, rejection_reason, cleaned_code, quality_score)
      - quality_score: 0.0-1.0 (1.0 = perfect, 0.6 = Gate 5 soft miss)
      - rejection_reason: '' if valid, else gate name
      - cleaned_code: extracted C code if valid, '' if rejected
    """
    cfg = config or CfaConfig()
    cfa_code = extract_c_code(cfa_raw)
    quality_score = 1.0

    # Gate 1: Identity — CFA must differ from original
    if cfa_code.strip() == original.strip():
        return False, "identical_to_original", "", 0.0

    # Gate 2: Similarity bounds — tier-aware
    if tier == 1 and cwe == "CWE-476":
        lower = cfg.similarity_lower_tier1_cwe476
    elif tier == 1:
        lower = cfg.similarity_lower_tier1
    else:
        lower = cfg.similarity_lower_default
    upper = cfg.similarity_upper
    sim = SequenceMatcher(None, original.split(), cfa_code.split()).ratio()
    if sim > upper:
        return False, "too_similar", "", 0.0
    if sim < lower:
        return False, "too_different", "", 0.0

    # Gate 3: Compilation — must parse without errors
    if not compiles(cfa_code, cfg):
        return False, "compile_fail", "", 0.0

    # Gate 4: Vulnerability pattern removed
    for pat in VULN_PATTERNS.get(cwe, []):
        if re.search(pat, cfa_code):
            return False, "vuln_pattern_remains", "", 0.0

    # Gate 5: Fix signature present (SOFT gate — miss = reduced score, NOT reject)
    sigs = FIX_SIGNATURES.get(cwe, [])
    if sigs:
        has_fix = any(re.search(p, cfa_code) for p in sigs)
        if not has_fix:
            quality_score = 0.6  # soft penalty, do NOT reject

    # Gate 6: Taint path broken (injection CWEs only)
    if cwe in _INJECTION_CWES:
        original_has_path = _has_taint_path(original, cwe)
        if original_has_path:
            cfa_has_path = _has_taint_path(cfa_code, cwe)
            if cfa_has_path:
                return False, "taint_path_intact", "", 0.0

    # Gate 7: CPG diff budget (optional, default disabled)
    if cfg.enable_cpg_diff:
        # Placeholder — will be implemented when Joern integration is ready
        pass

    return True, "", cfa_code, quality_score


# ── STEP 5: CfaQualityTracker ───────────────────────────────────
class CfaQualityTracker:
    """Tracks per-CWE CFA generation quality metrics."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        for cwe in CWE_TIER_MAP:
            self._data[cwe] = {
                "total_attempts": 0,
                "accepted": 0,
                "quality_scores": [],
                "rejection_reasons": defaultdict(int),
                "tiers_used": defaultdict(int),
            }

    def record_attempt(
        self,
        cwe: str,
        tier: int,
        rejection_reason: str | None,
        quality_score: float,
    ) -> None:
        """Record one CFA attempt result."""
        if cwe not in self._data:
            self._data[cwe] = {
                "total_attempts": 0,
                "accepted": 0,
                "quality_scores": [],
                "rejection_reasons": defaultdict(int),
                "tiers_used": defaultdict(int),
            }
        entry = self._data[cwe]
        entry["total_attempts"] += 1
        entry["tiers_used"][tier] += 1
        if rejection_reason:
            entry["rejection_reasons"][rejection_reason] += 1
        else:
            entry["accepted"] += 1
            entry["quality_scores"].append(quality_score)

    def get_report(self) -> dict[str, Any]:
        """Generate per-CWE quality report dict."""
        report: dict[str, Any] = {}
        for cwe, d in self._data.items():
            total = d["total_attempts"]
            accepted = d["accepted"]
            scores = d["quality_scores"]

            compile_fails = d["rejection_reasons"].get("compile_fail", 0)
            non_identical = total - d["rejection_reasons"].get(
                "identical_to_original", 0
            )
            compile_passed = non_identical - compile_fails if non_identical > 0 else 0

            pattern_fails = d["rejection_reasons"].get("vuln_pattern_remains", 0)
            fix_sig_misses = d["rejection_reasons"].get("no_fix_signature", 0)

            report[cwe] = {
                "total_attempts": total,
                "accepted": accepted,
                "compile_rate": (
                    (compile_passed / non_identical) if non_identical > 0 else 0.0
                ),
                "pattern_removal_rate": (
                    ((compile_passed - pattern_fails) / compile_passed)
                    if compile_passed > 0
                    else 0.0
                ),
                "fix_signature_rate": (
                    ((compile_passed - fix_sig_misses) / compile_passed)
                    if compile_passed > 0
                    else 0.0
                ),
                "avg_similarity": (
                    (sum(scores) / len(scores)) if scores else 0.0
                ),
                "rejection_breakdown": dict(d["rejection_reasons"]),
                "tiers_used": dict(d["tiers_used"]),
            }
        return report

    def save(self, path: str) -> None:
        """Write cfa_quality_report.json to disk."""
        report = self.get_report()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(out.parent), suffix=".tmp", prefix="cfa_report_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            os.replace(tmp_path, str(out))
            logger.info(f"CFA quality report saved to {path}")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# ── Per-CWE quality thresholds (Section 4.3 of research doc) ────
# compile_rate below these values triggers a logged warning.
CWE_QUALITY_THRESHOLDS: dict[str, float] = {
    "CWE-134": 0.85,
    "CWE-120": 0.80,
    "CWE-476": 0.80,
    "CWE-121": 0.75,
    "CWE-122": 0.75,
    "CWE-125": 0.75,
    "CWE-89":  0.65,
    "CWE-78":  0.65,
    "CWE-190": 0.70,
    "CWE-79":  0.60,
    "CWE-416": 0.60,
    "CWE-119": 0.65,
}


# ── CFA generation dispatch (with escalation) ────────────────────
def generate_cfa_for_sample(
    sample: dict,
    config: CfaConfig | None = None,
    exemplar_db: Any = None,
) -> tuple[list[str], int]:
    """
    Dispatch CFA generation to the appropriate tier generator,
    with automatic Tier 1 → Tier 2 escalation.

    Returns (list_of_cfa_strings, actual_tier_used).
    Returns ([], tier) if generator is unavailable or returns nothing.
    """
    cwe = sample.get("cwe", "")
    tier = CWE_TIER_MAP.get(cwe, 3)
    code = sample.get("code", "")
    cfg = config or CfaConfig()

    generator = TIER_GENERATORS.get(tier)
    if generator is None:
        return [], tier

    candidates = generator(code, cwe, cfg, exemplar_db) \
        if tier >= 2 else generator(code, cwe, cfg)

    # Tier 1 escalation: if no candidates produced, try Tier 2
    # Only escalate if Tier 2 has a prompt for this CWE (avoids dead-end for CWE-134/476)
    if not candidates and tier == 1:
        from training.scripts.preprocessing.cfa_tier2 import TIER2_PROMPTS
        tier2_gen = TIER_GENERATORS.get(2)
        if tier2_gen is not None and cwe in TIER2_PROMPTS:
            logger.debug(
                f"Tier 1 returned [] for {cwe} — escalating to Tier 2"
            )
            candidates = tier2_gen(code, cwe, cfg, exemplar_db)
            if candidates:
                return candidates, 2
        elif not candidates:
            logger.debug(f"Tier 1 returned [] for {cwe} — no Tier 2 prompt, skipping escalation")

    return candidates, tier


# ── STEP 6: run_stage3_tiered ────────────────────────────────────
def run_stage3_tiered(
    input_path: str,
    output_path: str,
    config: CfaConfig | None = None,
    checkpoint_dir: str | None = None,
    exemplar_db: Any = None,
    dry_run: bool = False,
    max_samples: int | None = None,
    cwe_filter: str | None = None,
) -> dict:
    """
    Main Stage 3 orchestration loop — fully integrated with all 5 tiers.

    For each sample:
      1. Always write original to output
      2. Skip if label != 1
      3. Skip if source is SARD or cfa_type == 'native'
      4. Skip if no cwe field (or filtered by --cwe-filter)
      5. Dispatch to generate_cfa_for_sample() with escalation
      6. Write valid CFAs with pair_id, cfa_tier, source=<orig>_cfa
      7. Log failures to cfa_failures.jsonl (never crash)
      8. Checkpoint every cfg.checkpoint_interval samples
      9. Progress log every 100 samples
      10. Quality threshold check every 500 samples

    Returns stats dict.
    """
    from training.scripts.collection.schema import make_sample_id, make_timestamp

    cfg = config or CfaConfig()
    tracker = CfaQualityTracker()

    logger.info(f"Stage 3: Loading samples from {input_path}")
    inp = Path(input_path)
    if not inp.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    samples: list[dict] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if max_samples is not None:
        samples = samples[:max_samples]

    n_input = len(samples)
    logger.info(f"Input: {n_input} samples")

    # Checkpoint resume
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    processed_ids: set[str] = set()
    output_buffer: list[dict] = []

    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "stage3_checkpoint.jsonl"
        if ckpt_file.exists():
            with open(ckpt_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            output_buffer.append(rec)
                            processed_ids.add(rec["id"])
                        except json.JSONDecodeError:
                            continue
            logger.info(
                f"Resumed from checkpoint: {len(processed_ids)} already processed"
            )

    # Failures log path
    out_parent = Path(output_path).parent
    failures_path = out_parent / "cfa_failures.jsonl"

    # Stats
    n_sard_skipped = 0
    n_cfa_generated = 0
    n_cfa_rejected = 0
    n_originals_written = 0
    n_processed_this_run = 0
    n_failures_logged = 0

    def _maybe_checkpoint() -> None:
        if (
            ckpt_dir
            and n_processed_this_run > 0
            and n_processed_this_run % cfg.checkpoint_interval == 0
        ):
            _write_checkpoint(ckpt_dir, output_buffer)

    def _log_failure(sample: dict, reason: str, tier: int) -> None:
        nonlocal n_failures_logged
        rec = {
            "id": sample.get("id", ""),
            "cwe": sample.get("cwe", ""),
            "source": sample.get("source", ""),
            "tier": tier,
            "reason": reason,
        }
        try:
            out_parent.mkdir(parents=True, exist_ok=True)
            with open(failures_path, "a", encoding="utf-8") as ff:
                ff.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        n_failures_logged += 1

    def _check_thresholds() -> None:
        report = tracker.get_report()
        for cwe, data in report.items():
            threshold = CWE_QUALITY_THRESHOLDS.get(cwe, 0.0)
            rate = data.get("compile_rate", 1.0)
            if data["total_attempts"] > 0 and rate < threshold:
                logger.warning(
                    f"CWE-{cwe} compile_rate {rate*100:.1f}% "
                    f"below threshold {threshold*100:.0f}%"
                )

    for i, sample in enumerate(samples):
        sid = sample.get("id", "")
        if sid in processed_ids:
            continue

        source = sample.get("source", "")
        cwe = sample.get("cwe", "")
        label = sample.get("label", -1)
        cfa_type = sample.get("cfa_type", "")

        # cwe-filter
        if cwe_filter and cwe != cwe_filter:
            output_buffer.append(sample)
            n_originals_written += 1
            processed_ids.add(sid)
            n_processed_this_run += 1
            _maybe_checkpoint()
            continue

        # Always write the original
        output_buffer.append(sample)
        n_originals_written += 1

        # SARD / native pairs: skip generation
        if source in _SARD_SOURCES or cfa_type == "native":
            n_sard_skipped += 1
            processed_ids.add(sid)
            n_processed_this_run += 1
            _maybe_checkpoint()
            continue

        # Only generate for vulnerable samples with a CWE
        if label != 1 or not cwe:
            processed_ids.add(sid)
            n_processed_this_run += 1
            _maybe_checkpoint()
            continue

        if dry_run:
            processed_ids.add(sid)
            n_processed_this_run += 1
            _maybe_checkpoint()
            continue

        # Skip samples whose original code doesn't parse — CFA can never
        # pass Gate 3 if the original is a broken fragment (33% of real-world data)
        if not compiles(sample.get("code", ""), cfg):
            _log_failure(sample, "original_unparseable", CWE_TIER_MAP.get(cwe, 3))
            processed_ids.add(sid)
            n_processed_this_run += 1
            _maybe_checkpoint()
            continue

        tier = CWE_TIER_MAP.get(cwe, 3)

        try:
            candidates, actual_tier = generate_cfa_for_sample(
                sample, cfg, exemplar_db
            )
        except Exception as exc:
            logger.warning(f"generate_cfa_for_sample error [{cwe}]: {exc}")
            _log_failure(sample, f"exception: {exc}", tier)
            processed_ids.add(sid)
            n_processed_this_run += 1
            _maybe_checkpoint()
            continue

        cfa_count_this_sample = 0

        for cfa_raw in candidates:
            is_valid, reason, cleaned, qscore = validate_cfa_v2(
                sample["code"], cfa_raw, cwe, actual_tier, cfg
            )
            tracker.record_attempt(
                cwe, actual_tier, reason if not is_valid else None, qscore
            )

            if is_valid:
                cfa_sample = {
                    **sample,
                    "id": make_sample_id(),
                    "code": cleaned,
                    "label": 0,
                    "source": source + "_cfa",
                    "cfa_origin": "llm_generated" if actual_tier >= 2 else "ast_rule",
                    "pair_id": sid,
                    "cfa_tier": actual_tier,
                    "cfa_quality_score": qscore,
                    "collected_at": make_timestamp(),
                }
                output_buffer.append(cfa_sample)
                n_cfa_generated += 1
                cfa_count_this_sample += 1
            else:
                n_cfa_rejected += 1

        if cfa_count_this_sample == 0:
            _log_failure(sample, "no_valid_cfa", actual_tier)

        processed_ids.add(sid)
        n_processed_this_run += 1
        _maybe_checkpoint()

        # Progress log every 100 samples processed this run
        if n_processed_this_run % 100 == 0:
            total_done = len(processed_ids)
            logger.info(
                f"[Stage3] {total_done}/{n_input} | "
                f"CWE {cwe} | tier {actual_tier} | "
                f"CFA total: {n_cfa_generated}"
            )

        # Quality threshold check every 500 samples
        if n_processed_this_run % 500 == 0:
            _check_thresholds()

    # Final quality check
    _check_thresholds()

    # Write final output (atomic)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(out.parent), suffix=".tmp", prefix="stage3_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for rec in output_buffer:
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp_path, str(out))
        logger.info(f"Output written to {output_path}")
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # Save quality report
    report_path = str(out.parent / "cfa_quality_report.json")
    tracker.save(report_path)

    # Print per-CWE summary table
    report = tracker.get_report()
    sep = "-" * 52
    print(f"\n{sep}")
    print("CFA Quality Report")
    print(sep)
    print(f"{'CWE':<10} {'Attempts':>8} {'Accepted':>8} {'Compile%':>9} {'AvgSim':>7}")
    print(sep)
    for cwe_key in sorted(report):
        d = report[cwe_key]
        if d["total_attempts"] == 0:
            continue
        print(
            f"{cwe_key:<10} {d['total_attempts']:>8} {d['accepted']:>8} "
            f"{d['compile_rate']*100:>8.1f}% {d['avg_similarity']:>7.3f}"
        )
    print(sep)

    stats = {
        "input": n_input,
        "originals_written": n_originals_written,
        "cfa_generated": n_cfa_generated,
        "cfa_rejected": n_cfa_rejected,
        "sard_skipped": n_sard_skipped,
        "failures_logged": n_failures_logged,
        "output_total": len(output_buffer),
    }
    logger.info(f"Stage 3 complete: {stats}")
    return stats


def _write_checkpoint(ckpt_dir: Path, records: list[dict]) -> None:
    """Write checkpoint file (atomic)."""
    ckpt_file = ckpt_dir / "stage3_checkpoint.jsonl"
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(ckpt_dir), suffix=".tmp", prefix="ckpt_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp_path, str(ckpt_file))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: CFA Generation")
    parser.add_argument(
        "--input",
        default="training/data/processed/deduped/samples.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="training/data/processed/with_cfa/samples.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--exemplar-db",
        default=None,
        help="Path to exemplar_db.json (required for CWE-416/119)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--enable-cpg-diff",
        action="store_true",
        help="Enable Gate 7 (CPG diff budget check)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write originals only, skip CFA generation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most N samples (for testing)",
    )
    parser.add_argument(
        "--cwe-filter",
        default=None,
        help="Process only samples with this CWE (e.g. CWE-89)",
    )
    args = parser.parse_args()

    exemplar_db = None
    if args.exemplar_db:
        from training.scripts.preprocessing.cfa_exemplar_db import ExemplarDatabase
        exemplar_db = ExemplarDatabase.load(args.exemplar_db)

    cfg = CfaConfig(enable_cpg_diff=args.enable_cpg_diff)
    run_stage3_tiered(
        input_path=args.input,
        output_path=args.output,
        config=cfg,
        checkpoint_dir=args.checkpoint_dir,
        exemplar_db=exemplar_db,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        cwe_filter=args.cwe_filter,
    )


if __name__ == "__main__":
    main()
