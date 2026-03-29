#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage1_clean.py

Stage 1: Clean & Normalize C functions for StreamGuard pipeline.

Input:  Raw JSONL from collectors (e.g. training/scripts/collection/data/raw/sard/)
Output: training/data/processed/cleaned/samples.jsonl

Operations (in order):
  1. chardet encoding fix -> UTF-8
  2. Source routing:
     - SARD: validate function_definition via tree-sitter (already single function)
     - Non-SARD: extract_functions() to split full files into individual functions
  3. Strip #ifdef/#endif preprocessor blocks (SARD/Juliet)
  4. Remove // and /* */ comments
  5. Normalize blank lines (max 2 consecutive)
  6. Filter: 5 <= lines <= 500, 10 <= tokens <= 4096

Source: docs/New Docs/PREPROCESSING.md §Stage 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import chardet
from loguru import logger
from tree_sitter import Language, Parser

try:
    import tree_sitter_c as tsc
except ImportError:
    logger.error("tree_sitter_c not installed. Run: pip install tree-sitter-c")
    sys.exit(1)

# ── Tree-sitter setup ──────────────────────────────────────────────
C_LANGUAGE = Language(tsc.language())
parser = Parser(C_LANGUAGE)

# ── Filter thresholds ──────────────────────────────────────────────
MIN_LINES = 5
MAX_LINES = 500
MIN_TOKENS = 10
MAX_TOKENS = 4096

# ── SARD source identifiers ────────────────────────────────────────
# Both "sard" and "sard_cfa" samples already have a single extracted
# function in the code field — do NOT re-run extract_functions() on them.
SARD_SOURCES = {"sard", "sard_cfa"}


# ── Function extraction (non-SARD sources) ─────────────────────────
def extract_functions(source_code: str, filepath: str = "") -> list[dict]:
    """
    Extract all function definitions from a full C file using tree-sitter.

    Used for non-SARD sources (CVE, ExploitDB, etc.) where the code field
    contains a full file. For SARD, code is already a single function.

    Uses iterative DFS to avoid RecursionError on deeply nested C files.
    """
    tree = parser.parse(source_code.encode("utf-8", errors="replace"))
    fns = []

    stack = list(reversed(tree.root_node.children))
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            code = source_code[node.start_byte:node.end_byte]
            decl = node.child_by_field_name("declarator")
            name = decl.text.decode("utf-8", errors="replace") if decl else "unknown"
            fns.append({
                "code": code,
                "name": name,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0],
                "filepath": filepath,
            })
            # don't recurse into nested functions
            continue
        # Push children in reverse order so leftmost is processed first
        stack.extend(reversed(node.children))

    return fns


# ── Preprocessor guard stripping ───────────────────────────────────
def strip_preprocessor_guards(code: str) -> str:
    """
    Strip Juliet #ifdef/#ifndef/#elif/#else/#endif lines.
    Juliet uses these for flow variants; without stripping, Joern
    produces incomplete CPGs in Stage 4.
    """
    return re.sub(r'#\s*(ifdef|ifndef|elif|else|endif)[^\n]*\n?', '', code)


# ── Normalization ──────────────────────────────────────────────────
def normalize(code: str, source: str = "") -> str | None:
    """
    Normalize a C function string. Returns None if it fails filters.

    Order:
      1. Strip #ifdef/#endif (SARD only)
      2. Remove comments
      3. Normalize blank lines (max 2 consecutive)
      4. Line count filter
      5. Token count filter
    """
    # 1. Strip preprocessor guards for SARD/Juliet (sard and sard_cfa)
    if source in SARD_SOURCES:
        code = strip_preprocessor_guards(code)

    # 2. Remove comments
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # 3. Normalize blank lines (max 2 consecutive)
    code = re.sub(r'\n{3,}', '\n\n', code)
    code = code.strip()

    # 4. Line count filter
    lines = code.splitlines()
    if not (MIN_LINES <= len(lines) <= MAX_LINES):
        return None

    # 5. Token count filter
    tokens = code.split()
    if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
        return None

    return code


# ── Encoding fix ───────────────────────────────────────────────────
def fix_encoding(raw_code: str | bytes) -> str:
    """Detect encoding and re-encode as UTF-8."""
    if isinstance(raw_code, bytes):
        det = chardet.detect(raw_code)
        encoding = det.get("encoding") or "utf-8"
        return raw_code.decode(encoding, errors="replace")
    return raw_code


# ── Validate function_definition (SARD path) ──────────────────────
def has_function_definition(code: str) -> bool:
    """Check that code parses as containing a function_definition node."""
    tree = parser.parse(code.encode("utf-8", errors="replace"))
    return any(n.type == "function_definition" for n in tree.root_node.children)


# ── Process a single sample ───────────────────────────────────────
def process_sample(sample: dict) -> list[dict]:
    """
    Process one canonical sample.
    Returns a list of cleaned samples (usually 1, but non-SARD sources
    may yield multiple functions from a single file).
    """
    raw_code = sample.get("code", "")
    if not raw_code:
        return []

    source = sample.get("source", "")

    # Fix encoding
    raw_code = fix_encoding(raw_code)

    results = []

    if source in SARD_SOURCES:
        # SARD/SARD_CFA: strip preprocessor guards BEFORE validation, since
        # #ifndef OMITBAD / #endif wrappers prevent tree-sitter parsing
        raw_code = strip_preprocessor_guards(raw_code)
        # Validate it IS a function, then normalize directly
        if not has_function_definition(raw_code):
            return []  # malformed — discard
        normalized = normalize(raw_code, source=source)
        if normalized is None:
            return []
        results.append({**sample, "code": normalized})
    else:
        # Non-SARD: code may be a full file — extract functions
        functions = extract_functions(raw_code, filepath=sample.get("file_path", ""))
        if not functions:
            # Fallback: try as single function
            if has_function_definition(raw_code):
                normalized = normalize(raw_code, source=source)
                if normalized is not None:
                    results.append({**sample, "code": normalized})
            return results

        for fn in functions:
            normalized = normalize(fn["code"], source=source)
            if normalized is None:
                continue
            fn_sample = {
                **sample,
                "code": normalized,
                "metadata": {
                    **sample.get("metadata", {}),
                    "function_name": fn.get("name", ""),
                    "start_line": fn.get("start_line", 0),
                    "end_line": fn.get("end_line", 0),
                },
            }
            results.append(fn_sample)

    return results


# ── Main pipeline ──────────────────────────────────────────────────
def run_stage1(
    input_dirs: list[str],
    output_path: str,
    dry_run: bool = False,
    max_samples: int | None = None,
    source_filter: str | None = None,
) -> dict:
    """
    Run Stage 1 cleaning on all JSONL files in input_dirs.

    Args:
        input_dirs: Directories (or files) containing .jsonl files.
        output_path: Destination JSONL path. Written atomically via .tmp.
        dry_run: If True, process but do not write output.
        max_samples: Stop after this many successfully processed output samples.
        source_filter: If set, skip samples whose source != source_filter.

    Returns:
        Stats dict with keys total_read, processed, skipped.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_read = 0
    processed = 0
    skipped = 0

    # Atomic write: write to .tmp, then os.replace() on success
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    out_f = None
    if not dry_run:
        out_f = open(tmp_path, "w", encoding="utf-8")

    _success = False
    try:
        for dir_path in input_dirs:
            p = Path(dir_path)
            if not p.exists():
                logger.warning(f"Input path does not exist: {dir_path}")
                continue

            # Accept both a single .jsonl file or a directory
            if p.is_file() and p.suffix == ".jsonl":
                jsonl_files = [p]
            else:
                jsonl_files = sorted(p.rglob("*.jsonl"))

            if not jsonl_files:
                logger.warning(f"No .jsonl files found in {dir_path}")
                continue

            for jsonl_file in jsonl_files:
                logger.info(f"Reading {jsonl_file}")
                with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        if max_samples is not None and processed >= max_samples:
                            break

                        total_read += 1

                        try:
                            sample = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Malformed JSON at line {total_read}")
                            skipped += 1
                            continue

                        # Source filter (skip samples from other sources)
                        if source_filter and sample.get("source") != source_filter:
                            skipped += 1
                            continue

                        cleaned = process_sample(sample)
                        if not cleaned:
                            skipped += 1
                            continue

                        for c in cleaned:
                            processed += 1
                            if out_f is not None:
                                out_f.write(json.dumps(c) + "\n")

                if max_samples is not None and processed >= max_samples:
                    break
            if max_samples is not None and processed >= max_samples:
                break
        _success = True
    finally:
        if out_f is not None:
            out_f.close()
            if _success:
                # Atomic promotion only on clean completion
                os.replace(tmp_path, output)
            else:
                # Clean up partial .tmp on error
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    stats = {
        "total_read": total_read,
        "processed": processed,
        "skipped": skipped,
    }

    mode = "[DRY RUN] " if dry_run else ""
    logger.info(
        f"{mode}Stage 1 complete: "
        f"{total_read} read, {processed} processed, {skipped} skipped"
    )
    return stats


# ── CLI ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Stage 1: Clean & Normalize C functions"
    )
    ap.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input directories containing .jsonl files",
    )
    ap.add_argument(
        "--output",
        default="training/data/processed/cleaned/samples.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to read (for testing)",
    )
    ap.add_argument(
        "--source-filter",
        default=None,
        help="Only process samples from this source (e.g. sard, cve)",
    )

    args = ap.parse_args()
    run_stage1(args.input, args.output, args.dry_run, args.max_samples, args.source_filter)


if __name__ == "__main__":
    main()
