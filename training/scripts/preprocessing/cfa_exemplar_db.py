#!/usr/bin/env python3
"""
training/scripts/preprocessing/cfa_exemplar_db.py

Offline exemplar database builder for Tier 4 CFA generation.

Scans a paired JSONL dataset for real (non-SARD) vulnerable/safe pairs
and builds a compact exemplar database keyed by CWE.  Each exemplar is a
(vulnerable_code, safe_code) pair selected for keyword diversity.

The database is saved as a plain JSON file and loaded at inference time by
cfa_tier4.py.

Usage (CLI):
    python -m training.scripts.preprocessing.cfa_exemplar_db \
        --input  training/data/processed/deduped/samples.jsonl \
        --output training/data/processed/exemplar_db.json \
        --max-per-cwe 8 \
        --min-real-pairs 1

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2.4
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

# CWEs handled by Tier 4 (must have exemplars)
TIER4_CWES = {"CWE-416", "CWE-119"}

# ── Builtin exemplars ────────────────────────────────────────────
# Handcrafted high-quality pairs covering diverse fix patterns so Tier 4
# always has good few-shot examples even when the dataset has few real
# pairs for these CWEs.  These are merged into the DB at load/build time.
# Each exemplar demonstrates a distinct fix strategy so the LLM learns
# multiple valid approaches.

_BUILTIN_EXEMPLARS: dict[str, list[dict[str, str]]] = {
    # ── CWE-416: Use-After-Free ──────────────────────────────────
    "CWE-416": [
        # Pattern A: null-after-free + null guard before use
        {
            "vuln": (
                "void process_data(int *input, int len) {\n"
                "    int *buf = (int *)malloc(len * sizeof(int));\n"
                "    if (!buf) return;\n"
                "    memcpy(buf, input, len * sizeof(int));\n"
                "    free(buf);\n"
                "    int sum = 0;\n"
                "    for (int i = 0; i < len; i++) {\n"
                "        sum += buf[i];\n"
                "    }\n"
                "    (void)sum;\n"
                "}"
            ),
            "safe": (
                "void process_data(int *input, int len) {\n"
                "    int *buf = (int *)malloc(len * sizeof(int));\n"
                "    if (!buf) return;\n"
                "    memcpy(buf, input, len * sizeof(int));\n"
                "    int sum = 0;\n"
                "    for (int i = 0; i < len; i++) {\n"
                "        sum += buf[i];\n"
                "    }\n"
                "    free(buf);\n"
                "    buf = NULL;\n"
                "    (void)sum;\n"
                "}"
            ),
        },
        # Pattern B: remove redundant free (double-free)
        {
            "vuln": (
                "void cleanup_conn(struct conn *c) {\n"
                "    if (c->buffer) {\n"
                "        free(c->buffer);\n"
                "    }\n"
                "    log_disconnect(c->id);\n"
                "    free(c->buffer);\n"
                "    free(c);\n"
                "}"
            ),
            "safe": (
                "void cleanup_conn(struct conn *c) {\n"
                "    if (c->buffer) {\n"
                "        free(c->buffer);\n"
                "        c->buffer = NULL;\n"
                "    }\n"
                "    log_disconnect(c->id);\n"
                "    free(c);\n"
                "}"
            ),
        },
        # Pattern C: null guard on conditional use after free
        {
            "vuln": (
                "void handle_request(char *req) {\n"
                "    char *copy = strdup(req);\n"
                "    if (!copy) return;\n"
                "    parse_header(copy);\n"
                "    free(copy);\n"
                "    if (needs_retry(req)) {\n"
                "        send_response(copy);\n"
                "    }\n"
                "}"
            ),
            "safe": (
                "void handle_request(char *req) {\n"
                "    char *copy = strdup(req);\n"
                "    if (!copy) return;\n"
                "    parse_header(copy);\n"
                "    if (needs_retry(req)) {\n"
                "        send_response(copy);\n"
                "    }\n"
                "    free(copy);\n"
                "    copy = NULL;\n"
                "}"
            ),
        },
        # Pattern D: deferred free — move free() after last use
        {
            "vuln": (
                "int read_value(int *arr, int n) {\n"
                "    int *tmp = (int *)malloc(n * sizeof(int));\n"
                "    if (!tmp) return -1;\n"
                "    memcpy(tmp, arr, n * sizeof(int));\n"
                "    free(tmp);\n"
                "    int result = tmp[0] + tmp[n-1];\n"
                "    return result;\n"
                "}"
            ),
            "safe": (
                "int read_value(int *arr, int n) {\n"
                "    int *tmp = (int *)malloc(n * sizeof(int));\n"
                "    if (!tmp) return -1;\n"
                "    memcpy(tmp, arr, n * sizeof(int));\n"
                "    int result = tmp[0] + tmp[n-1];\n"
                "    free(tmp);\n"
                "    tmp = NULL;\n"
                "    return result;\n"
                "}"
            ),
        },
    ],
    # ── CWE-119: Improper Restriction of Operations within Memory Bounds ─
    "CWE-119": [
        # Pattern A: cap memcpy size to buffer length
        {
            "vuln": (
                "void copy_input(char *src, int src_len) {\n"
                "    char dest[128];\n"
                "    memcpy(dest, src, src_len);\n"
                "    dest[src_len] = '\\0';\n"
                "}"
            ),
            "safe": (
                "void copy_input(char *src, int src_len) {\n"
                "    char dest[128];\n"
                "    int safe_len = src_len;\n"
                "    if (safe_len > (int)sizeof(dest) - 1) {\n"
                "        safe_len = (int)sizeof(dest) - 1;\n"
                "    }\n"
                "    memcpy(dest, src, safe_len);\n"
                "    dest[safe_len] = '\\0';\n"
                "}"
            ),
        },
        # Pattern B: replace strcpy with strncpy + null terminator
        {
            "vuln": (
                "void format_name(const char *first, const char *last) {\n"
                "    char full[64];\n"
                "    strcpy(full, first);\n"
                "    strcat(full, \" \");\n"
                "    strcat(full, last);\n"
                "}"
            ),
            "safe": (
                "void format_name(const char *first, const char *last) {\n"
                "    char full[64];\n"
                "    full[0] = '\\0';\n"
                "    strncat(full, first, sizeof(full) - 1);\n"
                "    strncat(full, \" \", sizeof(full) - strlen(full) - 1);\n"
                "    strncat(full, last, sizeof(full) - strlen(full) - 1);\n"
                "    full[sizeof(full) - 1] = '\\0';\n"
                "}"
            ),
        },
        # Pattern C: array index bounds check
        {
            "vuln": (
                "int get_element(int *arr, int arr_size, int index) {\n"
                "    return arr[index];\n"
                "}"
            ),
            "safe": (
                "int get_element(int *arr, int arr_size, int index) {\n"
                "    if (index < 0 || index >= arr_size) {\n"
                "        return -1;\n"
                "    }\n"
                "    return arr[index];\n"
                "}"
            ),
        },
        # Pattern D: heap buffer with dynamic size validation
        {
            "vuln": (
                "void read_packet(const char *data, int data_len) {\n"
                "    char *buf = (char *)malloc(256);\n"
                "    if (!buf) return;\n"
                "    memcpy(buf, data, data_len);\n"
                "    process(buf);\n"
                "    free(buf);\n"
                "}"
            ),
            "safe": (
                "void read_packet(const char *data, int data_len) {\n"
                "    char *buf = (char *)malloc(256);\n"
                "    if (!buf) return;\n"
                "    int copy_len = data_len;\n"
                "    if (copy_len > 255) {\n"
                "        copy_len = 255;\n"
                "    }\n"
                "    memcpy(buf, data, copy_len);\n"
                "    buf[copy_len] = '\\0';\n"
                "    process(buf);\n"
                "    free(buf);\n"
                "}"
            ),
        },
    ],
}

# Sources eligible as exemplars.
# Real-world sources are preferred, but SARD is also included for Tier-4 CWEs
# because CWE-416/119 real-world collectors only produce singletons (no safe
# counterpart), leaving 0 real pairs. SARD has clean vuln/fix pairs for these
# CWEs and is ideal as few-shot examples even though it is skipped for generation.
_REAL_SOURCES = {"cve", "exploitdb", "github_advisory", "osv", "repo"}
_EXEMPLAR_SOURCES = _REAL_SOURCES | {"sard", "sard_cfa"}


def _keyword_set(code: str) -> set[str]:
    """Extract alphanumeric tokens ≥ 3 chars from C code for overlap scoring."""
    import re
    return {w for w in re.findall(r'[A-Za-z_]\w*', code) if len(w) >= 3}


class ExemplarDatabase:
    """
    Per-CWE exemplar store for Tier 4 few-shot prompting.

    Each CWE maps to a list of {"vuln": str, "safe": str} dicts.
    """

    def __init__(self) -> None:
        self._db: dict[str, list[dict[str, str]]] = defaultdict(list)

    def _merge_builtins(self) -> None:
        """Merge ``_BUILTIN_EXEMPLARS`` into the DB, skipping duplicates."""
        for cwe, pairs in _BUILTIN_EXEMPLARS.items():
            existing_vulns = {ex["vuln"].strip() for ex in self._db.get(cwe, [])}
            for pair in pairs:
                if pair["vuln"].strip() not in existing_vulns:
                    self._db[cwe].append(pair)
                    existing_vulns.add(pair["vuln"].strip())

    # ── Construction ─────────────────────────────────────────────

    def build_from_pairs(
        self,
        jsonl_path: str,
        max_per_cwe: int = 8,
        min_real_pairs: int = 1,
    ) -> None:
        """
        Scan a paired samples.jsonl and populate the database.

        Only real-world (non-SARD) pairs are used as exemplars.
        For each CWE a diversity filter keeps exemplars whose keyword
        overlap with already-selected exemplars is < 0.80 (Jaccard).

        Raises ValueError if any Tier-4 CWE has fewer than
        ``min_real_pairs`` exemplars after scanning.
        """
        # group pair_id → list[sample]
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        singles: list[dict[str, Any]] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                source = sample.get("source", "")
                cwe = sample.get("cwe", "")
                pid = sample.get("pair_id", "")

                if cwe not in TIER4_CWES:
                    continue
                if source not in _EXEMPLAR_SOURCES:
                    continue

                if pid and pid not in ("", "None", "null", "0"):
                    by_pair[pid].append(sample)
                else:
                    singles.append(sample)

        # build vuln/safe pairs from grouped samples
        # Each tuple: (is_real_world, vuln_code, safe_code) — sort real-world first
        candidate_pairs: dict[str, list[tuple[bool, str, str]]] = defaultdict(list)

        for pid, members in by_pair.items():
            vuln = next((s for s in members if s.get("label") == 1), None)
            safe = next((s for s in members if s.get("label") == 0), None)
            if vuln and safe:
                cwe = vuln.get("cwe", "")
                if cwe in TIER4_CWES:
                    is_real = vuln.get("source", "") in _REAL_SOURCES
                    candidate_pairs[cwe].append(
                        (is_real, vuln["code"], safe["code"])
                    )

        # sort: real-world pairs first, then SARD
        for cwe in candidate_pairs:
            candidate_pairs[cwe].sort(key=lambda x: x[0], reverse=True)

        # diversity-filtered selection
        for cwe, pairs in candidate_pairs.items():
            selected: list[tuple[str, str]] = []
            selected_kws: list[set[str]] = []

            for _is_real, vuln_code, safe_code in pairs:
                if len(selected) >= max_per_cwe:
                    break
                kws = _keyword_set(vuln_code)
                # Jaccard overlap with any already-selected exemplar
                too_similar = any(
                    len(kws & s) / max(len(kws | s), 1) >= 0.80
                    for s in selected_kws
                )
                if not too_similar:
                    selected.append((vuln_code, safe_code))
                    selected_kws.append(kws)

            for vuln_code, safe_code in selected:
                self._db[cwe].append({"vuln": vuln_code, "safe": safe_code})

        # merge builtin exemplars so Tier 4 CWEs always have diverse patterns
        self._merge_builtins()

        # enforce minimum
        for cwe in TIER4_CWES:
            count = len(self._db.get(cwe, []))
            if count < min_real_pairs:
                raise ValueError(
                    f"ExemplarDatabase: only {count} real pairs found for {cwe} "
                    f"(min_real_pairs={min_real_pairs}). "
                    "Run Phase 2 collectors first to populate real-world pairs."
                )

        total = sum(len(v) for v in self._db.values())
        logger.info(
            f"ExemplarDatabase built: {total} exemplars across "
            f"{len(self._db)} CWEs"
        )

    # ── Retrieval ─────────────────────────────────────────────────

    def select_exemplars(
        self,
        cwe: str,
        query_code: str,
        n: int = 2,
    ) -> list[dict[str, str]]:
        """
        Return up to ``n`` exemplars for ``cwe`` ranked by keyword overlap
        with ``query_code`` (descending Jaccard similarity).

        If the CWE is not in the database, returns [].
        """
        pool = self._db.get(cwe, [])
        if not pool:
            return []

        query_kws = _keyword_set(query_code)

        def _score(ex: dict[str, str]) -> float:
            ex_kws = _keyword_set(ex["vuln"])
            union = query_kws | ex_kws
            if not union:
                return 0.0
            return len(query_kws & ex_kws) / len(union)

        ranked = sorted(pool, key=_score, reverse=True)
        return ranked[:n]

    def get(self, cwe: str) -> list[dict[str, str]]:
        """Return all exemplars for a CWE (unranked)."""
        return list(self._db.get(cwe, []))

    def __len__(self) -> int:
        return sum(len(v) for v in self._db.values())

    def __contains__(self, cwe: str) -> bool:
        return bool(self._db.get(cwe))

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write the exemplar database to a JSON file (atomic write)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {cwe: pairs for cwe, pairs in self._db.items()}
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(out.parent), suffix=".tmp", prefix="exemplar_db_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, str(out))
            logger.info(f"ExemplarDatabase saved to {path}")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load(cls, path: str) -> "ExemplarDatabase":
        """Load an exemplar database from a JSON file."""
        db = cls()
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for cwe, pairs in payload.items():
            db._db[cwe] = pairs
        # merge builtin exemplars so Tier 4 CWEs always have diverse patterns
        db._merge_builtins()
        total = len(db)
        logger.info(f"ExemplarDatabase loaded: {total} exemplars from {path}")
        return db


# ── CLI ───────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Tier 4 exemplar database from paired samples.jsonl"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to deduped samples.jsonl"
    )
    parser.add_argument(
        "--output", default="training/data/processed/exemplar_db.json",
        help="Output path for exemplar_db.json"
    )
    parser.add_argument(
        "--max-per-cwe", type=int, default=8,
        help="Maximum exemplars per CWE (default 8)"
    )
    parser.add_argument(
        "--min-real-pairs", type=int, default=1,
        help="Minimum real pairs required per Tier-4 CWE (default 1)"
    )
    args = parser.parse_args()

    db = ExemplarDatabase()
    db.build_from_pairs(
        args.input,
        max_per_cwe=args.max_per_cwe,
        min_real_pairs=args.min_real_pairs,
    )
    db.save(args.output)
    print(f"Saved {len(db)} exemplars to {args.output}")


if __name__ == "__main__":
    _main()
