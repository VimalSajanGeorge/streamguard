"""
training/scripts/collection/schema.py
Canonical sample schema for StreamGuard data pipeline.
Source: docs/New Docs/DATA_PIPELINE.md §Canonical Sample Schema
        cfa_gnn_proof_technical_plan.md.resolved Appendix Conflict 1
"""
from __future__ import annotations
import uuid
from datetime import datetime

VALID_CWE = {
    "CWE-89", "CWE-78", "CWE-79", "CWE-119", "CWE-120",
    "CWE-121", "CWE-122", "CWE-125", "CWE-134", "CWE-190",
    "CWE-416", "CWE-476"
}

VALID_SOURCES = {
    "sard", "exploitdb", "cve", "github_advisory",
    "osv", "repo", "manual",
    "sard_cfa", "exploitdb_cfa", "cve_cfa",
    "github_advisory_cfa", "osv_cfa", "repo_cfa",
}

# Defaults for optional forward-compat fields (Appendix Conflict 1)
OPTIONAL_FIELD_DEFAULTS = {
    "severity_score": 0.0,      # CVSS float for L_severity head
    "commit_sha":     "",        # cross-source dedup in Stage 7
    "cve_id":         "",        # cross-source dedup in Stage 2/7
    "cfa_origin":     "native",  # "native" | "llm_generated"
    "aliases":        {},        # {cve: "...", ghsa: "...", osv: "..."}
    "metadata":       {},        # extensible catch-all
    "pair_id":        "",        # CFA pair linkage (empty = unpaired)
    "file_path":      "",        # source file path (informational)
}


def validate_sample(s: dict) -> tuple[bool, list[str]]:
    """
    Validate a sample dict against the canonical schema.
    Returns (is_valid, list_of_errors). All errors must be empty to save.
    Also fills in OPTIONAL_FIELD_DEFAULTS for any missing optional fields.
    Note: mutates s in-place to fill optional defaults.
    """
    errors = []

    # --- Required fields ---
    if not s.get("id"):
        errors.append("missing id")
    if not s.get("source"):
        errors.append("missing source")
    if not s.get("code"):
        errors.append("missing code")
    if s.get("label") not in (0, 1):
        errors.append("label must be 0 or 1")
    if not s.get("cwe"):
        errors.append("missing cwe")
    if not s.get("language"):
        errors.append("missing language")
    if not s.get("collected_at"):
        errors.append("missing collected_at")

    # --- Value validation ---
    if s.get("source") and s["source"] not in VALID_SOURCES:
        errors.append(f"invalid source: {s['source']}")
    if s.get("cwe") and s["cwe"] not in VALID_CWE:
        errors.append(f"invalid CWE: {s['cwe']}")
    if s.get("language") and s["language"] != "c":
        errors.append("language must be 'c'")

    # --- Code sanity ---
    code = s.get("code", "")
    lines = [l for l in code.splitlines() if l.strip()]  # non-blank lines
    if len(lines) < 5:
        errors.append(f"code too short: {len(lines)} non-blank lines (min 5)")
    if len(lines) > 500:
        errors.append(f"code too long: {len(lines)} lines (max 500)")
    tokens = code.split()
    if len(tokens) < 10:
        errors.append(f"too few tokens: {len(tokens)} (min 10)")
    if len(tokens) > 4096:
        errors.append(f"too many tokens: {len(tokens)} (max 4096)")

    # --- Fill optional field defaults (mutates in place) ---
    for field, default in OPTIONAL_FIELD_DEFAULTS.items():
        if field not in s:
            s[field] = default

    return len(errors) == 0, errors


def make_sample_id() -> str:
    """Generate a new unique sample ID (UUID4)."""
    return str(uuid.uuid4())


def make_timestamp() -> str:
    """Return current UTC time in ISO 8601 format with Z suffix."""
    return datetime.utcnow().isoformat() + "Z"
