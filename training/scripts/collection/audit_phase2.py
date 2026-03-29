"""
training/scripts/collection/audit_phase2.py
P2-S7 — Post-collection audit + merge for Phase 2 data.

Runs 10 quality checks across all collected JSONL sources, then writes
data/raw/merged/all_samples.jsonl if no FAILs are found.

Usage:
    python audit_phase2.py [--data-root PATH] [--output-dir PATH]
                           [--min-samples N] [--skip-checks C1,C2,...]
                           [--self-test]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    from schema import validate_sample, VALID_CWE, make_sample_id, make_timestamp
except ImportError:
    raise SystemExit("ERROR: Cannot import schema.py — run from repo root or collection dir")

# ── Constants ────────────────────────────────────────────────────────────────
SOURCE_FILES = {
    "sard":             "sard/sard_samples.jsonl",
    "exploitdb":        "exploitdb/exploitdb_samples.jsonl",
    "cve":              "cve/cve_samples.jsonl",
    "github_advisory":  "github_advisory/github_advisory_samples.jsonl",
    "osv":              "osv/osv_samples.jsonl",
    "repo":             "repo/repo_samples.jsonl",
}

# Sentinels treated as "no pair_id"
_PAIR_SENTINELS = {"", "None", "null", "0"}

CHECK_NAMES = [
    "CHECK1",  # Schema validity
    "CHECK2",  # Total count
    "CHECK3",  # Per-CWE minimum
    "CHECK4",  # Label balance
    "CHECK5",  # CFA pair integrity
    "CHECK6",  # Commit SHA cross-source dedup
    "CHECK7",  # Code length sanity
    "CHECK8",  # Source distribution
    "CHECK9",  # Duplicate code (MD5)
    "CHECK10", # Disk space
]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_sources(data_root: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """
    Load all source JSONL files. Returns:
        source_data: {source_name: [sample, ...]}
        warnings:    list of warning strings
    """
    source_data: dict[str, list[dict]] = {}
    warnings: list[str] = []

    for name, rel_path in SOURCE_FILES.items():
        full_path = data_root / rel_path
        if not full_path.exists():
            warnings.append(f"Source file not found (skipped): {full_path}")
            continue
        samples = []
        bad_lines = 0
        with open(full_path, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    bad_lines += 1
                    if bad_lines <= 3:
                        warnings.append(
                            f"{full_path.name}:{lineno} — JSON decode error: {exc}"
                        )
        if bad_lines > 3:
            warnings.append(f"{full_path.name} — {bad_lines} total bad lines (showing first 3)")
        source_data[name] = samples
        print(f"  Loaded {len(samples):>6,} samples from {full_path.name}")

    return source_data, warnings


# ── Check helpers ─────────────────────────────────────────────────────────────

def _md5_code(code: str) -> str:
    normalized = " ".join(code.split())
    return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()


def _effective_pair_id(s: dict) -> str:
    pid = s.get("pair_id", "")
    return "" if str(pid) in _PAIR_SENTINELS else str(pid)


# ── The 10 checks ─────────────────────────────────────────────────────────────

def check1_schema_validity(
    source_data: dict[str, list[dict]],
) -> tuple[str, str, dict]:
    """Every sample must pass validate_sample()."""
    failed_count = 0
    first_errors: list[str] = []
    for source, samples in source_data.items():
        for s in samples:
            ok, errs = validate_sample(s)
            if not ok:
                failed_count += 1
                if len(first_errors) < 5:
                    first_errors.append(
                        f"[{source}] id={s.get('id','?')}: {'; '.join(errs)}"
                    )

    if failed_count == 0:
        status = "PASS"
        msg = "All samples pass schema validation"
    else:
        status = "FAIL"
        msg = f"{failed_count} sample(s) failed validation. First errors:\n"
        msg += "\n".join(f"    {e}" for e in first_errors)

    return status, msg, {"failed": failed_count, "first_errors": first_errors}


def check2_total_count(
    all_samples: list[dict], min_samples: int
) -> tuple[str, str, dict]:
    """Total must meet minimum threshold."""
    total = len(all_samples)
    if total >= 30_000:
        status = "PASS"
        msg = f"Total: {total:,} samples (>= 30,000 target)"
    elif total >= min_samples:
        status = "WARN"
        msg = f"Total: {total:,} samples — below 30,000 target (>= {min_samples:,} minimum)"
    else:
        status = "FAIL"
        msg = f"Total: {total:,} samples — below minimum threshold of {min_samples:,}"

    return status, msg, {"total": total, "min_required": min_samples}


def check3_per_cwe_minimum(
    all_samples: list[dict],
) -> tuple[str, str, dict]:
    """Every CWE in VALID_CWE should have >= 200 samples."""
    counts: dict[str, int] = defaultdict(int)
    for s in all_samples:
        cwe = s.get("cwe", "UNKNOWN")
        counts[cwe] += 1

    under_200: list[str] = []
    under_500: list[str] = []

    for cwe in sorted(VALID_CWE):
        n = counts.get(cwe, 0)
        if n < 200:
            under_200.append(f"{cwe}: {n}")
        elif n < 500:
            under_500.append(f"{cwe}: {n}")

    # Report per-CWE counts (valid CWEs only)
    cwe_report = {cwe: counts.get(cwe, 0) for cwe in sorted(VALID_CWE)}
    # Also report unknown CWEs seen
    unknown = {k: v for k, v in counts.items() if k not in VALID_CWE}

    detail_lines = [f"    {cwe}: {n:,}" for cwe, n in sorted(cwe_report.items())]
    if unknown:
        detail_lines.append(
            "    Unknown CWEs: " + ", ".join(f"{k}={v}" for k, v in sorted(unknown.items()))
        )

    if under_200:
        status = "FAIL"
        msg = (
            f"CWEs with < 200 samples: {', '.join(under_200)}\n"
            + "\n".join(detail_lines)
        )
    elif under_500:
        status = "WARN"
        msg = (
            f"CWEs with 200-499 samples (under 500 target): {', '.join(under_500)}\n"
            + "\n".join(detail_lines)
        )
    else:
        status = "PASS"
        msg = "All CWEs >= 500 samples\n" + "\n".join(detail_lines)

    return status, msg, {"cwe_counts": cwe_report, "under_200": under_200, "under_500": under_500}


def check4_label_balance(
    all_samples: list[dict],
) -> tuple[str, str, dict]:
    """40–60% vulnerable label distribution."""
    total = len(all_samples)
    if total == 0:
        return "FAIL", "No samples", {"label_1_pct": 0}

    label_1 = sum(1 for s in all_samples if s.get("label") == 1)
    label_0 = sum(1 for s in all_samples if s.get("label") == 0)
    pct = label_1 / total * 100

    detail = f"vuln={label_1:,} ({pct:.1f}%), safe={label_0:,} ({100-pct:.1f}%)"
    if pct < 30 or pct > 70:
        status = "FAIL"
        msg = f"Label balance severely skewed: {detail}"
    elif pct < 40 or pct > 60:
        status = "WARN"
        msg = f"Label balance outside 40-60% target: {detail}"
    else:
        status = "PASS"
        msg = f"Label balance within target range: {detail}"

    return status, msg, {"label_1": label_1, "label_0": label_0, "label_1_pct": round(pct, 2)}


def check5_cfa_pair_integrity(
    all_samples: list[dict],
) -> tuple[str, str, dict]:
    """Every pair_id must have both label=0 and label=1."""
    pairs: dict[str, set[int]] = defaultdict(set)
    for s in all_samples:
        pid = _effective_pair_id(s)
        if pid:
            pairs[pid].add(s.get("label"))

    broken: list[str] = []
    valid_pairs: list[str] = []
    for pid, labels in pairs.items():
        if 0 in labels and 1 in labels:
            valid_pairs.append(pid)
        else:
            broken.append(pid)

    if broken:
        status = "FAIL"
        first3 = broken[:3]
        msg = (
            f"{len(broken)} broken pair(s) found (showing first 3): {first3}\n"
            f"    Valid pairs: {len(valid_pairs):,}  |  Broken: {len(broken)}"
        )
    else:
        status = "PASS"
        msg = f"{len(valid_pairs):,} valid CFA pairs, 0 broken"

    return status, msg, {"valid_pairs": len(valid_pairs), "broken_pairs": len(broken)}


def check6_commit_sha_uniqueness(
    source_data: dict[str, list[dict]],
) -> tuple[str, str, dict]:
    """No commit_sha should appear in 2+ different source files."""
    # sha -> set of source names
    sha_sources: dict[str, set[str]] = defaultdict(set)
    for source, samples in source_data.items():
        for s in samples:
            sha = s.get("commit_sha", "")
            if sha and sha not in ("", "None", "null"):
                sha_sources[sha].add(source)

    duplicates = {
        sha: sorted(srcs)
        for sha, srcs in sha_sources.items()
        if len(srcs) > 1
    }

    if duplicates:
        first3 = list(duplicates.items())[:3]
        first3_str = "; ".join(f"{sha[:12]}... in {srcs}" for sha, srcs in first3)
        status = "FAIL"
        msg = (
            f"{len(duplicates)} commit SHA(s) appear in multiple sources. "
            f"First 3: {first3_str}"
        )
    else:
        total_shas = sum(
            1 for sha, srcs in sha_sources.items() if len(srcs) == 1
        )
        status = "PASS"
        msg = f"No cross-source SHA duplicates. {total_shas:,} unique SHAs across all sources."

    return status, msg, {"duplicate_shas": len(duplicates)}


def check7_code_length_sanity(
    all_samples: list[dict],
) -> tuple[str, str, dict]:
    """Code length: 5-500 non-blank lines (schema should catch this, but verify)."""
    too_short: list[str] = []
    too_long: list[str] = []
    for s in all_samples:
        code = s.get("code", "")
        non_blank = [l for l in code.splitlines() if l.strip()]
        n = len(non_blank)
        if n < 5:
            too_short.append(s.get("id", "?"))
        elif n > 500:
            too_long.append(s.get("id", "?"))

    if too_short or too_long:
        status = "FAIL"
        parts = []
        if too_short:
            parts.append(f"{len(too_short)} samples < 5 lines (ids: {too_short[:3]})")
        if too_long:
            parts.append(f"{len(too_long)} samples > 500 lines (ids: {too_long[:3]})")
        msg = "; ".join(parts)
    else:
        status = "PASS"
        msg = "All samples have 5–500 non-blank lines"

    return status, msg, {"too_short": len(too_short), "too_long": len(too_long)}


def check8_source_distribution(
    source_data: dict[str, list[dict]],
    all_samples: list[dict],
) -> tuple[str, str, dict]:
    """Warn if any single source > 80% of total."""
    total = len(all_samples)
    if total == 0:
        return "FAIL", "No samples", {}

    source_counts = {src: len(samples) for src, samples in source_data.items()}
    source_pcts = {src: count / total * 100 for src, count in source_counts.items()}

    dominant = {src: pct for src, pct in source_pcts.items() if pct > 80}

    detail_lines = [
        f"    {src}: {count:,} ({pct:.1f}%)"
        for src, (count, pct) in sorted(
            {s: (source_counts[s], source_pcts[s]) for s in source_counts}.items(),
            key=lambda x: -x[1][0],
        )
    ]
    detail = "\n".join(detail_lines)

    if dominant:
        status = "WARN"
        dom_str = ", ".join(f"{s} ({p:.1f}%)" for s, p in dominant.items())
        msg = f"Source(s) dominate > 80%: {dom_str}\n{detail}"
    else:
        status = "PASS"
        msg = f"Source distribution looks healthy\n{detail}"

    return status, msg, {"source_pcts": {s: round(p, 1) for s, p in source_pcts.items()}}


def check9_duplicate_code(
    all_samples: list[dict],
) -> tuple[str, str, dict]:
    """Warn if MD5 duplicate rate > 5%."""
    seen: dict[str, str] = {}  # md5 -> first sample id
    dup_count = 0
    for s in all_samples:
        h = _md5_code(s.get("code", ""))
        if h in seen:
            dup_count += 1
        else:
            seen[h] = s.get("id", "?")

    total = len(all_samples)
    dup_rate = dup_count / total * 100 if total > 0 else 0

    if dup_rate > 5:
        status = "WARN"
        msg = (
            f"{dup_count:,} exact duplicates ({dup_rate:.1f}% of total). "
            "These will be removed by Stage 2 MinHash dedup."
        )
    else:
        status = "PASS"
        msg = f"{dup_count:,} exact duplicates ({dup_rate:.1f}%) — within acceptable range"

    return status, msg, {"duplicate_count": dup_count, "duplicate_rate_pct": round(dup_rate, 2)}


def check10_disk_space(
    output_dir: Path,
) -> tuple[str, str, dict]:
    """Phase 3-4 needs >= 15 GB free on the data volume."""
    try:
        usage = shutil.disk_usage(str(output_dir.parent))
        free_gb = usage.free / (1024 ** 3)
    except OSError as exc:
        return "WARN", f"Could not check disk space: {exc}", {"free_gb": None}

    if free_gb < 15:
        status = "FAIL"
        msg = f"Only {free_gb:.1f} GB free — Phase 3-4 needs >= 15 GB"
    else:
        status = "PASS"
        msg = f"{free_gb:.1f} GB free on data volume (need >= 15 GB)"

    return status, msg, {"free_gb": round(free_gb, 1)}


# ── Runner ───────────────────────────────────────────────────────────────────

def run_audit(
    data_root: Path,
    output_dir: Path,
    min_samples: int = 10_000,
    skip_checks: set[str] | None = None,
) -> int:
    """
    Run the full Phase 2 audit.
    Returns 0 on PASS/WARN-only, 1 if any FAIL.
    """
    skip_checks = skip_checks or set()

    print("\n" + "=" * 70)
    print("  STREAMGUARD - PHASE 2 POST-COLLECTION AUDIT")
    print(f"  data-root : {data_root}")
    print(f"  output-dir: {output_dir}")
    print(f"  timestamp : {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[LOADING SOURCES]")
    source_data, load_warnings = load_sources(data_root)
    all_samples = [s for samples in source_data.values() for s in samples]
    print(f"  Total loaded: {len(all_samples):,} samples from {len(source_data)} source(s)")
    for w in load_warnings:
        print(f"  WARN: {w}")

    if not all_samples:
        print("\nFATAL: No samples loaded — nothing to audit.")
        return 1

    # ── Run checks ────────────────────────────────────────────────────────
    print("\n[RUNNING CHECKS]")
    results: dict[str, dict] = {}
    has_fail = False

    def _run(check_id: str, fn, *args):
        nonlocal has_fail
        if check_id in skip_checks:
            print(f"  {check_id}: SKIP (--skip-checks)")
            results[check_id] = {"status": "SKIP", "message": "Skipped by CLI flag", "data": {}}
            return
        status, msg, data = fn(*args)
        results[check_id] = {"status": status, "message": msg, "data": data}
        icon = {"PASS": "OK", "FAIL": "!!", "WARN": "W"}.get(status, "?")
        print(f"\n  {check_id} [{status}] {icon}")
        for line in msg.splitlines():
            print(f"    {line}")
        if status == "FAIL":
            has_fail = True

    _run("CHECK1",  check1_schema_validity,     source_data)
    _run("CHECK2",  check2_total_count,          all_samples, min_samples)
    _run("CHECK3",  check3_per_cwe_minimum,      all_samples)
    _run("CHECK4",  check4_label_balance,        all_samples)
    _run("CHECK5",  check5_cfa_pair_integrity,   all_samples)
    _run("CHECK6",  check6_commit_sha_uniqueness, source_data)
    _run("CHECK7",  check7_code_length_sanity,   all_samples)
    _run("CHECK8",  check8_source_distribution,  source_data, all_samples)
    _run("CHECK9",  check9_duplicate_code,       all_samples)
    _run("CHECK10", check10_disk_space,          output_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    print(f"  {'Check':<10} {'Status':<8}")
    print(f"  {'-'*9:<10} {'-'*6:<8}")
    for cid in CHECK_NAMES:
        r = results.get(cid, {})
        st = r.get("status", "?")
        print(f"  {cid:<10} {st:<8}")

    # ── Merge or abort ────────────────────────────────────────────────────
    print()
    if has_fail:
        print("PHASE 2 AUDIT FAILED — Fix the issues above before proceeding to Phase 3")
        print("=" * 70)
        return 1

    # Write merged output
    print("[WRITING MERGED OUTPUT]")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "all_samples.jsonl"
    report_path = output_dir / "audit_report.json"

    # Atomic write via .tmp
    tmp_merged = merged_path.with_suffix(".tmp")
    written = 0
    with open(tmp_merged, "w", encoding="utf-8") as fh:
        for s in all_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            written += 1
    os.replace(tmp_merged, merged_path)
    print(f"  Wrote {written:,} samples → {merged_path}")

    # Write audit report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_samples": len(all_samples),
        "sources_loaded": {src: len(samp) for src, samp in source_data.items()},
        "load_warnings": load_warnings,
        "checks": results,
        "merged_path": str(merged_path),
    }
    tmp_report = report_path.with_suffix(".tmp")
    tmp_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_report, report_path)
    print(f"  Wrote audit report → {report_path}")

    print("\nPHASE 2 AUDIT PASSED — Ready for Phase 3 Stage 1")
    print("=" * 70)
    return 0


# ── Self-test ────────────────────────────────────────────────────────────────

def _make_sample(
    source: str = "sard",
    cwe: str = "CWE-121",
    label: int = 1,
    pair_id: str = "",
    commit_sha: str = "",
    code: str | None = None,
) -> dict:
    if code is None:
        code = (
            "void vuln(char *src) {\n"
            "    char dest[10];\n"
            "    strcpy(dest, src);\n"
            "    dest[0] = 'x';\n"
            "    printf(\"%s\\n\", dest);\n"
            "}\n"
        )
    return {
        "id": make_sample_id(),
        "source": source,
        "code": code,
        "label": label,
        "cwe": cwe,
        "language": "c",
        "collected_at": make_timestamp(),
        "pair_id": pair_id,
        "commit_sha": commit_sha,
        "severity_score": 7.5,
        "cve_id": "",
        "file_path": "",
        "cfa_origin": "native",
        "aliases": {},
        "metadata": {},
    }


def run_self_test() -> int:
    print("\n" + "=" * 70)
    print("  SELF-TEST MODE")
    print("=" * 70)

    passed = 0
    failed = 0

    def _assert(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))

    # ── CHECK1: Schema validity ──────────────────────────────────────────
    print("\n[CHECK1] Schema validity")
    good = _make_sample()
    bad = {"id": "", "source": "sard", "code": "x", "label": 2, "cwe": "CWE-121",
           "language": "c", "collected_at": make_timestamp()}
    status, _, data = check1_schema_validity({"sard": [good, bad]})
    _assert("CHECK1 detects 1 bad sample", status == "FAIL", f"status={status}")
    _assert("CHECK1 failed count is 1", data["failed"] == 1, f"got {data['failed']}")
    status2, _, _ = check1_schema_validity({"sard": [good]})
    _assert("CHECK1 PASS on valid sample", status2 == "PASS", f"status={status2}")

    # ── CHECK2: Total count ──────────────────────────────────────────────
    print("\n[CHECK2] Total count")
    status, _, _ = check2_total_count([good] * 5_000, 10_000)
    _assert("CHECK2 FAIL below min", status == "FAIL", f"status={status}")
    status, _, _ = check2_total_count([good] * 15_000, 10_000)
    _assert("CHECK2 WARN 10K-29K", status == "WARN", f"status={status}")
    status, _, _ = check2_total_count([good] * 30_000, 10_000)
    _assert("CHECK2 PASS at 30K", status == "PASS", f"status={status}")

    # ── CHECK3: Per-CWE minimum ──────────────────────────────────────────
    print("\n[CHECK3] Per-CWE minimum")
    # Make 200 samples for one CWE, 0 for most
    samples_c3 = [_make_sample(cwe="CWE-121")] * 200
    status, msg, data = check3_per_cwe_minimum(samples_c3)
    _assert("CHECK3 FAIL when most CWEs < 200", status == "FAIL", f"status={status}")
    # Feed 200 per CWE for all CWEs
    all_cwe_samples = [_make_sample(cwe=cwe) for cwe in VALID_CWE for _ in range(500)]
    status, _, _ = check3_per_cwe_minimum(all_cwe_samples)
    _assert("CHECK3 PASS when all CWEs >= 500", status == "PASS", f"status={status}")

    # ── CHECK4: Label balance ────────────────────────────────────────────
    print("\n[CHECK4] Label balance")
    skewed = [_make_sample(label=1)] * 80 + [_make_sample(label=0)] * 20
    status, _, _ = check4_label_balance(skewed)
    _assert("CHECK4 FAIL on 80% vuln", status in ("FAIL", "WARN"), f"status={status}")
    balanced = [_make_sample(label=1)] * 50 + [_make_sample(label=0)] * 50
    status, _, _ = check4_label_balance(balanced)
    _assert("CHECK4 PASS on 50/50", status == "PASS", f"status={status}")

    # ── CHECK5: CFA pair integrity ───────────────────────────────────────
    print("\n[CHECK5] CFA pair integrity")
    good_pair = [
        _make_sample(label=1, pair_id="p1"),
        _make_sample(label=0, pair_id="p1"),
    ]
    broken_pair = [
        _make_sample(label=1, pair_id="p2"),
        _make_sample(label=1, pair_id="p2"),  # two vuln, no safe
    ]
    status, _, data = check5_cfa_pair_integrity(good_pair + broken_pair)
    _assert("CHECK5 FAIL on broken pair", status == "FAIL", f"status={status}")
    _assert("CHECK5 reports 1 broken", data["broken_pairs"] == 1,
            f"got {data['broken_pairs']}")
    status2, _, data2 = check5_cfa_pair_integrity(good_pair)
    _assert("CHECK5 PASS on valid pairs", status2 == "PASS", f"status={status2}")
    _assert("CHECK5 reports 1 valid pair", data2["valid_pairs"] == 1,
            f"got {data2['valid_pairs']}")

    # ── CHECK6: Commit SHA cross-source dedup ────────────────────────────
    print("\n[CHECK6] Commit SHA uniqueness")
    s_cve = _make_sample(source="cve", commit_sha="abc123")
    s_osv = _make_sample(source="osv", commit_sha="abc123")  # same SHA!
    status, _, data = check6_commit_sha_uniqueness({"cve": [s_cve], "osv": [s_osv]})
    _assert("CHECK6 FAIL on duplicate SHA", status == "FAIL", f"status={status}")
    _assert("CHECK6 reports 1 dup", data["duplicate_shas"] == 1,
            f"got {data['duplicate_shas']}")
    s_unique = _make_sample(source="osv", commit_sha="def456")
    status2, _, _ = check6_commit_sha_uniqueness({"cve": [s_cve], "osv": [s_unique]})
    _assert("CHECK6 PASS on unique SHAs", status2 == "PASS", f"status={status2}")

    # ── CHECK7: Code length sanity ───────────────────────────────────────
    print("\n[CHECK7] Code length sanity")
    short_code = "x=1;\ny=2;\n"  # only 2 non-blank lines
    short_sample = _make_sample(code=short_code)
    # Force it through (schema may reject it too, but check7 is independent)
    status, _, data = check7_code_length_sanity([short_sample])
    _assert("CHECK7 FAIL on short code", status == "FAIL", f"status={status}")
    status2, _, _ = check7_code_length_sanity([good])
    _assert("CHECK7 PASS on normal code", status2 == "PASS", f"status={status2}")

    # ── CHECK8: Source distribution ──────────────────────────────────────
    print("\n[CHECK8] Source distribution")
    sard_heavy = {"sard": [good] * 90, "cve": [good] * 10}
    all_90 = [good] * 100
    status, _, data = check8_source_distribution(sard_heavy, all_90)
    _assert("CHECK8 WARN on 90% dominance", status == "WARN", f"status={status}")
    balanced_src = {"sard": [good] * 50, "cve": [good] * 50}
    status2, _, _ = check8_source_distribution(balanced_src, [good] * 100)
    _assert("CHECK8 PASS on balanced sources", status2 == "PASS", f"status={status2}")

    # ── CHECK9: Duplicate code ───────────────────────────────────────────
    print("\n[CHECK9] Duplicate code")
    dups = [good] * 10  # all same code
    status, _, data = check9_duplicate_code(dups)
    _assert("CHECK9 WARN on 90% dup rate", status == "WARN", f"status={status}")
    # 4 unique out of 100 = 96 dups = 96% dup rate
    unique_samples = [_make_sample(code=good["code"].replace("vuln", f"f{i}")) for i in range(100)]
    status2, _, _ = check9_duplicate_code(unique_samples)
    _assert("CHECK9 PASS on unique code", status2 == "PASS", f"status={status2}")

    # ── CHECK10: Disk space ──────────────────────────────────────────────
    print("\n[CHECK10] Disk space")
    # Use a real temp dir — just verify it returns a status and doesn't crash
    with tempfile.TemporaryDirectory() as td:
        status, msg, data = check10_disk_space(Path(td))
        _assert(
            "CHECK10 returns valid status",
            status in ("PASS", "FAIL", "WARN"),
            f"status={status}",
        )
        _assert("CHECK10 returns free_gb", "free_gb" in data, f"data={data}")
        print(f"    Disk check result: {status} - {msg}")

    # ── Full pipeline test ───────────────────────────────────────────────
    print("\n[FULL PIPELINE TEST]")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_root = tmpdir / "raw"

        # Write synthetic source files
        sard_dir = data_root / "sard"
        sard_dir.mkdir(parents=True)

        # Build 300 samples per CWE (enough for all CWE checks)
        # with 50/50 pairs
        def _synth_code(fname: str, safe: bool, tag: str) -> str:
            # 6 non-blank lines, 15+ tokens — passes schema validation
            buf = "fgets" if safe else "gets"
            return (
                f"void {fname}(char *src) {{\n"
                f"    char dest[64];\n"
                f"    int len = sizeof(dest);\n"
                f"    {buf}(dest, src);\n"
                f"    dest[0] = 0;\n"
                f"    return; /* {tag} */\n"
                f"}}"
            )

        samples_per_cwe = 300
        synth: list[dict] = []
        for cwe in VALID_CWE:
            for i in range(samples_per_cwe // 2):
                pid = make_sample_id()
                tag = f"{cwe}_{i}"
                synth.append(_make_sample(cwe=cwe, label=1, pair_id=pid,
                                          code=_synth_code(f"fv_{tag}", False, tag)))
                synth.append(_make_sample(cwe=cwe, label=0, pair_id=pid,
                                          code=_synth_code(f"fs_{tag}", True, tag)))

        with open(sard_dir / "sard_samples.jsonl", "w") as fh:
            for s in synth:
                fh.write(json.dumps(s) + "\n")

        out_dir = tmpdir / "merged"
        ret = run_audit(
            data_root=data_root,
            output_dir=out_dir,
            min_samples=100,
            skip_checks={"CHECK10"},  # disk check irrelevant in temp dir
        )
        _assert("Full pipeline PASS exit code 0", ret == 0, f"exit code {ret}")
        _assert("Merged file created", (out_dir / "all_samples.jsonl").exists())
        _assert("Audit report created", (out_dir / "audit_report.json").exists())

        # Verify line count
        lines = (out_dir / "all_samples.jsonl").read_text().strip().splitlines()
        _assert(
            f"Merged file has {len(synth)} lines",
            len(lines) == len(synth),
            f"got {len(lines)}",
        )

        # Test FAIL path — inject a broken pair
        broken_dir = tmpdir / "raw2"
        broken_sard = broken_dir / "sard"
        broken_sard.mkdir(parents=True)
        broken_samples = synth[:10] + [_make_sample(label=1, pair_id="broken_pid"),
                                        _make_sample(label=1, pair_id="broken_pid")]
        with open(broken_sard / "sard_samples.jsonl", "w") as fh:
            for s in broken_samples:
                fh.write(json.dumps(s) + "\n")
        broken_out = tmpdir / "merged_broken"
        ret2 = run_audit(
            data_root=broken_dir,
            output_dir=broken_out,
            min_samples=1,
            skip_checks={"CHECK2", "CHECK3", "CHECK4", "CHECK8", "CHECK9", "CHECK10"},
        )
        _assert("FAIL pipeline returns exit 1", ret2 == 1, f"exit code {ret2}")
        _assert("Merged file NOT written on FAIL", not (broken_out / "all_samples.jsonl").exists())

    # ── Final tally ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  Self-test results: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StreamGuard Phase 2 post-collection audit + merge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="training/scripts/collection/data/raw",
        help="Root directory containing per-source subdirectories (default: training/scripts/collection/data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="training/scripts/collection/data/raw/merged",
        help="Output directory for all_samples.jsonl and audit_report.json",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10_000,
        help="Minimum total samples required to pass CHECK2 (default: 10,000)",
    )
    parser.add_argument(
        "--skip-checks",
        default="",
        help="Comma-separated list of checks to skip, e.g. CHECK8,CHECK10",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run inline self-tests and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.self_test:
        return run_self_test()

    skip = {c.strip().upper() for c in args.skip_checks.split(",") if c.strip()}
    unknown_skips = skip - set(CHECK_NAMES)
    if unknown_skips:
        print(f"WARNING: Unknown check names in --skip-checks: {unknown_skips}")
        print(f"         Valid names: {', '.join(CHECK_NAMES)}")

    # Resolve paths relative to CWD
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    return run_audit(
        data_root=data_root,
        output_dir=output_dir,
        min_samples=args.min_samples,
        skip_checks=skip,
    )


if __name__ == "__main__":
    sys.exit(main())
