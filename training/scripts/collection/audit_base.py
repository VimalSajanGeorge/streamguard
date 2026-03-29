"""
training/scripts/collection/audit_base.py
P2-S1 verification audit: 6 checks for hardened BaseCollector + Schema.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from base_collector import BaseCollector
from schema import validate_sample, make_sample_id, make_timestamp

PASS_COUNT = 0
FAIL_COUNT = 0

VALID_CODE = """\
int process_input(char *buf, int len) {
    char temp[256];
    if (len > 256) {
        return -1;
    }
    memcpy(temp, buf, len);
    temp[len] = '\\0';
    printf("Processed: %s\\n", temp);
    return 0;
}"""


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}: {detail}")


def make_valid_sample(**overrides) -> dict:
    s = {
        "id": make_sample_id(),
        "source": "sard",
        "code": VALID_CODE,
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": make_timestamp(),
    }
    s.update(overrides)
    return s


def make_unique_code(suffix: str) -> str:
    return VALID_CODE.replace("process_input", f"func_{suffix}")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  CHECK 1: Hash sidecar persistence")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_audit1_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="audit",
    )
    for i in range(5):
        s = make_valid_sample(code=make_unique_code(f"c1_{i}"))
        c.save_sample(s)

    hash_file = os.path.join(tmpdir, "out", "audit_hashes.txt")
    check("Hash sidecar file created", os.path.exists(hash_file))
    with open(hash_file, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    check("Hash sidecar has 5 lines", len(lines) == 5, f"got {len(lines)}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  CHECK 2: Fast-path hash reload + dedup on restart")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_audit2_")
try:
    c1 = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="audit",
    )
    samples = []
    for i in range(3):
        s = make_valid_sample(code=make_unique_code(f"c2_{i}"))
        c1.save_sample(s)
        samples.append(s)

    # Restart — should use fast path (hash sidecar exists)
    c2 = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="audit",
    )
    check("Hashes loaded on restart", len(c2._seen_hashes) == 3,
          f"got {len(c2._seen_hashes)}")

    # Duplicate rejected
    dup_result = c2.save_sample(samples[0])
    check("Duplicate rejected after restart", dup_result is False)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  CHECK 3: Valid CFA pair saved with pair_id intact")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_audit3_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="audit",
    )
    pair_id = "pair_test_001"
    s1 = make_valid_sample(code=make_unique_code("c3_vuln"), label=1, pair_id=pair_id)
    s2 = make_valid_sample(code=make_unique_code("c3_safe"), label=0, pair_id=pair_id)
    c.save_sample(s1)
    c.save_sample(s2)

    out_file = os.path.join(tmpdir, "out", "audit_samples.jsonl")
    with open(out_file, encoding="utf-8") as f:
        saved = [json.loads(l) for l in f if l.strip()]
    check("Both samples saved", len(saved) == 2, f"got {len(saved)}")
    check("First sample pair_id intact", saved[0]["pair_id"] == pair_id,
          f"got {saved[0].get('pair_id')}")
    check("Second sample pair_id intact", saved[1]["pair_id"] == pair_id,
          f"got {saved[1].get('pair_id')}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  CHECK 4: Same-label duplicate pair_id cleared")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_audit4_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="audit",
    )
    pair_id = "pair_dup_label"
    s1 = make_valid_sample(code=make_unique_code("c4_a"), label=1, pair_id=pair_id)
    s2 = make_valid_sample(code=make_unique_code("c4_b"), label=1, pair_id=pair_id)
    c.save_sample(s1)
    c.save_sample(s2)

    out_file = os.path.join(tmpdir, "out", "audit_samples.jsonl")
    with open(out_file, encoding="utf-8") as f:
        saved = [json.loads(l) for l in f if l.strip()]
    check("First sample pair_id kept", saved[0]["pair_id"] == pair_id,
          f"got {saved[0].get('pair_id')}")
    check("Second sample pair_id cleared", saved[1]["pair_id"] == "",
          f"got {saved[1].get('pair_id')}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  CHECK 5: finalize_pairs() clears orphan pair_ids")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_audit5_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="audit",
    )
    # Save one sample with a pair_id that has no mate (orphan)
    orphan_pid = "orphan_001"
    s = make_valid_sample(code=make_unique_code("c5_orphan"), label=1, pair_id=orphan_pid)
    c.save_sample(s)

    # Also save a proper pair
    good_pid = "good_pair_001"
    s1 = make_valid_sample(code=make_unique_code("c5_g1"), label=1, pair_id=good_pid)
    s2 = make_valid_sample(code=make_unique_code("c5_g2"), label=0, pair_id=good_pid)
    c.save_sample(s1)
    c.save_sample(s2)

    cleared = c.finalize_pairs()
    check("finalize_pairs returns 1 orphan cleared", cleared == 1, f"got {cleared}")

    out_file = os.path.join(tmpdir, "out", "audit_samples.jsonl")
    with open(out_file, encoding="utf-8") as f:
        saved = [json.loads(l) for l in f if l.strip()]
    orphan_sample = [s for s in saved if s["id"] == saved[0]["id"]][0]
    check("Orphan pair_id cleared in JSONL", orphan_sample["pair_id"] == "",
          f"got {orphan_sample.get('pair_id')}")
    good_samples = [s for s in saved if s.get("pair_id") == good_pid]
    check("Good pair pair_ids preserved", len(good_samples) == 2,
          f"got {len(good_samples)}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  CHECK 6: Schema rejects >4096 tokens")
print("=" * 64)

big_code = "int x;\n" * 5 + ("token " * 4097)
s = make_valid_sample(code=big_code)
ok, errs = validate_sample(s)
check("Sample with >4096 tokens rejected", not ok, f"was accepted; errors={errs}")
check("Error mentions 'too many tokens'",
      any("too many tokens" in e for e in errs), f"errors={errs}")


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print(f"  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed"
      f" out of {PASS_COUNT + FAIL_COUNT}")
if FAIL_COUNT:
    print(f"\n  {FAIL_COUNT} check(s) FAILED")
else:
    print(f"\n  All {PASS_COUNT} checks passed!")
print("=" * 64 + "\n")

sys.exit(FAIL_COUNT)
