"""
Story 2 Verification: schema.py + base_collector.py
Runs 5 test cases as specified in the acceptance criteria.
"""
import sys, os, json, tempfile, shutil

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.scripts.collection.schema import validate_sample, make_sample_id, make_timestamp
from training.scripts.collection.base_collector import BaseCollector

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  \u2705 {name}")
    else:
        FAIL_COUNT += 1
        print(f"  \u274c {name}: {detail}")

# ── Helper: valid C function with enough lines/tokens ───────────────
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

def make_valid_sample():
    return {
        "id": make_sample_id(),
        "source": "sard",
        "code": VALID_CODE,
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": make_timestamp(),
    }


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 1: validate_sample() accepts a correct SARD sample")
print("=" * 64)

s = make_valid_sample()
is_valid, errors = validate_sample(s)
check("Valid sample accepted", is_valid, f"errors: {errors}")
check("No errors returned", len(errors) == 0, f"errors: {errors}")
check("Optional defaults filled (severity_score)", "severity_score" in s)
check("Optional defaults filled (pair_id)", "pair_id" in s)
check("cfa_origin defaulted to 'native'", s.get("cfa_origin") == "native",
      f"got: {s.get('cfa_origin')}")


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 2: validate_sample() rejects bad samples")
print("=" * 64)

# 2a: missing cwe
s1 = make_valid_sample()
del s1["cwe"]
ok1, e1 = validate_sample(s1)
check("Missing CWE \u2192 rejected", not ok1, f"was accepted; errors={e1}")
check("  error mentions 'cwe'", any("cwe" in e.lower() for e in e1), f"errors={e1}")

# 2b: wrong language
s2 = make_valid_sample()
s2["language"] = "python"
ok2, e2 = validate_sample(s2)
check("language='python' \u2192 rejected", not ok2, f"was accepted; errors={e2}")
check("  error mentions 'language'", any("language" in e.lower() for e in e2), f"errors={e2}")

# 2c: invalid label
s3 = make_valid_sample()
s3["label"] = 5
ok3, e3 = validate_sample(s3)
check("label=5 \u2192 rejected", not ok3, f"was accepted; errors={e3}")
check("  error mentions 'label'", any("label" in e.lower() for e in e3), f"errors={e3}")

# 2d: multiple missing fields
s4 = {"id": "", "code": "x", "label": 99}
ok4, e4 = validate_sample(s4)
check("Multiple missing fields \u2192 rejected", not ok4)
check("  \u22653 errors reported", len(e4) >= 3, f"only {len(e4)} errors: {e4}")


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 3: save_checkpoint() + crash sim + load_checkpoint()")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story2_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        checkpoint_dir=os.path.join(tmpdir, "cp"),
        source_name="test_src",
    )

    state_out = {"page": 42, "last_id": "abc-123", "items": 100}
    c.save_checkpoint(state_out)

    cp_file = os.path.join(tmpdir, "cp", "test_src_checkpoint.json")
    check("Checkpoint file created", os.path.exists(cp_file))

    state_in = c.load_checkpoint()
    check("State round-trips correctly", state_in == state_out,
          f"got {state_in}")

    tmp_file = os.path.join(tmpdir, "cp", "test_src_checkpoint.tmp")
    check("No .tmp residue (atomic write)", not os.path.exists(tmp_file))

    # Simulate crash & restart: new collector reads old checkpoint
    c2 = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        checkpoint_dir=os.path.join(tmpdir, "cp"),
        source_name="test_src",
    )
    state_reloaded = c2.load_checkpoint()
    check("Checkpoint survives restart", state_reloaded == state_out,
          f"got {state_reloaded}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 4: save_sample() deduplication")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story2_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="dedup_test",
    )

    sample_a = make_valid_sample()
    sample_b = make_valid_sample()        # different id, same code

    saved_a = c.save_sample(sample_a)
    saved_b = c.save_sample(sample_b)

    check("First save \u2192 True", saved_a is True)
    check("Duplicate save \u2192 False", saved_b is False, "duplicate was saved")

    out_file = os.path.join(tmpdir, "out", "dedup_test_samples.jsonl")
    with open(out_file, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    check("JSONL has exactly 1 line", len(lines) == 1, f"found {len(lines)}")
    check("Stats: samples_saved == 1", c._samples_saved == 1,
          f"got {c._samples_saved}")

    # Different code IS saved
    sample_c = make_valid_sample()
    sample_c["code"] = VALID_CODE.replace("process_input", "handle_request")
    saved_c = c.save_sample(sample_c)
    check("Different code \u2192 saved", saved_c is True)
    check("Stats: samples_saved == 2", c._samples_saved == 2,
          f"got {c._samples_saved}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 5: safe_get() retries 5\u00d7 on ConnectionError")
print("=" * 64)

from unittest.mock import patch, MagicMock
import requests as req_lib

tmpdir = tempfile.mkdtemp(prefix="sg_story2_")
try:
    c = BaseCollector(
        output_dir=os.path.join(tmpdir, "out"),
        source_name="retry_test",
    )

    # 5a: 4 failures then success on 5th call
    counter = [0]  # mutable container avoids nonlocal
    def mock_get_succeed_5th(*args, **kwargs):
        counter[0] += 1
        if counter[0] < 5:
            raise req_lib.exceptions.ConnectionError("fail #" + str(counter[0]))
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        return resp

    with patch("requests.get", side_effect=mock_get_succeed_5th):
        resp = c.safe_get("http://fake.test/api")

    check("Retried until success (5 calls)", counter[0] == 5,
          "called " + str(counter[0]) + " times")
    check("Final response status 200", resp.status_code == 200)

    # 5b: all 5 attempts fail \u2192 exception raised
    counter2 = [0]
    def mock_get_always_fail(*args, **kwargs):
        counter2[0] += 1
        raise req_lib.exceptions.ConnectionError("permanent fail #" + str(counter2[0]))

    with patch("requests.get", side_effect=mock_get_always_fail):
        try:
            c.safe_get("http://fake.test/api")
            check("Should raise after exhausting retries", False, "no exception")
        except req_lib.exceptions.ConnectionError:
            check("Raises ConnectionError after 5 retries", True)
            check("Attempted exactly 5 times", counter2[0] == 5,
                  "called " + str(counter2[0]) + " times")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print(f"  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed"
      f" out of {PASS_COUNT + FAIL_COUNT}")
if FAIL_COUNT:
    print(f"\n  \u26a0\ufe0f  {FAIL_COUNT} test(s) FAILED")
else:
    print(f"\n  \U0001f389 All {PASS_COUNT} tests passed!")
print("=" * 64 + "\n")

sys.exit(FAIL_COUNT)
