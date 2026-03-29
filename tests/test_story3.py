"""
Story 3 Verification: process_sard.py — SARD Juliet Suite Collector
Tests the SARDCollector against synthetic Juliet-style test cases.
"""
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from collections import Counter

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.scripts.collection.process_sard import (
    SARDCollector,
    CWE_DIR_TO_CANONICAL,
    SKIP_NAMES,
)
from training.scripts.collection.schema import validate_sample, make_sample_id, make_timestamp

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}: {detail}")


# ── Helper: Create a synthetic Juliet directory structure ──────────
# Mirrors the actual structure:
#   sard_root/
#   └── 2022-08-11-juliet-.../
#       └── 61945-v1.0.0/
#           └── src/
#               └── testcases/
#                   ├── CWE121_Stack_Based_Buffer_Overflow/
#                   │   └── test_01.c
#                   ├── CWE476_NULL_Pointer_Dereference/
#                   │   └── test_01.c
#                   └── testcasesupport/   ← should be skipped
#                       └── std_testcase.c

# A Juliet-style .c file with BOTH bad() and good() functions
JULIET_FILE_BOTH = """\
#include "std_testcase.h"
#include <string.h>

/* bad: stack buffer overflow via unchecked memcpy */
void CWE121_Stack_Overflow__char_type_overrun_memcpy_01_bad()
{
    charFirst first;
    charSecond second;
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    /* FLAW: overrun into second struct */
    memcpy(first.charFirst_buf, SRC_STR, sizeof(SRC_STR));
    printLine(first.charFirst_buf);
    printLine(second.charSecond_buf);
}

/* good: bounds-checked copy */
void CWE121_Stack_Overflow__char_type_overrun_memcpy_01_good()
{
    charFirst first;
    charSecond second;
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    /* FIX: use sizeof destination */
    memcpy(first.charFirst_buf, SRC_STR, sizeof(first.charFirst_buf) - 1);
    first.charFirst_buf[sizeof(first.charFirst_buf) - 1] = '\\0';
    printLine(first.charFirst_buf);
    printLine(second.charSecond_buf);
}
"""

# A file with ONLY a bad function (no pair)
JULIET_FILE_BAD_ONLY = """\
#include "std_testcase.h"
#include <stdlib.h>

void CWE476_NULL_Pointer_Deref__int_01_bad()
{
    int *data = NULL;
    /* FLAW: dereference NULL pointer */
    printIntLine(*data);
    /* This is intentionally vulnerable */
    int result = *data + 1;
    printIntLine(result);
}
"""

# A file with ONLY good functions (no pair)
JULIET_FILE_GOOD_ONLY = """\
#include "std_testcase.h"
#include <stdlib.h>

void CWE476_NULL_Pointer_Deref__int_02_good1()
{
    int *data = NULL;
    data = (int *)malloc(sizeof(int));
    if (data != NULL) {
        *data = 5;
        printIntLine(*data);
        free(data);
    }
}

void CWE476_NULL_Pointer_Deref__int_02_goodG2B()
{
    int *data = NULL;
    data = (int *)malloc(sizeof(int));
    if (data != NULL) {
        *data = 10;
        printIntLine(*data);
        free(data);
    }
}
"""

# A file with helper functions only (should all be skipped)
JULIET_FILE_HELPERS_ONLY = """\
#include <stdio.h>

void printLine(const char *line)
{
    if (line != NULL)
    {
        printf("%s\\n", line);
    }
}

int main(int argc, char *argv[])
{
    CWE121_test_bad();
    CWE121_test_good();
    return 0;
}
"""

# testcasesupport file (should be skipped entirely)
TESTCASESUPPORT_FILE = """\
#include <stdio.h>

void printLine(const char *line)
{
    if (line != NULL)
    {
        printf("%s\\n", line);
    }
}
"""

# A file with a short function (< 5 non-blank lines — should be rejected by schema)
JULIET_FILE_SHORT_FUNC = """\
#include "std_testcase.h"

void CWE190_Integer_Overflow__int_01_bad()
{
    int x = INT_MAX + 1;
}
"""

# CWE not in target set (CWE999) — should be skipped
JULIET_FILE_NONTARGET = """\
#include "std_testcase.h"

void CWE999_Fake_Vuln__01_bad()
{
    char buf[10];
    memset(buf, 0, sizeof(buf));
    strcpy(buf, "hello world overflow");
    printf("%s\\n", buf);
    return;
}
"""

# File with pointer_declarator nesting (tests recursive find_identifier)
JULIET_FILE_POINTER_DECL = """\
#include "std_testcase.h"
#include <string.h>

void *CWE122_Heap_Overflow__char_cpy_01_bad()
{
    char *data = NULL;
    data = (char *)malloc(10 * sizeof(char));
    /* FLAW: copy more than allocated */
    strcpy(data, "AAAAAAAAAAAAAAAAAAAAAAAA");
    printLine(data);
    free(data);
    return NULL;
}

void CWE122_Heap_Overflow__char_cpy_01_good()
{
    char *data = NULL;
    data = (char *)malloc(100 * sizeof(char));
    /* FIX: allocate enough space */
    strcpy(data, "AAAAAAAAAAAAAAAAAAAAAAAA");
    printLine(data);
    free(data);
}
"""


def create_juliet_tree(tmpdir: str) -> str:
    """
    Create a synthetic Juliet directory structure.
    Returns the sard_root path (what user passes as --sard-root).
    """
    sard_root = Path(tmpdir) / "data" / "raw" / "sard"

    # Mimic actual deep nesting
    testcases = (
        sard_root
        / "2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support (1)"
        / "61945-v1.0.0"
        / "src"
        / "testcases"
    )

    # CWE121 directory
    cwe121 = testcases / "CWE121_Stack_Based_Buffer_Overflow"
    cwe121.mkdir(parents=True, exist_ok=True)
    (cwe121 / "CWE121_Stack_Overflow__char_type_overrun_memcpy_01.c").write_text(
        JULIET_FILE_BOTH, encoding="utf-8"
    )

    # CWE476 directory with multiple files
    cwe476 = testcases / "CWE476_NULL_Pointer_Dereference"
    cwe476.mkdir(parents=True, exist_ok=True)
    (cwe476 / "CWE476_NULL_Pointer_Deref__int_01.c").write_text(
        JULIET_FILE_BAD_ONLY, encoding="utf-8"
    )
    (cwe476 / "CWE476_NULL_Pointer_Deref__int_02.c").write_text(
        JULIET_FILE_GOOD_ONLY, encoding="utf-8"
    )

    # CWE122 directory (pointer declarator test)
    cwe122 = testcases / "CWE122_Heap_Based_Buffer_Overflow"
    cwe122.mkdir(parents=True, exist_ok=True)
    (cwe122 / "CWE122_Heap_Overflow__char_cpy_01.c").write_text(
        JULIET_FILE_POINTER_DECL, encoding="utf-8"
    )

    # CWE190 directory (short function — should be rejected)
    cwe190 = testcases / "CWE190_Integer_Overflow"
    cwe190.mkdir(parents=True, exist_ok=True)
    (cwe190 / "CWE190_Integer_Overflow__int_01.c").write_text(
        JULIET_FILE_SHORT_FUNC, encoding="utf-8"
    )

    # testcasesupport directory (should be skipped)
    support = testcases / "testcasesupport"
    support.mkdir(parents=True, exist_ok=True)
    (support / "std_testcase.c").write_text(
        TESTCASESUPPORT_FILE, encoding="utf-8"
    )

    # Non-target CWE directory (should be skipped)
    cwe999 = testcases / "CWE999_Fake_Vulnerability"
    cwe999.mkdir(parents=True, exist_ok=True)
    (cwe999 / "CWE999_Fake_Vuln__01.c").write_text(
        JULIET_FILE_NONTARGET, encoding="utf-8"
    )

    # File with only helpers (should produce no samples)
    (cwe121 / "testcasesupport_helpers.c").write_text(
        JULIET_FILE_HELPERS_ONLY, encoding="utf-8"
    )

    return str(sard_root)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 1: _iter_testcases_roots() auto-discovers structure")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = create_juliet_tree(tmpdir)
    outdir = os.path.join(tmpdir, "output")

    collector = SARDCollector(
        sard_root=sard_root,
        output_dir=outdir,
    )

    roots = list(collector._iter_testcases_roots())
    check("Found exactly 1 testcases root", len(roots) == 1, f"found {len(roots)}")
    check(
        "Root path ends with 'testcases'",
        roots[0].name == "testcases",
        f"got: {roots[0]}",
    )
    # Verify CWE dirs are children
    children = [d.name for d in roots[0].iterdir() if d.is_dir()]
    cwe_children = [c for c in children if c.startswith("CWE")]
    check(
        "testcases root has CWE* children",
        len(cwe_children) >= 3,
        f"found: {cwe_children}",
    )
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 2: _extract_cwe() maps directory names correctly")
print("=" * 64)

check(
    "CWE121_Stack_Based_Buffer_Overflow -> CWE-121",
    SARDCollector._extract_cwe("CWE121_Stack_Based_Buffer_Overflow") == "CWE-121",
)
check(
    "CWE78_OS_Command_Injection -> CWE-78",
    SARDCollector._extract_cwe("CWE78_OS_Command_Injection") == "CWE-78",
)
check(
    "CWE476_NULL_Pointer_Dereference -> CWE-476",
    SARDCollector._extract_cwe("CWE476_NULL_Pointer_Dereference") == "CWE-476",
)
check(
    "CWE999_Not_Target -> None",
    SARDCollector._extract_cwe("CWE999_Not_Target") is None,
)
check(
    "not_a_cwe_dir -> None",
    SARDCollector._extract_cwe("not_a_cwe_dir") is None,
)
check(
    "All 7 target CWEs mapped",
    len(CWE_DIR_TO_CANONICAL) == 7,
    f"got {len(CWE_DIR_TO_CANONICAL)}",
)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 3: _classify() labels functions correctly")
print("=" * 64)

check(
    "CWE121_..._01_bad -> 1 (vulnerable)",
    SARDCollector._classify("CWE121_Stack_Overflow__char_type_overrun_memcpy_01_bad") == 1,
)
check(
    "CWE121_..._01_good -> 0 (safe)",
    SARDCollector._classify("CWE121_Stack_Overflow__char_type_overrun_memcpy_01_good") == 0,
)
check(
    "goodG2B1 -> 0 (safe variant)",
    SARDCollector._classify("goodG2B1") == 0,
)
check(
    "goodG2B -> 0 (safe variant)",
    SARDCollector._classify("goodG2B") == 0,
)
check(
    "bad -> 1 (bare bad)",
    SARDCollector._classify("bad") == 1,
)
check(
    "main -> None (skip)",
    SARDCollector._classify("main") is None,
)
check(
    "printLine -> None (skip)",
    SARDCollector._classify("printLine") is None,
)
check(
    "printIntLine -> None (skip)",
    SARDCollector._classify("printIntLine") is None,
)
check(
    "globalReturnsTrueOrFalse -> None (skip)",
    SARDCollector._classify("globalReturnsTrueOrFalse") is None,
)
check(
    "helperFunc (no bad/good) -> None (unclassified)",
    SARDCollector._classify("helperFunc") is None,
)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 4: Full collection on synthetic Juliet tree")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = create_juliet_tree(tmpdir)
    outdir = os.path.join(tmpdir, "output")

    collector = SARDCollector(
        sard_root=sard_root,
        output_dir=outdir,
    )

    summary = collector.collect()

    # Load saved samples
    jsonl_path = os.path.join(outdir, "sard_samples.jsonl")
    check("Output JSONL file created", os.path.exists(jsonl_path))

    samples = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    check(
        "Samples saved > 0",
        len(samples) > 0,
        f"got {len(samples)}",
    )

    # Verify all samples pass schema validation
    all_valid = True
    for s in samples:
        is_valid, errors = validate_sample(s)
        if not is_valid:
            all_valid = False
            print(f"    Invalid sample: {s.get('function_name')}: {errors}")
    check("All samples pass validate_sample()", all_valid)

    # Check labels
    labels = Counter(s["label"] for s in samples)
    check("Has vulnerable samples (label=1)", labels.get(1, 0) > 0, f"labels: {dict(labels)}")
    check("Has safe samples (label=0)", labels.get(0, 0) > 0, f"labels: {dict(labels)}")

    # Check CWEs present
    cwes = set(s["cwe"] for s in samples)
    check("CWE-121 present", "CWE-121" in cwes, f"cwes: {cwes}")
    check("CWE-476 present", "CWE-476" in cwes, f"cwes: {cwes}")
    check("CWE-122 present", "CWE-122" in cwes, f"cwes: {cwes}")

    # Check that CWE-999 (non-target) is NOT present
    check("CWE-999 NOT present (non-target filtered)", "CWE-999" not in cwes, f"cwes: {cwes}")

    # All samples have source='sard'
    sources = set(s["source"] for s in samples)
    check("All samples have source='sard'", sources == {"sard"}, f"sources: {sources}")

    # All samples have language='c'
    langs = set(s["language"] for s in samples)
    check("All samples have language='c'", langs == {"c"}, f"langs: {langs}")

    # All samples have cfa_origin='native'
    origins = set(s["cfa_origin"] for s in samples)
    check(
        "All samples have cfa_origin='native'",
        origins == {"native"},
        f"origins: {origins}",
    )

    # Check pair_id integrity
    pairs = {}
    for s in samples:
        if s["pair_id"]:
            pairs.setdefault(s["pair_id"], []).append(s["label"])
    valid_pairs = {k: v for k, v in pairs.items() if 0 in v and 1 in v}
    broken_pairs = {k: v for k, v in pairs.items() if 0 not in v or 1 not in v}
    check(
        "Valid CFA pairs exist (files with bad+good)",
        len(valid_pairs) > 0,
        f"valid: {len(valid_pairs)}, broken: {len(broken_pairs)}",
    )
    check(
        "No broken pairs (0 missing bad or good)",
        len(broken_pairs) == 0,
        f"broken: {len(broken_pairs)}, details: {broken_pairs}",
    )

    # Check that unpaired functions have pair_id=''
    unpaired = [s for s in samples if s["pair_id"] == ""]
    check(
        "Unpaired functions have pair_id=''",
        len(unpaired) > 0,
        "expected some unpaired functions from bad-only/good-only files",
    )

    # Verify summary dict matches
    check(
        "Summary samples_saved matches JSONL count",
        summary["samples_saved"] == len(samples),
        f"summary: {summary['samples_saved']}, jsonl: {len(samples)}",
    )

    # Check testcasesupport files were skipped
    file_paths = [s["file_path"] for s in samples]
    has_support = any("testcasesupport" in fp for fp in file_paths)
    check("testcasesupport/ files skipped", not has_support)

    # Verify function_name field is populated
    all_have_name = all(s.get("function_name") for s in samples)
    check("All samples have function_name", all_have_name)

    # Check the short function was rejected (< 5 non-blank lines)
    func_names = [s["function_name"] for s in samples]
    check(
        "Short function (CWE190) rejected by schema",
        "CWE190_Integer_Overflow__int_01_bad" not in func_names,
        f"found in samples: {[n for n in func_names if 'CWE190' in n]}",
    )

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 5: --dry-run mode validates without writing")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = create_juliet_tree(tmpdir)
    outdir = os.path.join(tmpdir, "output_dryrun")

    collector = SARDCollector(
        sard_root=sard_root,
        output_dir=outdir,
        dry_run=True,
    )

    summary = collector.collect()

    # Dry-run should still count samples
    check(
        "Dry-run counts samples_saved > 0",
        summary["samples_saved"] > 0,
        f"got {summary['samples_saved']}",
    )

    # But should NOT write JSONL file
    jsonl_path = os.path.join(outdir, "sard_samples.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        check(
            "Dry-run did not write samples to JSONL",
            len(lines) == 0,
            f"found {len(lines)} lines",
        )
    else:
        check("Dry-run did not create JSONL file", True)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 6: --max-samples limits output")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = create_juliet_tree(tmpdir)
    outdir = os.path.join(tmpdir, "output_max")

    collector = SARDCollector(
        sard_root=sard_root,
        output_dir=outdir,
        max_samples=3,
    )

    summary = collector.collect()

    check(
        "max_samples=3 limits to <= 3 samples",
        summary["samples_saved"] <= 3,
        f"got {summary['samples_saved']}",
    )
    check(
        "At least 1 sample saved",
        summary["samples_saved"] >= 1,
        f"got {summary['samples_saved']}",
    )

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 7: Encoding fallback (non-UTF-8 file)")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = Path(tmpdir) / "sard_enc"
    testcases = sard_root / "61945-v1.0.0" / "src" / "testcases"
    cwe134 = testcases / "CWE134_Uncontrolled_Format_String"
    cwe134.mkdir(parents=True, exist_ok=True)

    # Write a file with Windows-1252 encoding (contains \x93 = left double quote)
    c_code = (
        b'#include "std_testcase.h"\n'
        b"#include <string.h>\n\n"
        b"void CWE134_Format_String__char_01_bad()\n"
        b"{\n"
        b"    char buf[256];\n"
        b"    /* Comment with \x93smart quotes\x94 */\n"
        b"    snprintf(buf, sizeof(buf), user_input);\n"
        b"    printf(buf);\n"
        b"    printLine(buf);\n"
        b"    return;\n"
        b"}\n\n"
        b"void CWE134_Format_String__char_01_good()\n"
        b"{\n"
        b"    char buf[256];\n"
        b"    /* FIX: use format specifier */\n"
        b"    snprintf(buf, sizeof(buf), \"%s\", user_input);\n"
        b"    printf(\"%s\", buf);\n"
        b"    printLine(buf);\n"
        b"    return;\n"
        b"}\n"
    )
    (cwe134 / "CWE134_Format_String__char_01.c").write_bytes(c_code)

    outdir = os.path.join(tmpdir, "output_enc")
    collector = SARDCollector(
        sard_root=str(sard_root),
        output_dir=outdir,
    )

    summary = collector.collect()

    check(
        "Non-UTF-8 file processed successfully",
        summary["samples_saved"] > 0,
        f"saved: {summary['samples_saved']}",
    )
    check(
        "Encoding fallback triggered",
        collector.encoding_fallbacks > 0,
        f"fallbacks: {collector.encoding_fallbacks}",
    )

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 8: Checkpoint saves during collection")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = create_juliet_tree(tmpdir)
    outdir = os.path.join(tmpdir, "output_cp")

    collector = SARDCollector(
        sard_root=sard_root,
        output_dir=outdir,
    )

    # Collection on small synthetic data won't trigger the 100-file checkpoint,
    # but we can verify the checkpoint mechanism works by manually saving
    collector.save_checkpoint({
        "files_processed": 42,
        "samples_saved": 100,
        "last_cwe_dir": "CWE121",
    })

    loaded = collector.load_checkpoint()
    check(
        "Checkpoint round-trips correctly",
        loaded["files_processed"] == 42 and loaded["samples_saved"] == 100,
        f"got: {loaded}",
    )

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 9: _iter_testcases_roots() fails on invalid path")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    empty_root = os.path.join(tmpdir, "empty_sard")
    os.makedirs(empty_root, exist_ok=True)
    outdir = os.path.join(tmpdir, "output_err")

    collector = SARDCollector(
        sard_root=empty_root,
        output_dir=outdir,
    )

    try:
        list(collector._iter_testcases_roots())
        check("Should raise FileNotFoundError on empty root", False, "no exception raised")
    except FileNotFoundError:
        check("Raises FileNotFoundError on empty root", True)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  TEST 10: Deduplication across identical functions")
print("=" * 64)

tmpdir = tempfile.mkdtemp(prefix="sg_story3_")
try:
    sard_root = Path(tmpdir) / "sard_dedup"
    testcases = sard_root / "61945-v1.0.0" / "src" / "testcases"
    cwe121 = testcases / "CWE121_Stack_Based_Buffer_Overflow"
    cwe121.mkdir(parents=True, exist_ok=True)

    # Write the SAME file content to two different filenames
    (cwe121 / "test_01.c").write_text(JULIET_FILE_BOTH, encoding="utf-8")
    (cwe121 / "test_02.c").write_text(JULIET_FILE_BOTH, encoding="utf-8")

    outdir = os.path.join(tmpdir, "output_dedup")
    collector = SARDCollector(
        sard_root=str(sard_root),
        output_dir=outdir,
    )

    summary = collector.collect()

    # Should only save 2 unique functions (1 bad + 1 good), not 4
    check(
        "Dedup: identical functions saved only once",
        summary["samples_saved"] == 2,
        f"expected 2, got {summary['samples_saved']}",
    )

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print(f"  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed"
      f" out of {PASS_COUNT + FAIL_COUNT}")
if FAIL_COUNT:
    print(f"\n  WARNING: {FAIL_COUNT} test(s) FAILED")
else:
    print(f"\n  All {PASS_COUNT} tests passed!")
print("=" * 64 + "\n")

sys.exit(FAIL_COUNT)
