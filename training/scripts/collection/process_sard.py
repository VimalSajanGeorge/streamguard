#!/usr/bin/env python3
"""
training/scripts/collection/process_sard.py

SARD Juliet Test Suite C collector for StreamGuard.
Processes the NIST Juliet Suite v1.3.1 C test cases from local disk.

The Juliet Suite structure is:
    data/raw/sard/2022-08-11-.../
        NNNNN-v1.0.0/src/testcases/CWExx_Description/.../*.c

Each .c file contains BOTH bad() and good() functions.
Label is determined by function name, NOT by filename.

Production issues addressed:
  1. No compile-check — files use custom headers from testcasesupport/
  2. Non-UTF-8 encodings — chardet fallback
  3. Recursive identifier extraction — handles nested declarators

Source: docs/New Docs/DATA_PIPELINE.md §Collector 1: SARD
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import hashlib
import uuid
from collections import Counter
from pathlib import Path

import chardet
from loguru import logger
from tree_sitter import Language, Parser

try:
    import tree_sitter_c as tsc
except ImportError:
    logger.error("tree_sitter_c not installed. Run: pip install tree-sitter-c")
    sys.exit(1)

try:
    from .base_collector import BaseCollector
    from .schema import make_sample_id, make_timestamp, validate_sample
except ImportError:
    from base_collector import BaseCollector
    from schema import make_sample_id, make_timestamp, validate_sample


# ── M1 Target CWEs ──────────────────────────────────────────────────────
# Maps Juliet directory prefix (e.g. "CWE121") to canonical CWE ID
CWE_DIR_TO_CANONICAL = {
    "CWE78":  "CWE-78",
    "CWE121": "CWE-121",
    "CWE122": "CWE-122",
    "CWE134": "CWE-134",
    "CWE190": "CWE-190",
    "CWE416": "CWE-416",
    "CWE476": "CWE-476",
}

# Functions to SKIP — these are Juliet harness/utility functions, not
# vulnerability test functions. We do NOT want them in training data.
SKIP_NAMES = {
    "main",
    "printLine", "printIntLine", "printLongLine",
    "printLongLongLine", "printFloatLine", "printDoubleLine",
    "printWLine", "printHexCharLine", "printHexUnsignedCharLine",
    "printUnsignedLine", "printSizeTLine",
    "globalReturnsTrueOrFalse", "globalReturnsTrue", "globalReturnsFalse",
}


class SARDCollector(BaseCollector):
    """
    Processes NIST SARD Juliet Suite C test cases.
    Inherits checkpoint/resume, dedup, and validation from BaseCollector.
    """

    def __init__(
        self,
        sard_root: str,
        output_dir: str,
        dry_run: bool = False,
        max_samples: int = 0,
    ):
        super().__init__(
            output_dir=output_dir,
            source_name="sard",
        )
        self.sard_root = Path(sard_root)
        self.dry_run = dry_run
        self.max_samples = max_samples

        # Tree-sitter C parser
        lang = Language(tsc.language())
        self.parser = Parser(lang)

        # Counters
        self.files_processed = 0
        self.files_skipped = 0
        self.functions_extracted = 0
        self.functions_classified = 0
        self.functions_skipped_helper = 0
        self.functions_skipped_unclassified = 0
        self.encoding_fallbacks = 0
        self.cwe_counts: Counter = Counter()
        self.label_counts: Counter = Counter()
        self.pair_count = 0

    # ── Discovery ────────────────────────────────────────────────────

    def _iter_testcases_roots(self):
        """
        Generator: yield testcases/ directories that contain CWE* subdirs.

        The actual Juliet Suite 116 structure on disk is:
            data/raw/sard/
            └── 2022-08-11-juliet-....(1)/
                ├── 100000-v1.0.0/src/testcases/CWE_X/   ← one CWE per numbered dir
                ├── 100001-v1.0.0/src/testcases/CWE_Y/
                └── ...  (64,100 numbered dirs total)

        Using a generator avoids loading all 64K paths into RAM before
        starting collection, and allows max_samples early-exit to work
        immediately without waiting for the full scan to finish.

        Raises FileNotFoundError if zero valid roots are found.
        """
        found_any = False
        count = 0
        for candidate in self.sard_root.rglob("testcases"):
            if not candidate.is_dir():
                continue
            # Fast check: does it have any CWE* child dir?
            has_cwe = any(
                d.is_dir() and d.name.startswith("CWE")
                for d in candidate.iterdir()
            )
            if not has_cwe:
                continue
            found_any = True
            count += 1
            if count % 5000 == 0:
                logger.info(f"Scanning... {count} testcases roots found so far")
            yield candidate

        if not found_any:
            raise FileNotFoundError(
                f"No testcases/ directory with CWE* subdirectories found under "
                f"{self.sard_root}. Expected structure: "
                "<root>/.../src/testcases/CWE*/"
            )
        logger.info(f"Finished scanning: {count} testcases roots processed")

    # ── CWE Extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_cwe(dirname: str) -> str | None:
        """
        Extract canonical CWE ID from a directory name.
        E.g. "CWE121_Stack_Based_Buffer_Overflow" → "CWE-121"
        Returns None if not in M1 target set.
        """
        m = re.match(r"CWE(\d+)", dirname)
        if not m:
            return None
        raw_cwe = f"CWE{m.group(1)}"
        return CWE_DIR_TO_CANONICAL.get(raw_cwe)

    # ── File Reading (encoding-safe) ─────────────────────────────────

    def _read_file_safe(self, filepath: Path) -> str | None:
        """
        Read a file with encoding detection.
        1. Try UTF-8 first (fast path for ~95% of files)
        2. Fall back to chardet detection
        3. Last resort: latin-1 (never fails)
        Returns decoded string or None on total failure.
        """
        raw = filepath.read_bytes()
        if not raw:
            return None

        # Fast path: try UTF-8
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Use chardet
            detection = chardet.detect(raw)
            encoding = detection.get("encoding") or "latin-1"
            try:
                self.encoding_fallbacks += 1
                text = raw.decode(encoding, errors="replace")
            except (UnicodeDecodeError, LookupError):
                # Final fallback — latin-1 never raises
                self.encoding_fallbacks += 1
                text = raw.decode("latin-1", errors="replace")

        # Normalize CRLF → LF
        return text.replace("\r\n", "\n")

    # ── Tree-sitter Function Extraction ──────────────────────────────

    @staticmethod
    def _find_identifier(node) -> object | None:
        """
        Recursively find the identifier node within a declarator.
        Handles nesting like:
            pointer_declarator → function_declarator → identifier
            function_declarator → identifier
        """
        if node is None:
            return None
        if node.type == "identifier":
            return node
        for child in node.children:
            result = SARDCollector._find_identifier(child)
            if result is not None:
                return result
        return None

    def _extract_functions(self, source_code: str) -> list[dict]:
        """
        Parse C source with tree-sitter and extract all function definitions.
        Returns list of {'name': str, 'code': str}.

        Juliet files wrap functions in #ifndef OMITBAD / #ifndef OMITGOOD
        preprocessor guards, so function_definition nodes are children of
        preproc_ifdef nodes, NOT direct children of the root. We walk the
        full AST recursively to find them at any depth.
        """
        tree = self.parser.parse(source_code.encode("utf-8", errors="replace"))
        functions = []

        def walk(node) -> None:
            if node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                name_node = self._find_identifier(declarator)
                if name_node is not None:
                    func_name = name_node.text.decode("utf-8", errors="replace")
                    func_code = source_code[node.start_byte:node.end_byte]
                    functions.append({"name": func_name, "code": func_code})
                return  # don't recurse into nested functions
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return functions

    # ── Classification ───────────────────────────────────────────────

    @staticmethod
    def _classify(func_name: str) -> int | None:
        """
        Classify a function by its name.
        Returns:
            1 if vulnerable (contains 'bad')
            0 if safe (contains 'good')
            None if helper/utility (skip)

        Juliet naming conventions:
            CWE122_..._04_bad          → vulnerable
            CWE122_..._04_good         → safe
            goodG2B1, goodG2B2, good1  → safe (helper variants)
        """
        if func_name in SKIP_NAMES:
            return None

        # Juliet uses _bad / _bad1 suffixes and good / goodG2B prefixes
        # Since _ is a word char, \b won't match after _ — use substring search
        lower = func_name.lower()
        if "bad" in lower:
            return 1
        if "good" in lower:
            return 0

        return None

    def _is_duplicate(self, sample: dict) -> bool:
        """Return True if this sample's code was already saved (MD5 dedup check).

        Uses the same hash set as BaseCollector.save_sample() so we can
        pre-check whether a sample will be deduplicated BEFORE assigning
        pair_id, avoiding broken pairs caused by duplicate good/bad functions.
        Does NOT add to _seen_hashes (save_sample does that).
        """
        code_hash = hashlib.md5(
            " ".join(sample["code"].split()).encode("utf-8")
        ).hexdigest()
        return code_hash in self._seen_hashes

    # ── Main Collection Loop ─────────────────────────────────────────

    def collect(self) -> dict:
        """
        Main collection method. Streams testcases roots, iterates CWE
        directories, extracts and classifies functions, saves samples.

        Uses a generator for discovery so max_samples early-exit works
        immediately — no need to scan all 64K dirs before starting.

        Returns summary dict.
        """
        if self.dry_run:
            logger.info("DRY RUN MODE — will process but not save samples")

        for testcases_root in self._iter_testcases_roots():
            for cwe_dir in testcases_root.iterdir():
                if not cwe_dir.is_dir():
                    continue

                cwe = self._extract_cwe(cwe_dir.name)
                if cwe is None:
                    continue  # not a target CWE

                # Process all .c files recursively in this CWE directory
                for filepath in sorted(cwe_dir.rglob("*.c")):
                    # Skip testcasesupport files
                    if "testcasesupport" in str(filepath):
                        continue

                    if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                        logger.info(
                            f"Reached max_samples limit ({self.max_samples}). Stopping."
                        )
                        return self._build_summary()

                    self._process_file(filepath, cwe)

                    # Checkpoint every 500 files
                    if self.files_processed % 500 == 0 and self.files_processed > 0:
                        self.save_checkpoint({
                            "files_processed": self.files_processed,
                            "samples_saved": self._samples_saved,
                        })
                        logger.info(
                            f"Progress: {self.files_processed} files, "
                            f"{self._samples_saved} samples saved"
                        )

        return self._build_summary()

    def _process_file(self, filepath: Path, cwe: str) -> None:
        """Process a single .c file: read, parse, classify, save."""
        source = self._read_file_safe(filepath)
        if source is None:
            self.files_skipped += 1
            return

        functions = self._extract_functions(source)
        self.functions_extracted += len(functions)

        if not functions:
            self.files_skipped += 1
            return

        self.files_processed += 1

        # Classify all functions
        bad_funcs = []
        good_funcs = []
        for func in functions:
            label = self._classify(func["name"])
            if label is None:
                if func["name"] in SKIP_NAMES:
                    self.functions_skipped_helper += 1
                else:
                    self.functions_skipped_unclassified += 1
                continue
            self.functions_classified += 1
            if label == 1:
                bad_funcs.append(func)
            else:
                good_funcs.append(func)

        # Pre-validate every classified function so we know which will actually
        # be saved before assigning pair_id. This prevents broken pairs where
        # the bad function passes schema but the good wrapper fails (e.g. thin
        # dispatcher functions like CWE_44_good() with <10 tokens).
        timestamp = make_timestamp()

        def make_sample(func: dict, label: int, pair_id: str) -> dict:
            return {
                "id":            make_sample_id(),
                "source":        "sard",
                "code":          func["code"],
                "label":         label,
                "cwe":           cwe,
                "language":      "c",
                "collected_at":  timestamp,
                "pair_id":       pair_id,
                "file_path":     str(filepath),
                "function_name": func["name"],
                "cfa_origin":    "native",
            }

        # Build samples without pair_id first, run schema check, and check
        # deduplication. We only assign a pair_id after confirming that both
        # a bad sample AND a good sample will actually be written (schema-valid
        # AND not a duplicate of an already-saved sample). This prevents broken
        # pairs caused by either schema rejection or MD5 deduplication.
        bad_samples = []   # (func, sample) tuples that are valid and not dupes
        good_samples = []  # same for good
        failed_funcs = []  # (func, errors) for schema-rejected functions

        for func in bad_funcs:
            s = make_sample(func, 1, "")
            is_valid, errors = validate_sample(s)
            if not is_valid:
                failed_funcs.append((func, errors))
                continue
            if self.dry_run or not self._is_duplicate(s):
                bad_samples.append((func, s))
            # If it IS a duplicate, it was saved in a previous file — skip,
            # don't count as a schema failure.

        for func in good_funcs:
            s = make_sample(func, 0, "")
            is_valid, errors = validate_sample(s)
            if not is_valid:
                failed_funcs.append((func, errors))
                continue
            if self.dry_run or not self._is_duplicate(s):
                good_samples.append((func, s))

        # Assign pair_id only when BOTH sides will actually be written
        has_pair = len(bad_samples) > 0 and len(good_samples) > 0
        pair_id = str(uuid.uuid4()) if has_pair else ""
        if has_pair:
            self.pair_count += 1

        # Patch pair_id and save
        for func, sample in bad_samples + good_samples:
            sample["pair_id"] = pair_id
            label = sample["label"]

            if self.dry_run:
                self._samples_saved += 1
                self.cwe_counts[cwe] += 1
                self.label_counts[label] += 1
                logger.debug(
                    f"[DRY] {func['name']} -> label={label} cwe={cwe}"
                )
            else:
                saved = self.save_sample(sample)
                if saved:
                    self.cwe_counts[cwe] += 1
                    self.label_counts[label] += 1

            # Check max_samples
            if self.max_samples > 0 and self._samples_saved >= self.max_samples:
                return

        # Schema-failed functions — save to failed.jsonl for debug
        for func, errors in failed_funcs:
            label = self._classify(func["name"])
            s = make_sample(func, label, "")
            self._samples_failed += 1
            if not self.dry_run:
                self.save_failed_item(s, reason=str(errors))
            logger.debug(f"[DRY] Rejected {func['name']}: {errors}"
                         if self.dry_run else
                         f"Rejected {func['name']}: {errors}")

    def _build_summary(self) -> dict:
        """Build summary statistics dict."""
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "functions_extracted": self.functions_extracted,
            "functions_classified": self.functions_classified,
            "functions_skipped_helper": self.functions_skipped_helper,
            "functions_skipped_unclassified": self.functions_skipped_unclassified,
            "encoding_fallbacks": self.encoding_fallbacks,
            "samples_saved": self._samples_saved,
            "samples_failed": self._samples_failed,
            "label_distribution": dict(self.label_counts),
            "cwe_distribution": dict(self.cwe_counts),
            "valid_pairs": self.pair_count,
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        summary = self._build_summary()

        print("\n" + "=" * 60)
        print("SARD COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Files processed:     {summary['files_processed']}")
        print(f"Files skipped:       {summary['files_skipped']}")
        print(f"Functions extracted:  {summary['functions_extracted']}")
        print(f"Functions classified: {summary['functions_classified']}")
        print(f"  Skipped (helper):  {summary['functions_skipped_helper']}")
        print(f"  Skipped (unknown): {summary['functions_skipped_unclassified']}")
        print(f"Encoding fallbacks:  {summary['encoding_fallbacks']}")
        print(f"Samples saved:       {summary['samples_saved']}")
        print(f"Samples failed:      {summary['samples_failed']}")
        print(f"Valid CFA pairs:     {summary['valid_pairs']}")
        print()

        print("Label Distribution:")
        for label, count in sorted(summary["label_distribution"].items()):
            label_str = "vulnerable" if label == 1 else "safe"
            print(f"  {label_str} (label={label}): {count}")

        total_labeled = sum(summary["label_distribution"].values())
        if total_labeled > 0:
            vuln_pct = summary["label_distribution"].get(1, 0) / total_labeled * 100
            safe_pct = summary["label_distribution"].get(0, 0) / total_labeled * 100
            print(f"  Ratio: {vuln_pct:.1f}% vuln / {safe_pct:.1f}% safe")
        print()

        print("CWE Distribution:")
        for cwe, count in sorted(
            summary["cwe_distribution"].items(), key=lambda x: -x[1]
        ):
            print(f"  {cwe}: {count}")
        print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process SARD Juliet Suite C test cases for StreamGuard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run with 20 samples
  python process_sard.py --sard-root data/raw/sard/ --output-dir training/data/raw/sard/ --dry-run --max-samples 20

  # Full collection
  python process_sard.py --sard-root data/raw/sard/ --output-dir training/data/raw/sard/
        """,
    )
    parser.add_argument(
        "--sard-root",
        required=True,
        help="Root directory containing SARD Juliet Suite data",
    )
    parser.add_argument(
        "--output-dir",
        default="training/data/raw/sard/",
        help="Output directory for JSONL samples (default: training/data/raw/sard/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing output files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after saving N samples (0=unlimited)",
    )

    args = parser.parse_args()

    logger.info("Starting SARD collector...")
    logger.info(f"  sard_root:   {args.sard_root}")
    logger.info(f"  output_dir:  {args.output_dir}")
    logger.info(f"  dry_run:     {args.dry_run}")
    logger.info(f"  max_samples: {args.max_samples}")

    collector = SARDCollector(
        sard_root=args.sard_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
    )

    try:
        collector.collect()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving checkpoint...")
        collector.save_checkpoint({
            "files_processed": collector.files_processed,
            "samples_saved": collector._samples_saved,
            "interrupted": True,
        })
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise
    finally:
        collector.print_summary()


if __name__ == "__main__":
    main()
