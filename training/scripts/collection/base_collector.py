"""
training/scripts/collection/base_collector.py
Base collector class for all StreamGuard data collectors.
Source: docs/New Docs/DATA_PIPELINE.md §Base Collector Class
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

try:
    from .schema import validate_sample
except ImportError:
    from schema import validate_sample

_PAIR_ID_SENTINELS = {"", "None", "null", "0"}


class BaseCollector:
    """
    Base class for all StreamGuard data collectors.
    Provides: checkpoint/resume, HTTP retry, deduplication, validation.
    Every collector MUST inherit from this.
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_dir: str | None = None,
        source_name: str = "unknown",
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        )
        self.source_name = source_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._seen_hashes: set[str] = set()
        self._samples_saved: int = 0
        self._samples_failed: int = 0
        self._pending_pairs: dict = {}

        self._load_seen_hashes()

    # ── HTTP ─────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(min=0.5, max=3.0),
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ),
        reraise=True,
    )
    def safe_get(self, url: str, **kwargs) -> requests.Response:
        """GET with automatic retry on connection errors. Handles 429/403/503."""
        kwargs.setdefault("timeout", 45)
        resp = requests.get(url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning(f"[{self.source_name}] Rate limited on {url}. Sleeping {retry_after}s")
            time.sleep(retry_after + 1)
            resp = requests.get(url, **kwargs)
        if resp.status_code == 403 and "secondary rate limit" in resp.text.lower():
            sleep_secs = 120 + random.uniform(0, 30)
            logger.warning(f"[{self.source_name}] 403 abuse detection on {url}. Sleeping {sleep_secs:.0f}s")
            time.sleep(sleep_secs)
            resp = requests.get(url, **kwargs)
        if resp.status_code == 503:
            sleep_secs = 30 + random.uniform(0, 10)
            logger.warning(f"[{self.source_name}] 503 on {url}. Sleeping {sleep_secs:.0f}s")
            time.sleep(sleep_secs)
            resp = requests.get(url, **kwargs)
        if resp.status_code == 401:
            raise ValueError(f"401 Unauthorized — check API key for {url}")
        return resp

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(min=0.5, max=3.0),
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ),
        reraise=True,
    )
    def safe_post(self, url: str, **kwargs) -> requests.Response:
        """POST with automatic retry on connection errors. Handles 429/503."""
        kwargs.setdefault("timeout", 45)
        resp = requests.post(url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning(f"[{self.source_name}] Rate limited on {url}. Sleeping {retry_after}s")
            time.sleep(retry_after + 1)
            resp = requests.post(url, **kwargs)
        if resp.status_code == 503:
            sleep_secs = 30 + random.uniform(0, 10)
            logger.warning(f"[{self.source_name}] 503 on {url}. Sleeping {sleep_secs:.0f}s")
            time.sleep(sleep_secs)
            resp = requests.post(url, **kwargs)
        if resp.status_code == 401:
            raise ValueError(f"401 Unauthorized — check API key for {url}")
        return resp

    def jitter_sleep(self, base: float = 0.5) -> None:
        """Sleep for base + random(0, base) seconds to jitter request timing."""
        time.sleep(base + random.uniform(0, base))

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def load_checkpoint(self) -> dict:
        """Load checkpoint state dict. Returns {} if no checkpoint exists."""
        cp_path = self.checkpoint_dir / f"{self.source_name}_checkpoint.json"
        if cp_path.exists():
            try:
                return json.loads(cp_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"[{self.source_name}] Corrupt checkpoint, starting fresh")
                return {}
        return {}

    def save_checkpoint(self, state: dict) -> None:
        """Atomic checkpoint write: write to .tmp then os.replace()."""
        cp_path = self.checkpoint_dir / f"{self.source_name}_checkpoint.json"
        tmp_path = cp_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        # Retry os.replace on Windows — Defender/indexer can briefly lock the file
        for attempt in range(5):
            try:
                os.replace(tmp_path, cp_path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    # ── Sample saving ─────────────────────────────────────────────────────────

    def save_sample(self, sample: dict) -> bool:
        """
        Validate, deduplicate, check pairs, and save a sample.
        Returns True if saved, False if rejected (invalid or duplicate).
        Note: validate_sample() mutates sample in-place to fill optional defaults.
        """
        is_valid, errors = validate_sample(sample)
        if not is_valid:
            logger.debug(f"[{self.source_name}] Rejected sample {sample.get('id', '?')}: {errors}")
            self.save_failed_item(sample, reason=str(errors))
            self._samples_failed += 1
            return False

        # MD5 of whitespace-normalised code for dedup
        code_hash = hashlib.md5(
            " ".join(sample["code"].split()).encode("utf-8")
        ).hexdigest()
        if code_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(code_hash)

        # Persist hash to sidecar file for fast reload
        hash_path = self.output_dir / f"{self.source_name}_hashes.txt"
        with open(hash_path, "a", encoding="utf-8") as hf:
            hf.write(code_hash + "\n")

        # Inline broken-pair detection
        pair_id = str(sample.get("pair_id", ""))
        if pair_id not in _PAIR_ID_SENTINELS:
            if pair_id in self._pending_pairs:
                existing_label = self._pending_pairs[pair_id]
                if existing_label == sample.get("label"):
                    logger.warning(
                        f"[{self.source_name}] Duplicate label {sample['label']} "
                        f"for pair_id {pair_id} — clearing pair_id"
                    )
                    sample["pair_id"] = ""
                else:
                    # Valid pair completed
                    del self._pending_pairs[pair_id]
            else:
                self._pending_pairs[pair_id] = sample.get("label")

        out_path = self.output_dir / f"{self.source_name}_samples.jsonl"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        self._samples_saved += 1
        if self._samples_saved % 100 == 0:
            logger.info(f"[{self.source_name}] Saved {self._samples_saved} samples")
            self._check_disk_space()
        return True

    def save_failed_item(self, item: dict, reason: str) -> None:
        """Save rejected item with reason to *_failed.jsonl for debugging."""
        failed_path = self.output_dir / f"{self.source_name}_failed.jsonl"
        with open(failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"item": item, "reason": reason}, ensure_ascii=False) + "\n")

    def finalize_pairs(self) -> int:
        """
        Post-collection pass: clear pair_id on orphan samples (pair_id with count==1).
        Returns count of cleared orphans.
        """
        out_path = self.output_dir / f"{self.source_name}_samples.jsonl"
        if not out_path.exists():
            return 0

        # Pass 1: count pair_id occurrences
        pair_counts: dict[str, int] = {}
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    pid = str(s.get("pair_id", ""))
                    if pid not in _PAIR_ID_SENTINELS:
                        pair_counts[pid] = pair_counts.get(pid, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass

        orphan_pids = {pid for pid, count in pair_counts.items() if count == 1}
        if not orphan_pids:
            return 0

        # Pass 2: rewrite atomically, clearing orphan pair_ids
        tmp_path = out_path.with_suffix(".tmp")
        cleared = 0
        with open(out_path, encoding="utf-8") as fin, \
             open(tmp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                stripped = line.strip()
                if not stripped:
                    fout.write(line)
                    continue
                try:
                    s = json.loads(stripped)
                    pid = str(s.get("pair_id", ""))
                    if pid in orphan_pids:
                        s["pair_id"] = ""
                        cleared += 1
                    fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    fout.write(line)

        os.replace(tmp_path, out_path)
        logger.info(f"[{self.source_name}] finalize_pairs: cleared {cleared} orphan pair_ids")
        return cleared

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load_seen_hashes(self) -> None:
        """
        Pre-load MD5 hashes of all previously saved samples on init.
        Fast path: read from {source}_hashes.txt sidecar.
        Slow path: rebuild from JSONL if hash file missing.
        """
        hash_path = self.output_dir / f"{self.source_name}_hashes.txt"

        # Fast path: read hash sidecar
        if hash_path.exists():
            count = 0
            with open(hash_path, encoding="utf-8") as f:
                for line in f:
                    h = line.strip()
                    if h:
                        self._seen_hashes.add(h)
                        count += 1
            if count:
                logger.info(f"[{self.source_name}] Fast-loaded {count} hashes from sidecar")
            return

        # Slow path: rebuild from JSONL
        out_path = self.output_dir / f"{self.source_name}_samples.jsonl"
        if not out_path.exists():
            return
        count = 0
        skipped = 0
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    code = s.get("code", "")
                    h = hashlib.md5(" ".join(code.split()).encode("utf-8")).hexdigest()
                    self._seen_hashes.add(h)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
        if skipped:
            logger.warning(f"[{self.source_name}] Skipped {skipped} malformed lines during hash rebuild")
        if count:
            logger.info(f"[{self.source_name}] Rebuilt {count} hashes from JSONL (slow path)")
            # Write sidecar for future fast loads
            with open(hash_path, "w", encoding="utf-8") as hf:
                for h in self._seen_hashes:
                    hf.write(h + "\n")

    def _check_disk_space(self) -> None:
        """Raise if < 5GB free on the output drive or data dir."""
        min_gb = 5.0
        paths_to_check = [self.output_dir]

        data_dir = Path(os.environ.get("STREAMGUARD_DATA_DIR", "./training/data"))
        if data_dir.exists():
            paths_to_check.append(data_dir)

        for p in paths_to_check:
            free_gb = shutil.disk_usage(p).free / (1024 ** 3)
            if free_gb < min_gb:
                raise RuntimeError(
                    f"[{self.source_name}] Disk space critical: {free_gb:.1f} GB remaining on {p}"
                )

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "source": self.source_name,
            "samples_saved": self._samples_saved,
            "samples_failed": self._samples_failed,
            "unique_hashes": len(self._seen_hashes),
        }
