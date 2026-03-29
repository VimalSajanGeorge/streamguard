"""Tests for the new BaseCollector (DATA_PIPELINE.md spec)."""
import json
import hashlib
import sys
from pathlib import Path
import pytest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "scripts" / "collection"))

from base_collector import BaseCollector
from schema import make_sample_id, make_timestamp


def _make_collector(tmp_path):
    return BaseCollector(
        output_dir=str(tmp_path / "output"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        source_name="test_src",
    )


def _make_valid_sample():
    return {
        "id": make_sample_id(),
        "source": "sard",
        "code": "void foo(char *s) {\n    char buf[16];\n    strcpy(buf, s);\n    return;\n}\n",
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": make_timestamp(),
    }


# --- save_sample ---
def test_save_valid_sample(tmp_path):
    c = _make_collector(tmp_path)
    s = _make_valid_sample()
    assert c.save_sample(s) is True
    assert c._samples_saved == 1
    out = (tmp_path / "output" / "test_src_samples.jsonl").read_text()
    assert json.loads(out.strip())["cwe"] == "CWE-120"


def test_save_invalid_sample_rejected(tmp_path):
    c = _make_collector(tmp_path)
    s = _make_valid_sample()
    s["label"] = 99
    assert c.save_sample(s) is False
    assert c._samples_saved == 0


def test_dedup_identical_code(tmp_path):
    c = _make_collector(tmp_path)
    s1 = _make_valid_sample()
    s2 = _make_valid_sample()
    s2["id"] = make_sample_id()  # different id, same code
    s2["code"] = s1["code"]
    assert c.save_sample(s1) is True
    assert c.save_sample(s2) is False  # duplicate code
    assert c._samples_saved == 1


def test_dedup_whitespace_normalized(tmp_path):
    c = _make_collector(tmp_path)
    s1 = _make_valid_sample()
    s2 = _make_valid_sample()
    s2["id"] = make_sample_id()
    s2["code"] = s1["code"].replace("\n", "\n\n")  # extra blank lines — same content
    assert c.save_sample(s1) is True
    assert c.save_sample(s2) is False


def test_save_failed_item_written(tmp_path):
    c = _make_collector(tmp_path)
    s = _make_valid_sample()
    del s["cwe"]  # invalid
    c.save_sample(s)
    failed = (tmp_path / "output" / "test_src_failed.jsonl")
    assert failed.exists()


# --- checkpoint ---
def test_checkpoint_roundtrip(tmp_path):
    c = _make_collector(tmp_path)
    state = {"page": 42, "processed": ["a", "b"]}
    c.save_checkpoint(state)
    loaded = c.load_checkpoint()
    assert loaded == state


def test_checkpoint_atomic_on_crash(tmp_path):
    """Verify .tmp file is not left behind after successful save."""
    c = _make_collector(tmp_path)
    c.save_checkpoint({"x": 1})
    tmp_files = list((tmp_path / "checkpoints").glob("*.tmp"))
    assert tmp_files == []


def test_checkpoint_empty_on_missing(tmp_path):
    c = _make_collector(tmp_path)
    assert c.load_checkpoint() == {}


# --- _load_seen_hashes (resume) ---
def test_hashes_loaded_on_resume(tmp_path):
    c1 = _make_collector(tmp_path)
    s = _make_valid_sample()
    c1.save_sample(s)
    # Simulate restart
    c2 = _make_collector(tmp_path)
    assert len(c2._seen_hashes) == 1
    # Duplicate rejected on new instance
    assert c2.save_sample(s) is False


# --- stats ---
def test_stats_returns_dict(tmp_path):
    c = _make_collector(tmp_path)
    st = c.stats
    assert st["source"] == "test_src"
    assert "samples_saved" in st
