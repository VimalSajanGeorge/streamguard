"""Tests for the new StreamGuard canonical schema (DATA_PIPELINE.md spec)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "scripts" / "collection"))

from schema import (
    VALID_CWE, VALID_SOURCES,
    validate_sample, make_sample_id, make_timestamp,
    OPTIONAL_FIELD_DEFAULTS,
)


def _make_valid() -> dict:
    """Minimal valid sample for CWE-120."""
    return {
        "id": make_sample_id(),
        "source": "sard",
        "code": "void foo(char *s) {\n    char buf[16];\n    strcpy(buf, s);\n    return;\n}\n",
        "label": 1,
        "cwe": "CWE-120",
        "language": "c",
        "collected_at": make_timestamp(),
    }


# --- validate_sample: acceptance ---
def test_valid_sample_passes():
    ok, errors = validate_sample(_make_valid())
    assert ok, errors


def test_optional_defaults_filled_on_valid():
    s = _make_valid()
    validate_sample(s)
    assert s["severity_score"] == 0.0
    assert s["commit_sha"] == ""
    assert s["cfa_origin"] == "native"
    assert s["pair_id"] == ""
    assert s["aliases"] == {}


def test_label_zero_passes():
    s = _make_valid()
    s["label"] = 0
    ok, _ = validate_sample(s)
    assert ok


# --- validate_sample: rejections ---
def test_missing_id_fails():
    s = _make_valid()
    del s["id"]
    ok, errors = validate_sample(s)
    assert not ok and any("missing id" in e for e in errors)


def test_missing_code_fails():
    s = _make_valid()
    del s["code"]
    ok, errors = validate_sample(s)
    assert not ok


def test_invalid_label_fails():
    s = _make_valid()
    s["label"] = 2
    ok, errors = validate_sample(s)
    assert not ok and any("label" in e for e in errors)


def test_invalid_source_fails():
    s = _make_valid()
    s["source"] = "unknown_source"
    ok, errors = validate_sample(s)
    assert not ok and any("invalid source" in e for e in errors)


def test_invalid_cwe_fails():
    s = _make_valid()
    s["cwe"] = "CWE-999"
    ok, errors = validate_sample(s)
    assert not ok and any("invalid CWE" in e for e in errors)


def test_non_c_language_fails():
    s = _make_valid()
    s["language"] = "python"
    ok, errors = validate_sample(s)
    assert not ok and any("language" in e for e in errors)


def test_code_too_short_fails():
    s = _make_valid()
    s["code"] = "void f(){}"  # < 5 non-blank lines
    ok, errors = validate_sample(s)
    assert not ok and any("too short" in e for e in errors)


def test_code_too_long_fails():
    s = _make_valid()
    s["code"] = "\n".join(["int x;"] * 501)
    ok, errors = validate_sample(s)
    assert not ok and any("too long" in e for e in errors)


# --- make_sample_id ---
def test_make_sample_id_returns_uuid():
    id1 = make_sample_id()
    id2 = make_sample_id()
    assert id1 != id2
    assert len(id1) == 36  # UUID4 format


# --- make_timestamp ---
def test_make_timestamp_ends_with_z():
    ts = make_timestamp()
    assert ts.endswith("Z")


# --- VALID sets ---
def test_valid_cwe_has_12_entries():
    assert len(VALID_CWE) == 12


def test_valid_sources_has_cfa_variants():
    assert "sard_cfa" in VALID_SOURCES
    assert "repo_cfa" in VALID_SOURCES
