import json
import sys
from pathlib import Path

COLLECTION_PATH = Path(__file__).resolve().parents[2] / "training" / "scripts" / "collection"
sys.path.insert(0, str(COLLECTION_PATH))

from schema import VulnSample, make_sample_id, to_jsonl


def test_make_sample_id_is_stable() -> None:
    aliases = {"cve": ["CVE-2024-0001"], "ghsa": []}
    source = "nvd"
    normalized_code = "print('hello')"

    first = make_sample_id(aliases, normalized_code, source)
    second = make_sample_id(aliases, normalized_code, source)

    assert first == second
    assert "CVE-2024-0001" in first


def test_to_jsonl_produces_single_line_valid_json() -> None:
    sample = VulnSample(
        id="CVE-2024-0001:nvd:abcd1234",
        source="nvd",
        vuln_type="injection",
        language="python",
        vulnerable_code="print('hello')",
        fixed_code=None,
        metadata={"severity": "high"},
    )

    line = to_jsonl(sample)
    parsed = json.loads(line)

    assert parsed["metadata"]["severity"] == "high"
    assert not line.endswith("\n")
