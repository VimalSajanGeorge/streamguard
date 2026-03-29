import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def _parse_semver(version: str) -> tuple[int, int, int]:
    core = version.split("+", 1)[0]
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", core)
    if not match:
        raise AssertionError(f"Unable to parse semantic version from: {version}")
    return tuple(int(part) for part in match.groups())


@pytest.mark.skipif(
    os.getenv("RUN_STORY1_INSTALL_TEST") != "1",
    reason="Set RUN_STORY1_INSTALL_TEST=1 to run full requirements install in a fresh venv.",
)
def test_story1_requirements_install_succeeds(tmp_path: Path) -> None:
    """Acceptance: pip install -r requirements.txt succeeds."""
    venv_dir = tmp_path / "story1_env"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    python_bin = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    timeout_seconds = int(os.getenv("STORY1_PIP_INSTALL_TIMEOUT", "2400"))

    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        check=True,
        timeout=timeout_seconds,
    )


def test_story1_torch_geometric_version_is_at_least_2_4_0() -> None:
    """Acceptance: torch_geometric version prints >= 2.4.0."""
    tg = pytest.importorskip("torch_geometric")
    assert _parse_semver(tg.__version__) >= (2, 4, 0), (
        f"Expected torch-geometric>=2.4.0, got {tg.__version__}"
    )


def test_story1_joern_generates_cpg_json_from_5_line_c_function(tmp_path: Path) -> None:
    """Acceptance: Joern produces CPG JSON from a 5-line C function."""
    joern_bin = os.getenv("JOERN_BIN")
    if not joern_bin:
        pytest.skip("JOERN_BIN is not set.")

    c_file = tmp_path / "story1_joern_test.c"
    c_file.write_text(
        "#include <string.h>\n"
        "void foo(char *s) {\n"
        "  char buf[8];\n"
        "  strcpy(buf, s);\n"
        "}\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "joern_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        joern_bin,
        "--script",
        os.getenv("JOERN_SMOKE_SCRIPT", "generate_cpg.sc"),
        "--params",
        f"inputFile={c_file},outputDir={output_dir}",
    ]

    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("STORY1_JOERN_TIMEOUT", "300")),
    )

    assert result.returncode == 0, (
        f"Joern failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    json_files = list(output_dir.rglob("*.json"))
    if not json_files:
        # Fallback for scripts that write output outside explicit outputDir.
        json_files = [p for p in tmp_path.rglob("*.json") if p.is_file()]

    assert json_files, (
        "Joern command succeeded but no CPG JSON was found. "
        "Check JOERN_SMOKE_SCRIPT output behavior."
    )

    combined_text = "".join(
        path.read_text(encoding="utf-8", errors="ignore")[:20000] for path in json_files[:10]
    )
    has_dfg = ("DFG" in combined_text) or ("REACHING_DEF" in combined_text)
    assert "AST" in combined_text and "CFG" in combined_text and has_dfg, (
        "CPG JSON does not contain expected AST/CFG/DFG edge labels."
    )


def test_story1_codebert_cls_embedding_is_768() -> None:
    """Acceptance: CodeBERT [CLS] embedding dimension is 768."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = transformers.AutoModel.from_pretrained("microsoft/codebert-base")
    except OSError as exc:
        pytest.skip(f"CodeBERT model/tokenizer unavailable locally and could not be downloaded: {exc}")

    encoded = tokenizer(
        "void foo() { return; }",
        return_tensors="pt",
        max_length=64,
        truncation=True,
    )
    output = model(**encoded)
    assert output.last_hidden_state.shape[-1] == 768


def test_story1_tree_sitter_c_parses_function_definition() -> None:
    """Acceptance: tree-sitter-c parses function_definition in C code."""
    tree_sitter = pytest.importorskip("tree_sitter")
    tree_sitter_c = pytest.importorskip("tree_sitter_c")

    c_lang = tree_sitter.Language(tree_sitter_c.language())
    parser = tree_sitter.Parser(c_lang)
    tree = parser.parse(b"void foo() { int x = 1; }")

    assert tree.root_node.type == "translation_unit"
    assert any(node.type == "function_definition" for node in tree.root_node.children)
