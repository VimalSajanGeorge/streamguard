"""
Story 1 — Environment Verification Script
Run: python training/scripts/verify_setup.py
"""

import sys

PASS = "\u2705"
FAIL = "\u274c"
SKIP = "\u26a0\ufe0f"
results = []


def check(name, fn):
    try:
        msg = fn()
        results.append((PASS, name, msg))
        print(f"  {PASS} {name}: {msg}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL} {name}: {e}")


# ── 1. Python version ──
def check_python():
    v = sys.version_info
    assert v >= (3, 10), f"Need Python >=3.10, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


# ── 2. PyTorch ──
def check_torch():
    import torch
    assert torch.__version__ >= "2.2", f"Need torch >=2.2, got {torch.__version__}"
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU only"
    return f"{torch.__version__} | CUDA: {cuda} | {device}"


# ── 3. PyTorch Geometric + GatedGraphConv ──
def check_pyg():
    import torch_geometric
    from torch_geometric.nn import GatedGraphConv
    return f"{torch_geometric.__version__} | GatedGraphConv available"


# ── 4. Transformers + CodeBERT ──
def check_codebert():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    mdl = AutoModel.from_pretrained("microsoft/codebert-base")
    inp = tok("void foo() { return; }", return_tensors="pt",
              max_length=64, truncation=True)
    out = mdl(**inp)
    dim = out.last_hidden_state.shape[-1]
    assert dim == 768, f"Expected 768-d, got {dim}"
    return f"[CLS] dim={dim}"


# ── 5. tree-sitter-c ──
def check_tree_sitter_c():
    from tree_sitter import Language, Parser
    import tree_sitter_c as tsc
    lang = Language(tsc.language())
    parser = Parser(lang)
    tree = parser.parse(b"void foo() { int x = 1; }")
    root = tree.root_node
    assert root.type == "translation_unit"
    func_types = [n.type for n in root.children]
    assert "function_definition" in func_types, f"No function_definition in {func_types}"
    return "C parser OK | function_definition parsed"


# ── 6. New pipeline dependencies ──
def check_pipeline_deps():
    import tenacity, loguru, datasketch, h5py, chardet, networkx
    versions = {
        "tenacity": tenacity.__version__ if hasattr(tenacity, '__version__') else "OK",
        "loguru": loguru.__version__ if hasattr(loguru, '__version__') else "OK",
        "datasketch": datasketch.__version__ if hasattr(datasketch, '__version__') else "OK",
        "h5py": h5py.__version__,
        "chardet": chardet.__version__,
        "networkx": networkx.__version__,
    }
    return " | ".join(f"{k}={v}" for k, v in versions.items())


# ── 7. MLflow ──
def check_mlflow():
    import mlflow
    return f"{mlflow.__version__}"


# ── 8. Java (for Joern) ──
def check_java():
    import subprocess
    r = subprocess.run(["java", "--version"], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError("Java not found on PATH — install JDK 17+ from https://adoptium.net/")
    first_line = r.stdout.strip().split("\n")[0]
    return first_line


# ── 9. Joern CPG smoke test ──
def check_joern():
    import os, subprocess, tempfile, json
    from pathlib import Path

    joern_bin = os.getenv("JOERN_BIN", "")
    if not joern_bin:
        raise RuntimeError("JOERN_BIN not set in .env — set path to joern.bat")
    if not Path(joern_bin).exists():
        raise FileNotFoundError(f"JOERN_BIN={joern_bin} does not exist")

    # Write a test C file with a known taint path
    test_code = 'void foo(char *s) { char buf[64]; strcpy(buf, s); }\n'
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.c"
        test_file.write_text(test_code)
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        # Run Joern: import code, run dataflow, query CPG
        # Uses high-level traversals (works on both overflowdb and flatgraph Joern)
        joern_script = (
            f'importCode("{test_file.as_posix()}")\n'
            f'run.ossdataflow\n'
            f'val methods = cpg.method.name("foo").l\n'
            f'println("METHOD_COUNT=" + methods.size)\n'
            f'val astNodes = cpg.method.name("foo").ast.l\n'
            f'println("NODE_COUNT=" + astNodes.size)\n'
            f'val calls = cpg.call.name("strcpy").l\n'
            f'println("CALL_COUNT=" + calls.size)\n'
            f'val params = cpg.method.name("foo").parameter.l\n'
            f'println("PARAM_COUNT=" + params.size)\n'
        )
        script_file = Path(tmpdir) / "test_cpg.sc"
        script_file.write_text(joern_script)

        r = subprocess.run(
            [joern_bin, "--script", str(script_file)],
            capture_output=True, text=True, timeout=120, cwd=tmpdir
        )

        output = r.stdout + r.stderr
        if r.returncode != 0 and "NODE_COUNT=" not in output:
            raise RuntimeError(f"Joern exited with code {r.returncode}: {output[-500:]}")

        # Parse output
        node_count = 0
        method_count = 0
        call_count = 0
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("METHOD_COUNT="):
                method_count = int(line.split("=")[1])
            if line.startswith("NODE_COUNT="):
                node_count = int(line.split("=")[1])
            if line.startswith("CALL_COUNT="):
                call_count = int(line.split("=")[1])

        if method_count == 0:
            raise RuntimeError("Joern found 0 methods — CPG construction failed")
        if node_count < 3:
            raise RuntimeError(f"Joern produced only {node_count} AST nodes (expected >=3)")

        parts = [f"{node_count} AST nodes", f"strcpy call found={call_count > 0}"]
        return f"CPG OK | {' | '.join(parts)}"


# ── Run all checks ──
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  StreamGuard v2 — Story 1 Environment Verification")
    print("=" * 60 + "\n")

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    checks = [
        ("Python >= 3.10", check_python),
        ("PyTorch >= 2.2", check_torch),
        ("PyG + GatedGraphConv", check_pyg),
        ("CodeBERT [CLS] 768-d", check_codebert),
        ("tree-sitter-c parser", check_tree_sitter_c),
        ("Pipeline deps", check_pipeline_deps),
        ("MLflow", check_mlflow),
        ("Java 17+ (for Joern)", check_java),
        ("Joern CLI", check_joern),
    ]

    for name, fn in checks:
        check(name, fn)

    # Summary
    passed = sum(1 for s, _, _ in results if s == PASS)
    failed = sum(1 for s, _, _ in results if s == FAIL)

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(results)}")
    if failed > 0:
        print(f"\n  Failed checks:")
        for s, name, msg in results:
            if s == FAIL:
                print(f"    {FAIL} {name}: {msg}")
    print("=" * 60 + "\n")

    sys.exit(1 if failed > 0 else 0)
