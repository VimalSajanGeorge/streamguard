#!/usr/bin/env python3
"""
training/scripts/preprocessing/stage4_cpg.py

Stage 4: CPG (Code Property Graph) Construction for StreamGuard pipeline.

Phase 3 update: handles full multi-source dataset (SARD + CVE + repo + ExploitDB + OSV + GitHub Advisory).
- SARD samples get JULIET_STUB prepended + Juliet #include lines stripped
- Non-SARD samples get minimal standard includes only
- Taint sets extended for all 12 CWEs including SQL/command/XSS injection
- Checkpoint/resume for 40K+ samples with configurable checkpoint dir

Input:  training/data/processed/with_cfa/samples.jsonl  (Stage 3 CFA output)
Output: training/data/processed/cpg/{first_2_chars}/{sample_id}.json  (sharded, one CPG per sample)
        training/data/processed/cpg/cpg_failures.jsonl  (failed samples log)
        training/data/processed/cpg/cpg_stats.json      (run statistics)

Per-sample pipeline:
  1. Write sample code to temp .c file (with common header stubs)
  2. Run joern-parse to create cpg.bin (default overlays include REACHING_DEF/DFG)
  3. Run joern-export --repr cpg --format graphson to get JSON
  4. Parse GraphSON TinkerGraph format from export
  5. Filter edges: only keep AST(0), CFG(1), DFG(2) — drop all others
  6. Label taint roles on nodes (SOURCE, SINK, SANITIZER)
  7. Build TPG edges: BFS from SOURCE→SINK along DFG, add type=3
  8. Context slice: 2-hop BFS from taint seed nodes, max 200 nodes
  9. Validate: nodes >= 3, DFG edges > 0
  10. Save CPG JSON

Joern 4.x facts (verified against local install v4.0.498):
  - joern-parse default overlays INCLUDE data-flow (REACHING_DEF edges)
  - DFG edge label is "REACHING_DEF" (not "REACHES" which was Joern 1.x)
  - joern-export --format graphson produces TinkerGraph JSON
  - Export creates subdirectories per method, each with export.json
  - Node properties are nested: properties.NAME.@value.@value[0]

Edge types: AST=0, CFG=1, DFG=2, TPG=3 (4 types only).
CDG, DOMINATE, POST_DOMINATE, CONTAINS, ARGUMENT, REF, PARAMETER_LINK
are all DROPPED. Storing any as edge_type>=4 would cause CUDA index
out-of-bounds crash during GGNN training.

Source: docs/New Docs/StreamGuard_Stories5to8_RiskAnalysis.docx §Story 5
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

# ── Edge type mapping (4 types ONLY) ────────────────────────────
# Everything not listed here is DROPPED.
# CDG, DOMINATE, POST_DOMINATE, CONTAINS, ARGUMENT, REF,
# PARAMETER_LINK — all excluded to keep exactly 4 types for CUDA.
EDGE_TYPE_MAP = {"AST": 0, "CFG": 1, "DFG": 2}
TPG_TYPE = 3  # Taint Propagation Graph — built by BFS, not from Joern

# Joern 4.x edge labels that map to our types.
# REACHING_DEF is Joern 4.x's data-flow edge (was "REACHES" in Joern 1.x).
JOERN_LABEL_TO_TYPE = {
    "AST": 0,
    "CFG": 1,
    "REACHING_DEF": 2,
    # Legacy Joern 1.x — keep for compatibility if someone uses older Joern
    "REACHES": 2,
}

# ── Taint analysis sets ─────────────────────────────────────────
# SOURCE = where untrusted/attacker-controlled data ENTERS the program
# SINK   = where that data causes damage if unvalidated
#
# Design decisions:
#   - malloc/calloc/alloca are NOT sources — they allocate memory, not data.
#     Making them sources would teach the GNN "allocation = vulnerable".
#   - malloc/calloc are SINKS for integer overflow → heap overflow chains
#     (CWE-190 → CWE-122), where attacker-controlled size causes under-allocation.
#   - For output-parameter functions (recv, fscanf), Joern's DFG flows
#     call→RET, NOT call→buffer. We handle this with OUTPUT_PARAM_FUNCTIONS
#     in _build_tpg_edges to inject synthetic taint on buffer arguments.
SOURCES = frozenset({
    # Standard C input sources (untrusted data entry points)
    "scanf", "gets", "fgets", "getenv", "recv", "read", "fread",
    "getchar", "fscanf", "sscanf", "fgetc", "getline", "recvfrom",
    # SARD/Juliet network sources (Windows & POSIX)
    "accept", "listen", "connect",
    # Value-returning sources (return tainted value, NOT output-param)
    "fopen", "atoi", "strtol", "strtoul", "rand",
    # SARD/Juliet wide-char input
    "fgetws", "wscanf", "fwscanf",
    # CWE-89/78/79: web/CGI input sources
    "cgi_param", "getenv", "query_string",
})
SINKS = frozenset({
    # Standard C sinks — dangerous when receiving untrusted data
    "strcpy", "strcat", "system", "popen", "mysql_query", "sprintf",
    "printf", "fprintf", "execve", "execl", "execvp", "memcpy",
    "memmove", "gets", "vsprintf", "vfprintf",
    # SARD/Juliet sinks — wide string operations
    "wcscpy", "wcscat", "wmemcpy", "swprintf",
    "wprintf", "fwprintf",
    # Memory sinks (UAF, double-free)
    "free", "realloc",
    # Allocation sinks — dangerous when size is attacker-controlled
    # (CWE-190 integer overflow → CWE-122 heap under-allocation)
    "malloc", "calloc",
    # Juliet helper sinks
    "printline", "printintline", "printhexcharline",
    # CWE-89: SQL injection sinks
    "mysql_real_query", "sqlite3_exec", "pg_exec", "sql_exec",
    "sqlite3_prepare", "mysql_stmt_execute",
    # CWE-78: OS command injection sinks
    "execlp", "execle", "execvpe", "pclose", "dlopen",
    # CWE-79: XSS output sinks (when output goes to browser/response)
    "write", "send", "sendto",
})
SANITIZERS = frozenset({
    "strncpy", "strncat", "snprintf", "validate", "sanitize",
    "escape", "filter", "check", "bound_check", "verify",
    "strlcpy", "strlcat",
    # Wide string bounded variants
    "wcsncpy", "wcsncat",
    # CWE-89: SQL injection sanitizers
    "mysql_real_escape_string", "sqlite3_snprintf",
    "parameterize", "prepared_statement",
    # CWE-78: command injection sanitizers
    "escapeshellarg", "escapeshellcmd", "sanitize_cmd",
    # CWE-79: XSS sanitizers
    "htmlspecialchars", "html_escape", "encode_html",
})

# ── Output parameter functions ─────────────────────────────────
# Functions that write untrusted data into an output-parameter buffer.
# Joern's DFG tracks return values (call→RET), not side-effects on
# pointer arguments. We use this map to inject synthetic taint from
# the CALL node onto the correct argument's AST child node.
#
# Key: function name (lowercase)
# Value: 0-based indices of arguments that receive tainted data
OUTPUT_PARAM_FUNCTIONS: dict[str, list[int]] = {
    "recv":     [1],   # buf = arg[1]
    "recvfrom": [1],   # buf = arg[1]
    "read":     [1],   # buf = arg[1]
    "fread":    [0],   # ptr = arg[0]
    "fgets":    [0],   # buf = arg[0]
    "fgetws":   [0],   # buf = arg[0]
    "gets":     [0],   # buf = arg[0]
    "scanf":    slice(1, None),  # all args after format
    "fscanf":   slice(2, None),  # all args after stream, format
    "sscanf":   slice(2, None),  # all args after str, format
    "fgetc":    [],    # returns char, no output param
    "getchar":  [],    # returns char, no output param
    "getenv":   [],    # returns pointer, no output param
    "getline":  [0],   # lineptr = arg[0]
}

# ── Joern configuration ─────────────────────────────────────────
JOERN_TIMEOUT = 60  # seconds per step (parse + export separately)


def _run_subprocess_safe(
    cmd: list[str],
    timeout: int,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """R-17: Run subprocess with process-group kill on timeout.

    Joern's JVM spawns child processes. A simple subprocess.run() timeout
    sends SIGTERM to the parent only — JVM children survive and consume
    resources. This function:
    - Creates a new process group (POSIX) / job object (Windows)
    - On timeout, kills the ENTIRE process group
    - Prevents zombie Joern processes from accumulating
    """
    kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
    }
    # On POSIX, create new process group so we can kill all children
    if sys.platform != "win32":
        kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(cmd, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            cmd, proc.returncode, stdout, stderr
        )
    except subprocess.TimeoutExpired:
        # Kill entire process group
        if sys.platform == "win32":
            # Windows: taskkill /T kills process tree
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=10,
                )
            except Exception:
                proc.kill()
        else:
            # POSIX: kill process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        proc.wait(timeout=5)
        raise

# ── Common C header stubs ───────────────────────────────────────
# Minimal includes for non-SARD samples (CVE, repo, ExploitDB, etc.)
# These samples are already self-contained; we just add standard headers.
HEADER_STUBS = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
"""

# ── JULIET_STUB: SARD-specific stubs ───────────────────────────
# Juliet/SARD samples reference Juliet-internal helper functions
# (printLine, printIntLine, etc.) and standard library types.
# Without these, Joern fails to parse or produces incomplete CPGs.
JULIET_STUB = """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void printLine(const char* msg) {}
void printIntLine(int i) {}
void printDoubleLine(double d) {}
void printHexCharLine(char c) {}
void printLongLine(long l) {}
void printLongLongLine(long long ll) {}
void printUnsignedLine(unsigned u) {}
void printSizeTLine(size_t s) {}
"""

# Blocker #3: SARD #include lines that cause Joern parse failures.
# These refer to Juliet-internal headers that don't exist on disk.
# We strip them before writing the temp .c file.
SARD_STRIP_INCLUDES = frozenset({
    "std_testcase.h", "std_testcase_io.h", "std_thread.h",
})

# ── Context slice limits ────────────────────────────────────────
MAX_CONTEXT_NODES = 200
CONTEXT_HOPS = 2


# ── Blocker #3: Strip SARD-specific #include lines ──────────────
def _strip_sard_includes(code: str) -> str:
    """Remove #include lines for Juliet-internal headers that don't exist."""
    import re
    def _replace(m):
        header = m.group(1)
        if header in SARD_STRIP_INCLUDES:
            return ""  # Strip the line
        return m.group(0)  # Keep standard includes
    return re.sub(r'#\s*include\s*[<"]([^>"]+)[>"]', _replace, code)


def _is_sard_source(source: str) -> bool:
    """Check if sample source is SARD/Juliet."""
    return source.startswith("sard") if source else False


def write_temp_c(sample: dict, tmpdir: Path) -> Path:
    """Write sample code to a temp .c file with source-aware stubs.

    - SARD samples: prepend JULIET_STUB, strip Juliet-internal #include lines
    - Non-SARD samples: prepend minimal standard includes only

    Returns the path to the written .c file.
    """
    path = tmpdir / f"{sample['id']}.c"
    content = sample["code"]
    source = sample.get("source", "")

    if _is_sard_source(source):
        # SARD: strip Juliet includes, prepend JULIET_STUB
        content = _strip_sard_includes(content)
        content = JULIET_STUB + "\n" + content
    else:
        # Non-SARD: just prepend standard includes
        content = HEADER_STUBS + content

    path.write_text(content, encoding="utf-8")
    return path


# ── Blocker #10: Sharded output directory ───────────────────────
def _shard_path(output_dir: Path, sample_id: str) -> Path:
    """Return sharded path: cpg/{first_2_chars}/{sample_id}.json.
    Max 256 subdirs (hex), ~200 files each at 53K scale."""
    shard = sample_id[:2]
    shard_dir = output_dir / shard
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir / f"{sample_id}.json"


# ── Joern edge label → our edge type ───────────────────────────
def _map_edge_label(label: str) -> int | None:
    """Map Joern edge label to our 3-type system. Returns None for dropped types."""
    return JOERN_LABEL_TO_TYPE.get(label)


# ── Operator-based sinks ───────────────────────────────────────
# Joern models array writes as CALL nodes with operator names.
# buffer[data] = 1  →  <operator>.indirectIndexAccess (the read)
#                   →  <operator>.assignment (the write)
# When the index is attacker-controlled, this IS a sink (CWE-121/122).
# We detect: assignment CALL whose code contains "[" (array indexing).
OPERATOR_SINK_NAMES = frozenset({
    "<operator>.indirectIndexAccess",
    "<operator>.indexAccess",
})


# ── Taint role labeling ─────────────────────────────────────────
def _get_taint_role(node: dict) -> str:
    """Determine taint role from node properties.
    Only labels CALL nodes to avoid false-tagging BLOCK/METHOD nodes."""
    if node.get("_label") != "CALL":
        return "NONE"

    call_name = node.get("name", "")
    call_lower = call_name.lower()

    if call_lower in SOURCES:
        return "SOURCE"
    if call_lower in SINKS:
        return "SINK"
    if call_lower in SANITIZERS:
        return "SANITIZER"

    # Operator-based sinks: array writes like buffer[data] = 1
    if call_name in OPERATOR_SINK_NAMES:
        return "SINK"

    return "NONE"


# ── TPG edge construction via BFS ────────────────────────────────
def _build_tpg_edges(
    nodes: list[dict],
    edges: list[dict],
) -> list[dict]:
    """
    Build Taint Propagation Graph edges.

    BFS from each SOURCE node to each SINK node along DFG edges.
    If path exists AND does NOT pass through a SANITIZER -> add TPG edges.
    Intermediate nodes get PROPAGATION role.

    For output-parameter functions (recv, fscanf, etc.), Joern's DFG flows
    from the CALL node to RET, NOT to the buffer argument. We handle this by:
    1. Finding AST children of SOURCE CALL nodes
    2. Matching them to OUTPUT_PARAM_FUNCTIONS argument indices
    3. Starting BFS from those argument nodes (the actual tainted buffers)
    """
    # Build adjacency list from DFG edges only
    dfg_adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["type"] == EDGE_TYPE_MAP["DFG"]:
            dfg_adj[e["src"]].append(e["dst"])

    # Build AST child list to resolve argument nodes of CALL nodes
    ast_children: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["type"] == EDGE_TYPE_MAP["AST"]:
            ast_children[e["src"]].append(e["dst"])

    # Collect taint-role node IDs
    node_map = {n["id"]: n for n in nodes}
    node_roles = {n["id"]: n.get("taint_role", "NONE") for n in nodes}
    source_ids = [nid for nid, role in node_roles.items() if role == "SOURCE"]
    sink_ids = set(nid for nid, role in node_roles.items() if role == "SINK")
    sanitizer_ids = set(nid for nid, role in node_roles.items() if role == "SANITIZER")

    # Build BFS seed list: SOURCE nodes + their output-parameter arguments
    bfs_seeds: list[str] = list(source_ids)
    for src_id in source_ids:
        src_node = node_map.get(src_id)
        if src_node is None:
            continue
        call_name = src_node.get("name", "").lower()
        param_spec = OUTPUT_PARAM_FUNCTIONS.get(call_name)
        if param_spec is None:
            continue

        # Get ordered AST children (arguments) of this CALL node
        children = ast_children.get(src_id, [])
        if not children:
            # Fallback: add ALL AST children as seeds (best effort)
            continue

        # Resolve which children are tainted output params
        if isinstance(param_spec, slice):
            tainted_indices = range(*param_spec.indices(len(children)))
        else:
            tainted_indices = param_spec

        for idx in tainted_indices:
            if 0 <= idx < len(children):
                child_id = children[idx]
                if child_id not in bfs_seeds:
                    bfs_seeds.append(child_id)

        # Also add ALL children as fallback seeds for DFG traversal
        # because Joern's AST child ordering isn't always positional
        for child_id in children:
            if child_id not in bfs_seeds:
                bfs_seeds.append(child_id)

    tpg_edges = []
    propagation_nodes: set[str] = set()

    for src_id in bfs_seeds:
        # BFS from source
        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((src_id, [src_id]))
        visited: set[str] = {src_id}

        while queue:
            current, path = queue.popleft()

            if current in sink_ids and len(path) > 1:
                # Check no sanitizer in path
                if not any(p in sanitizer_ids for p in path[1:-1]):
                    # Add TPG edges along the path
                    for i in range(len(path) - 1):
                        tpg_edge = {
                            "src": path[i],
                            "dst": path[i + 1],
                            "type": TPG_TYPE,
                            "label": "TPG",
                        }
                        tpg_edges.append(tpg_edge)
                    # Mark intermediate nodes as PROPAGATION
                    for nid in path[1:-1]:
                        propagation_nodes.add(nid)
                continue  # Don't BFS past a sink

            for neighbor in dfg_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    # Update propagation roles on nodes
    for n in nodes:
        if n["id"] in propagation_nodes and n.get("taint_role") == "NONE":
            n["taint_role"] = "PROPAGATION"

    # Deduplicate TPG edges
    seen = set()
    unique_tpg = []
    for e in tpg_edges:
        key = (e["src"], e["dst"])
        if key not in seen:
            seen.add(key)
            unique_tpg.append(e)

    return unique_tpg


# ── Context slicing ─────────────────────────────────────────────
def _context_slice(
    nodes: list[dict],
    edges: list[dict],
    max_nodes: int = MAX_CONTEXT_NODES,
    hops: int = CONTEXT_HOPS,
) -> tuple[list[dict], list[dict]]:
    """
    2-hop BFS from taint seed nodes. Keep max 200 nodes.
    Taint seeds = nodes with taint_role != NONE.
    """
    # Find seed nodes
    seeds = [n["id"] for n in nodes if n.get("taint_role", "NONE") != "NONE"]

    if not seeds:
        # No taint nodes — keep all (up to max)
        if len(nodes) <= max_nodes:
            return nodes, edges
        return nodes[:max_nodes], [
            e for e in edges
            if e["src"] in {n["id"] for n in nodes[:max_nodes]}
            and e["dst"] in {n["id"] for n in nodes[:max_nodes]}
        ]

    # Build undirected adjacency for BFS
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e["src"]].add(e["dst"])
        adj[e["dst"]].add(e["src"])

    # BFS from all seeds simultaneously
    keep: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier: set[str] = set()
        for nid in frontier:
            for neighbor in adj.get(nid, set()):
                if neighbor not in keep:
                    next_frontier.add(neighbor)
                    keep.add(neighbor)
                    if len(keep) >= max_nodes:
                        break
            if len(keep) >= max_nodes:
                break
        frontier = next_frontier
        if len(keep) >= max_nodes:
            break

    sliced_nodes = [n for n in nodes if n["id"] in keep]
    sliced_edges = [e for e in edges if e["src"] in keep and e["dst"] in keep]
    return sliced_nodes, sliced_edges


# ── GraphSON TinkerGraph parser ──────────────────────────────────
def _extract_graphson_property(props: dict, key: str, default="") -> str:
    """Extract a property value from GraphSON nested structure.

    GraphSON format (strings):
      "NAME": {"@type": "g:VertexProperty",
               "@value": {"@type": "g:List", "@value": ["strcpy"]},
               "id": ...}

    GraphSON format (typed values like Int32):
      "LINE_NUMBER": {"@type": "g:VertexProperty",
                      "@value": {"@type": "g:List",
                                 "@value": [{"@type": "g:Int32", "@value": 13}]},
                      "id": ...}
    """
    prop = props.get(key)
    if prop is None:
        return default
    val = prop.get("@value", {})
    if isinstance(val, dict):
        inner = val.get("@value", default)
        if isinstance(inner, list) and inner:
            item = inner[0]
            # Handle typed values: {"@type": "g:Int32", "@value": 13}
            if isinstance(item, dict) and "@value" in item:
                return str(item["@value"])
            return str(item)
        return str(inner) if inner != default else default
    return str(val) if val else default


def _parse_graphson_export(export_dir: Path) -> tuple[list[dict], list[dict]] | None:
    """
    Parse joern-export GraphSON output.

    Joern-export creates subdirectories per method, each containing export.json.
    We merge all methods into a single node/edge list.
    Returns (nodes, edges) or None on failure.
    """
    all_nodes = []
    all_edges = []
    seen_node_ids: set[str] = set()

    # Find all export.json files in subdirectories
    export_files = sorted(export_dir.rglob("export.json"))
    if not export_files:
        return None

    for ef in export_files:
        try:
            raw = json.loads(ef.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.debug(f"Failed to parse {ef}: {exc}")
            continue

        graph = raw.get("@value", raw)
        raw_vertices = graph.get("vertices", [])
        raw_edges = graph.get("edges", [])

        # Parse vertices
        for rv in raw_vertices:
            nid = str(rv.get("id", {}).get("@value", rv.get("id", "")))
            if nid in seen_node_ids:
                continue
            seen_node_ids.add(nid)

            props = rv.get("properties", {})
            node = {
                "id": nid,
                "_label": rv.get("label", ""),
                "code": _extract_graphson_property(props, "CODE"),
                "name": _extract_graphson_property(props, "NAME"),
                "line": int(_extract_graphson_property(props, "LINE_NUMBER", "-1") or -1),
                "type_full": _extract_graphson_property(props, "TYPE_FULL_NAME"),
            }
            all_nodes.append(node)

        # Parse edges — filter to our 3 types
        for re_ in raw_edges:
            label = re_.get("label", "")
            edge_type = _map_edge_label(label)
            if edge_type is None:
                continue  # Drop CDG, DOMINATE, CONTAINS, ARGUMENT, REF, etc.

            src_id = re_.get("outV", {})
            dst_id = re_.get("inV", {})
            # Handle nested @value or plain int
            if isinstance(src_id, dict):
                src_id = str(src_id.get("@value", ""))
            else:
                src_id = str(src_id)
            if isinstance(dst_id, dict):
                dst_id = str(dst_id.get("@value", ""))
            else:
                dst_id = str(dst_id)

            edge = {
                "src": src_id,
                "dst": dst_id,
                "type": edge_type,
                "label": "DFG" if label in ("REACHING_DEF", "REACHES") else label,
            }
            all_edges.append(edge)

    if not all_nodes:
        return None

    return all_nodes, all_edges


# ── Legacy parser for Joern --script output (nodes.json/edges.json) ──
def _parse_joern_script_output(output_dir: Path) -> tuple[list[dict], list[dict]] | None:
    """
    Parse nodes.json and edges.json from Joern --script output.
    Fallback for non-CLI-pipeline setups.
    Returns (nodes, edges) or None on failure.
    """
    nodes_file = output_dir / "nodes.json"
    edges_file = output_dir / "edges.json"

    if not nodes_file.exists() or not edges_file.exists():
        return None

    try:
        raw_nodes = json.loads(nodes_file.read_text(encoding="utf-8"))
        raw_edges = json.loads(edges_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug(f"Failed to parse Joern JSON: {exc}")
        return None

    nodes = []
    for rn in raw_nodes:
        node = {
            "id": str(rn.get("id", rn.get("_id", ""))),
            "_label": rn.get("_label", rn.get("label", "")),
            "code": rn.get("code", rn.get("CODE", "")),
            "name": rn.get("name", rn.get("NAME", "")),
            "line": rn.get("lineNumber", rn.get("LINE_NUMBER", -1)),
            "type_full": rn.get("typeFullName", ""),
        }
        nodes.append(node)

    edges = []
    for re_ in raw_edges:
        label = re_.get("_label", re_.get("label", ""))
        edge_type = _map_edge_label(label)
        if edge_type is None:
            continue
        edge = {
            "src": str(re_.get("_outV", re_.get("outV", ""))),
            "dst": str(re_.get("_inV", re_.get("inV", ""))),
            "type": edge_type,
            "label": "DFG" if label in ("REACHING_DEF", "REACHES") else label,
        }
        edges.append(edge)

    return nodes, edges


# ── Run Joern CLI pipeline on a single sample ───────────────────
def _run_joern_cli(
    joern_dir: str,
    sample: dict,
) -> tuple[list[dict], list[dict]] | None:
    """
    Write code to temp .c, run joern-parse + joern-export, parse GraphSON.
    Returns (nodes, edges) or None on failure.

    Uses write_temp_c() for source-aware stub prepending:
    - SARD samples: JULIET_STUB + stripped Juliet includes
    - Non-SARD samples: minimal standard includes

    Pipeline:
      1. joern-parse sample.c -o cpg.bin  (generates CPG with default overlays incl DFG)
      2. joern-export cpg.bin -o export/ --repr cpg --format graphson
      3. Parse all export.json files from the export directory
    """
    sample_id = sample["id"]
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix=f"sg_cpg_{sample_id[:8]}_")
        tmp_path = Path(tmp_dir)
        cpg_bin = os.path.join(tmp_dir, "cpg.bin")
        export_dir = os.path.join(tmp_dir, "export")

        # Source-aware temp file writing
        tmp_c = str(write_temp_c(sample, tmp_path))

        # Resolve Joern binaries (prefer .bat on Windows)
        parse_bin = _resolve_joern_binary(joern_dir, "joern-parse")
        export_bin = _resolve_joern_binary(joern_dir, "joern-export")

        # Blocker #8: Limit JVM heap to 2GB per worker to avoid OOM
        env = os.environ.copy()
        env["_JAVA_OPTIONS"] = "-Xmx2g"

        # Step 1: joern-parse — generates cpg.bin with default overlays (incl DFG)
        # R-17: Use _run_subprocess_safe for process-group kill on timeout
        parse_cmd = [parse_bin, tmp_c, "-o", cpg_bin]
        result = _run_subprocess_safe(parse_cmd, timeout=JOERN_TIMEOUT, env=env)
        if result.returncode != 0:
            logger.debug(
                f"joern-parse failed for {sample_id}: rc={result.returncode}, "
                f"stderr={result.stderr[:200]}"
            )
            return None

        if not os.path.exists(cpg_bin):
            logger.debug(f"cpg.bin not created for {sample_id}")
            return None

        # Step 2: joern-export — GraphSON JSON output
        export_cmd = [
            export_bin, cpg_bin,
            "-o", export_dir,
            "--repr", "cpg",
            "--format", "graphson",
        ]
        result = _run_subprocess_safe(export_cmd, timeout=JOERN_TIMEOUT, env=env)
        if result.returncode != 0:
            logger.debug(
                f"joern-export failed for {sample_id}: rc={result.returncode}, "
                f"stderr={result.stderr[:200]}"
            )
            return None

        # Step 3: Parse GraphSON export
        return _parse_graphson_export(Path(export_dir))

    except subprocess.TimeoutExpired:
        logger.debug(f"Joern timed out for {sample_id} ({JOERN_TIMEOUT}s)")
        return None
    except Exception as exc:
        logger.debug(f"Joern error for {sample_id}: {exc}")
        return None
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_joern_binary(joern_dir: str, name: str) -> str:
    """Resolve Joern binary path, preferring .bat on Windows."""
    base = os.path.join(joern_dir, name)
    if sys.platform == "win32" or os.name == "nt":
        bat = base + ".bat"
        if os.path.exists(bat):
            return bat
    if os.path.exists(base):
        return base
    # Fallback: assume it's on PATH
    return name


# ── Process one sample end-to-end ───────────────────────────────
def process_sample(
    sample: dict,
    joern_dir: str,
) -> dict | None:
    """
    Full CPG pipeline for one sample:
    Joern → filter edges → taint labeling → TPG → context slice → validate.

    Returns CPG dict or None on failure.
    """
    sample_id = sample["id"]

    # Step 1-3: Run Joern CLI pipeline and parse output
    parsed = _run_joern_cli(joern_dir, sample)
    if parsed is None:
        return None
    nodes, edges = parsed

    # Step 4: Edge filtering already done in parser
    # (only AST/CFG/REACHING_DEF kept; CDG, DOMINATE, etc. dropped)

    # Step 5: Label taint roles on nodes
    for node in nodes:
        node["taint_role"] = _get_taint_role(node)

    # Step 6: Build TPG edges (BFS from SOURCE→SINK along DFG)
    tpg_edges = _build_tpg_edges(nodes, edges)
    edges.extend(tpg_edges)

    # Step 7: Context slice — 2-hop BFS from taint seeds, max 200 nodes
    nodes, edges = _context_slice(nodes, edges)

    # Step 8: Validate
    dfg_count = sum(1 for e in edges if e["type"] == EDGE_TYPE_MAP["DFG"])
    if len(nodes) < 3:
        return None
    if dfg_count == 0:
        return None

    # R-16 (CRITICAL): Assert no edge_type >= 4. CDG stored as type=4
    # causes CUDA index OOB during GGNN training — silent corruption.
    invalid_edges = [e for e in edges if e["type"] >= 4]
    if invalid_edges:
        logger.error(
            f"R-16: {len(invalid_edges)} edges with type>=4 in {sample_id}! "
            f"Labels: {set(e.get('label','?') for e in invalid_edges)}. "
            f"Dropping sample to prevent CUDA crash."
        )
        return None

    # Build final CPG
    cpg = {
        "sample_id": sample_id,
        "label": sample["label"],
        "cwe": sample.get("cwe", ""),
        "pair_id": sample.get("pair_id", ""),
        "source": sample.get("source", ""),
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "ast_edges": sum(1 for e in edges if e["type"] == EDGE_TYPE_MAP["AST"]),
            "cfg_edges": sum(1 for e in edges if e["type"] == EDGE_TYPE_MAP["CFG"]),
            "dfg_edges": dfg_count,
            "tpg_edges": sum(1 for e in edges if e["type"] == TPG_TYPE),
            "taint_sources": sum(1 for n in nodes if n.get("taint_role") == "SOURCE"),
            "taint_sinks": sum(1 for n in nodes if n.get("taint_role") == "SINK"),
            "taint_sanitizers": sum(1 for n in nodes if n.get("taint_role") == "SANITIZER"),
            "taint_propagation": sum(1 for n in nodes if n.get("taint_role") == "PROPAGATION"),
        },
    }
    return cpg


# ── Worker function for multiprocessing ─────────────────────────
def _worker(args: tuple) -> tuple[str, dict | None, dict | None]:
    """Worker for ProcessPoolExecutor. Returns (sample_id, cpg, failure_info)."""
    sample, joern_dir = args
    sample_id = sample["id"]
    try:
        cpg = process_sample(sample, joern_dir)
        if cpg is None:
            failure = {
                "sample_id": sample_id,
                "reason": "invalid_cpg",
                "cwe": sample.get("cwe", ""),
                "label": sample.get("label", -1),
            }
            return sample_id, None, failure
        return sample_id, cpg, None
    except Exception as exc:
        failure = {
            "sample_id": sample_id,
            "reason": str(exc)[:200],
            "cwe": sample.get("cwe", ""),
            "label": sample.get("label", -1),
        }
        return sample_id, None, failure


# ── Blocker #1 + R-34: Checkpoint save ────────────────────────────
def _save_checkpoint(checkpoint_path: Path, completed_ids: list[str]) -> None:
    """Atomic checkpoint save — write to .tmp then rename.

    R-34: On Windows, os.replace can fail if the target file is locked
    (e.g., antivirus scanning). We catch PermissionError and retry once
    after a short delay. A .bak backup is kept for crash recovery.
    """
    data = json.dumps({"completed_ids": completed_ids, "count": len(completed_ids)})
    tmp = checkpoint_path.with_suffix(".tmp")
    try:
        tmp.write_text(data, encoding="utf-8")
        # Keep a backup for crash recovery
        if checkpoint_path.exists():
            bak = checkpoint_path.with_suffix(".bak")
            try:
                shutil.copy2(checkpoint_path, bak)
            except OSError:
                pass  # Non-critical
        try:
            os.replace(tmp, checkpoint_path)
        except PermissionError:
            # R-34: Windows file lock — wait briefly and retry
            time.sleep(0.5)
            try:
                os.replace(tmp, checkpoint_path)
            except PermissionError:
                # Last resort: write directly (non-atomic but better than losing data)
                checkpoint_path.write_text(data, encoding="utf-8")
                logger.warning("R-34: Atomic checkpoint failed, used direct write fallback")
    except Exception as exc:
        logger.warning(f"Checkpoint save failed: {exc}")


# ── Main pipeline ───────────────────────────────────────────────
def run_stage4(
    input_path: str,
    output_dir: str,
    joern_dir: str = "C:/Tools/joern-cli",
    workers: int = 4,
    dry_run: bool = False,
    max_samples: int | None = None,
    checkpoint_dir: str | None = None,
) -> dict:
    """
    Run Stage 4 CPG construction on all samples.
    Returns stats dict.
    """
    logger.info("Stage 4: CPG Construction (Phase 3 — multi-source)")
    logger.info(f"  Input:      {input_path}")
    logger.info(f"  Output:     {output_dir}")
    logger.info(f"  Joern:      {joern_dir}")
    logger.info(f"  Workers:    {workers}")
    logger.info(f"  Checkpoint: {checkpoint_dir or output_dir}")

    # R-18: Memory guard — each Joern worker uses up to 2GB JVM heap.
    # Warn if total estimated memory exceeds available RAM.
    jvm_heap_per_worker_gb = 2
    estimated_mem_gb = workers * jvm_heap_per_worker_gb
    try:
        import shutil as _shutil
        total_mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (AttributeError, ValueError):
        # Windows: use ctypes or fallback
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_mem_gb = stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            total_mem_gb = 0  # Unknown — skip check

    if total_mem_gb > 0:
        if estimated_mem_gb > total_mem_gb * 0.8:
            logger.warning(
                f"R-18: {workers} workers × 2GB JVM = {estimated_mem_gb}GB estimated, "
                f"but system has {total_mem_gb:.1f}GB total RAM. "
                f"Risk of OOM. Consider reducing --workers to {max(1, int(total_mem_gb * 0.8 / jvm_heap_per_worker_gb))}."
            )
        else:
            logger.info(f"  Memory:     {estimated_mem_gb}GB estimated / {total_mem_gb:.1f}GB total")

    inp = Path(input_path)
    if not inp.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # R-33 (HIGH): Stage 4 MUST read with_cfa/ not deduped/.
    # If user accidentally passes deduped/, CFA samples get no CPGs and L_CFA = 0.
    inp_str = str(inp).replace("\\", "/")
    if "deduped" in inp_str and "with_cfa" not in inp_str:
        logger.warning(
            "R-33: Input path contains 'deduped' but NOT 'with_cfa'. "
            "Stage 4 should read from with_cfa/samples.jsonl (Stage 3 output) "
            "so that CFA counterpart samples also get CPGs. "
            f"Current input: {input_path}"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures_path = out / "cpg_failures.jsonl"
    stats_path = out / "cpg_stats.json"

    # Checkpoint can live in a separate directory (useful for shared storage)
    cp_dir = Path(checkpoint_dir) if checkpoint_dir else out
    cp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cp_dir / "checkpoint.json"

    # Load samples
    samples = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    n_total = len(samples)

    # R-33: Log source distribution and verify CFA samples are present
    source_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        source_counts[s.get("source", "unknown")] += 1
    logger.info(f"  Source distribution: {dict(source_counts)}")
    cfa_count = sum(v for k, v in source_counts.items() if k.endswith("_cfa"))
    if cfa_count == 0 and n_total > 0:
        logger.warning(
            "R-33: No CFA samples (source ending in '_cfa') found in input. "
            "If Stage 3 CFA generation was run, this input may be wrong."
        )
    else:
        logger.info(f"  CFA samples: {cfa_count} ({cfa_count/n_total*100:.1f}%)")

    # Blocker #9: Build pair_id lookup from ALL samples (before filtering)
    # so pair-aware discard can detect broken pairs across checkpoint boundaries.
    all_pair_lookup: dict[str, list[str]] = {}
    for s in samples:
        pid = s.get("pair_id", "")
        if pid and pid not in ("", "None", "null", "0"):
            all_pair_lookup.setdefault(pid, []).append(s["id"])

    if max_samples is not None:
        samples = samples[:max_samples]

    # Blocker #1: Checkpoint/resume — skip already-processed samples
    already_done: set[str] = set()
    if checkpoint_path.exists() and max_samples is None:
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            already_done = set(cp.get("completed_ids", []))
            logger.info(f"Resuming from checkpoint: {len(already_done)} already done")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt checkpoint, starting fresh")
    if already_done:
        samples = [s for s in samples if s["id"] not in already_done]

    n_input = len(samples)

    logger.info(f"Loaded {n_total} samples (processing {n_input})")

    if dry_run:
        logger.info("[DRY RUN] Would process samples, showing first 5:")
        for s in samples[:5]:
            code_preview = s["code"][:80].replace("\n", " ")
            logger.info(
                f"  {s['id'][:12]}... cwe={s.get('cwe','')} "
                f"label={s.get('label','')} code={code_preview}..."
            )
        return {"mode": "dry_run", "total": n_total, "would_process": n_input}

    # Verify Joern is available
    parse_bin = _resolve_joern_binary(joern_dir, "joern-parse")
    try:
        probe = subprocess.run(
            [parse_bin, "--help"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0:
            logger.info(f"Joern parse binary OK: {parse_bin}")
        else:
            raise FileNotFoundError(f"joern-parse returned rc={probe.returncode}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.error(
            f"Joern not found at {joern_dir}: {exc}\n"
            f"Install Joern: https://joern.io/docs/installation\n"
            f"Or set --joern-dir to the correct path."
        )
        sys.exit(1)

    # Process samples
    start_time = time.time()
    succeeded = 0
    failed = 0
    dfg_edge_total = 0
    tpg_edge_total = 0
    taint_node_total = 0
    dfg_check_done = False
    completed_ids: list[str] = list(already_done)
    failed_ids: set[str] = set()
    samples_with_taint = 0  # Blocker #6 counter

    # Per-source and per-CWE tracking for VC-S4-04, VC-S4-05, VC-S4-08
    per_source_ok: dict[str, int] = defaultdict(int)
    per_source_fail: dict[str, int] = defaultdict(int)
    per_cwe_node_counts: dict[str, list[int]] = defaultdict(list)
    # Build sample_id -> metadata for tracking
    _sample_meta: dict[str, tuple[str, str]] = {
        s["id"]: (s.get("source", "unknown"), s.get("cwe", "unknown"))
        for s in samples
    }

    # Open failures log (append if resuming)
    fail_mode = "a" if already_done else "w"
    fail_f = open(failures_path, fail_mode, encoding="utf-8")

    # ETA helper
    def _print_progress(done: int, ok: int, fail: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed > 0 else 0
        remaining = n_input - done
        eta_s = remaining / rate if rate > 0 else 0
        eta_h = eta_s / 3600
        total_done = len(already_done) + done
        total_all = n_total
        pct = total_done / total_all * 100 if total_all > 0 else 0
        print(
            f"[{total_done}/{total_all} {pct:.1f}%] "
            f"ok={ok} fail={fail} rate={rate:.2f}/s "
            f"ETA={eta_h:.1f}h ({eta_s/60:.0f}min)",
            flush=True,
        )
        logger.info(
            f"Progress: {done}/{n_input} "
            f"({ok} ok, {fail} fail, {rate:.2f} samples/s, ETA {eta_h:.1f}h)"
        )

    try:
        if workers <= 1:
            # Sequential processing
            for i, sample in enumerate(samples):
                sid = sample["id"]
                print(f"  -> [{i+1}/{n_input}] processing {sid[:16]}...", flush=True)
                sid, cpg, failure = _worker((sample, joern_dir))

                if cpg is not None:
                    # Blocker #10: Sharded output
                    cpg_path = _shard_path(out, sid)
                    cpg_path.write_text(
                        json.dumps(cpg, ensure_ascii=False), encoding="utf-8"
                    )
                    succeeded += 1
                    completed_ids.append(sid)
                    dfg_edge_total += cpg["stats"]["dfg_edges"]
                    tpg_edge_total += cpg["stats"]["tpg_edges"]
                    tn = sum(1 for n in cpg["nodes"]
                             if n.get("taint_role", "NONE") != "NONE")
                    taint_node_total += tn
                    if tn > 0:
                        samples_with_taint += 1
                    # Per-source/CWE tracking
                    _src, _cwe = _sample_meta.get(sid, ("unknown", "unknown"))
                    per_source_ok[_src] += 1
                    per_cwe_node_counts[_cwe].append(cpg["stats"]["node_count"])
                    print(
                        f"     OK  nodes={cpg['stats']['node_count']} "
                        f"ast={cpg['stats']['ast_edges']} "
                        f"cfg={cpg['stats']['cfg_edges']} "
                        f"dfg={cpg['stats']['dfg_edges']} "
                        f"tpg={cpg['stats']['tpg_edges']}",
                        flush=True,
                    )
                else:
                    failed += 1
                    failed_ids.add(sid)
                    fail_f.write(json.dumps(failure) + "\n")
                    _src, _ = _sample_meta.get(sid, ("unknown", "unknown"))
                    per_source_fail[_src] += 1
                    print(f"     FAIL reason={failure.get('reason','?')[:60]}", flush=True)

                # R-15 (CRITICAL): DFG sanity check after first 5 completed.
                # If DFG=0, STOP immediately — 10 min check prevents 4 days wasted.
                if not dfg_check_done and (succeeded + failed) >= 5:
                    dfg_check_done = True
                    if succeeded > 0 and dfg_edge_total == 0:
                        logger.critical(
                            "R-15 TRIGGERED: First 5 CPGs have 0 DFG edges! "
                            "Data-flow signal is MISSING. GGNN will fail. "
                            "Fix: ensure joern-parse runs WITHOUT --nooverlays. "
                            "In Joern REPL mode, call run.ossdataflow explicitly. "
                            "STOPPING to prevent 4 days of wasted CPG generation."
                        )
                        fail_f.close()
                        _save_checkpoint(checkpoint_path, completed_ids)
                        sys.exit(1)

                # Blocker #1: Save checkpoint every 500 samples
                if (i + 1) % 500 == 0:
                    _save_checkpoint(checkpoint_path, completed_ids)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    _print_progress(i + 1, succeeded, failed, elapsed)
        else:
            # Parallel processing
            work_items = [(s, joern_dir) for s in samples]
            completed = 0

            print(
                f"\nStarting parallel CPG construction: {n_input} samples, "
                f"{workers} workers\n"
                f"Estimated time: {n_input / workers / 12 / 3600:.1f}h "
                f"(assuming ~12s/sample)\n"
                f"Ctrl+C to stop — checkpoint saves every 500 samples\n",
                flush=True,
            )

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_worker, item): item[0]["id"]
                    for item in work_items
                }

                for future in as_completed(futures):
                    completed += 1
                    sid, cpg, failure = future.result()

                    if cpg is not None:
                        cpg_path = _shard_path(out, sid)
                        cpg_path.write_text(
                            json.dumps(cpg, ensure_ascii=False), encoding="utf-8"
                        )
                        succeeded += 1
                        completed_ids.append(sid)
                        dfg_edge_total += cpg["stats"]["dfg_edges"]
                        tpg_edge_total += cpg["stats"]["tpg_edges"]
                        tn = sum(1 for n in cpg["nodes"]
                                 if n.get("taint_role", "NONE") != "NONE")
                        taint_node_total += tn
                        if tn > 0:
                            samples_with_taint += 1
                        # Per-source/CWE tracking
                        _src, _cwe = _sample_meta.get(sid, ("unknown", "unknown"))
                        per_source_ok[_src] += 1
                        per_cwe_node_counts[_cwe].append(cpg["stats"]["node_count"])
                        print(
                            f"  OK  [{completed}/{n_input}] {sid[:16]} "
                            f"nodes={cpg['stats']['node_count']} "
                            f"ast={cpg['stats']['ast_edges']} "
                            f"cfg={cpg['stats']['cfg_edges']} "
                            f"dfg={cpg['stats']['dfg_edges']} "
                            f"tpg={cpg['stats']['tpg_edges']}",
                            flush=True,
                        )
                    else:
                        failed += 1
                        failed_ids.add(sid)
                        fail_f.write(json.dumps(failure) + "\n")
                        _src, _ = _sample_meta.get(sid, ("unknown", "unknown"))
                        per_source_fail[_src] += 1
                        print(
                            f"  FAIL [{completed}/{n_input}] {sid[:16]} "
                            f"reason={failure.get('reason','?')[:50]}",
                            flush=True,
                        )

                    # R-15 (CRITICAL): DFG sanity check — STOP if zero DFG.
                    if not dfg_check_done and completed >= 5:
                        dfg_check_done = True
                        if succeeded > 0 and dfg_edge_total == 0:
                            logger.critical(
                                "R-15 TRIGGERED: First 5 CPGs have 0 DFG edges! "
                                "Data-flow signal is MISSING. GGNN will fail. "
                                "STOPPING to prevent 4 days of wasted CPG generation."
                            )
                            fail_f.close()
                            _save_checkpoint(checkpoint_path, completed_ids)
                            sys.exit(1)

                    if completed % 500 == 0:
                        _save_checkpoint(checkpoint_path, completed_ids)

                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        _print_progress(completed, succeeded, failed, elapsed)

    finally:
        fail_f.close()
        _save_checkpoint(checkpoint_path, completed_ids)

    # Blocker #9: Pair-aware discard — if one member failed, remove its partner
    broken_pairs = 0
    for pid, members in all_pair_lookup.items():
        member_failed = [m for m in members if m in failed_ids]
        member_ok = [m for m in members if m not in failed_ids and m in set(completed_ids)]
        if member_failed and member_ok:
            for m in member_ok:
                cpg_path = _shard_path(out, m)
                if cpg_path.exists():
                    cpg_path.unlink()
                    succeeded -= 1
                    broken_pairs += 1
    if broken_pairs > 0:
        logger.info(
            f"Blocker #9: Removed {broken_pairs} pair partners of failed samples"
        )

    elapsed = time.time() - start_time
    success_rate = (succeeded / n_input * 100) if n_input > 0 else 0.0
    avg_dfg = dfg_edge_total / succeeded if succeeded > 0 else 0
    avg_taint = taint_node_total / succeeded if succeeded > 0 else 0

    # Per-CWE avg node counts for VC-S4-08 sanity check
    per_cwe_avg_nodes = {
        cwe: round(sum(counts) / len(counts), 1)
        for cwe, counts in per_cwe_node_counts.items()
        if counts
    }

    stats = {
        "input_total": n_total,
        "input_processed": n_input,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate_pct": round(success_rate, 2),
        "dfg_edges_total": dfg_edge_total,
        "tpg_edges_total": tpg_edge_total,
        "taint_nodes_total": taint_node_total,
        "avg_dfg_per_cpg": round(avg_dfg, 1),
        "avg_taint_per_cpg": round(avg_taint, 1),
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(n_input / elapsed, 2) if elapsed > 0 else 0,
        # Per-source breakdown for VC-S4-05
        "per_source_succeeded": dict(per_source_ok),
        "per_source_failed": dict(per_source_fail),
        # Per-CWE avg node count for VC-S4-08
        "per_cwe_avg_nodes": per_cwe_avg_nodes,
    }

    # Write stats
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(
        f"Stage 4 complete: {n_input} processed -> "
        f"{succeeded} CPGs ({success_rate:.1f}%), {failed} failed"
    )
    logger.info(f"  DFG edges: {dfg_edge_total} total (avg {avg_dfg:.1f}/CPG)")
    logger.info(f"  TPG edges: {tpg_edge_total} total")
    logger.info(f"  Taint nodes: {taint_node_total} total (avg {avg_taint:.1f}/CPG)")
    logger.info(f"  Time: {elapsed:.1f}s ({stats['samples_per_second']} samples/s)")

    if succeeded > 0 and dfg_edge_total == 0:
        logger.critical(
            "R-15: ALL CPGs have 0 DFG edges! "
            "Data-flow signal completely absent. Training will fail. "
            "Fix Joern overlay settings and re-run Stage 4."
        )
        sys.exit(1)

    # Blocker #6: Taint coverage audit
    if succeeded > 0:
        taint_pct = samples_with_taint / succeeded * 100
        stats["taint_coverage_pct"] = round(taint_pct, 2)
        stats["samples_with_taint"] = samples_with_taint
        stats["broken_pairs_removed"] = broken_pairs
        # Re-write stats with new fields
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        if taint_pct < 20.0:
            logger.warning(
                f"Blocker #6: Taint coverage is {taint_pct:.1f}% (< 20% target). "
                f"SOURCES/SINKS sets may need expansion for this CWE mix."
            )
        else:
            logger.info(f"Taint coverage: {taint_pct:.1f}% ({samples_with_taint}/{succeeded})")

    # VC-S4-08: Node count sanity — SARD CWE-121 avg should be 20-80.
    # If avg < 5, SARD stripping failed (R-01 cross-ref).
    for cwe, avg_nc in per_cwe_avg_nodes.items():
        if avg_nc < 5:
            logger.warning(
                f"VC-S4-08: {cwe} avg node count = {avg_nc} (< 5). "
                f"Possible R-01: #ifdef stripping may have produced empty CPGs. "
                f"Check Stage 1 clean output."
            )

    # Log per-source summary
    logger.info("Per-source results:")
    for src in sorted(set(list(per_source_ok.keys()) + list(per_source_fail.keys()))):
        ok = per_source_ok.get(src, 0)
        fl = per_source_fail.get(src, 0)
        total_src = ok + fl
        pct = ok / total_src * 100 if total_src > 0 else 0
        logger.info(f"  {src}: {ok}/{total_src} ({pct:.0f}% success)")

    return stats


# ── CLI ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Stage 4: CPG Construction (Joern → AST+CFG+DFG+TPG)"
    )
    ap.add_argument(
        "--input",
        default="training/data/processed/with_cfa/samples.jsonl",
        help="Input JSONL path (Stage 3 CFA output)",
    )
    ap.add_argument(
        "--output",
        default="training/data/processed/cpg/",
        help="Output directory for CPG JSON files",
    )
    ap.add_argument(
        "--joern-dir",
        default="C:/Tools/joern-cli",
        help="Path to Joern installation directory",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel Joern workers (default 2 to limit memory)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running Joern",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (for testing)",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoint file (defaults to --output dir)",
    )

    args = ap.parse_args()
    run_stage4(
        args.input,
        args.output,
        joern_dir=args.joern_dir,
        workers=args.workers,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
