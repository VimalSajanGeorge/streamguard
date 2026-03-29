#!/usr/bin/env python3
"""
tests/test_story5.py

Tests for Story 5: Stage 4 CPG Construction.

Tests the pure-Python logic (edge filtering, taint labeling, TPG BFS,
context slicing, GraphSON parsing, validation). Does NOT require Joern
to be installed — Joern integration is tested via --max-samples runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.preprocessing.stage4_cpg import (
    EDGE_TYPE_MAP,
    JOERN_LABEL_TO_TYPE,
    TPG_TYPE,
    SANITIZERS,
    SINKS,
    SOURCES,
    _build_tpg_edges,
    _context_slice,
    _extract_graphson_property,
    _get_taint_role,
    _map_edge_label,
    _parse_graphson_export,
    _parse_joern_script_output,
)


# ── Fixtures ────────────────────────────────────────────────────
@pytest.fixture
def basic_nodes():
    """A minimal set of CPG nodes with taint roles."""
    return [
        {"id": "1", "_label": "CALL", "code": "gets(buf)", "name": "gets", "line": 1, "type_full": "", "taint_role": "SOURCE"},
        {"id": "2", "_label": "IDENTIFIER", "code": "buf", "name": "buf", "line": 2, "type_full": "", "taint_role": "NONE"},
        {"id": "3", "_label": "CALL", "code": "strcpy(dst, buf)", "name": "strcpy", "line": 3, "type_full": "", "taint_role": "SINK"},
        {"id": "4", "_label": "CALL", "code": "printf(buf)", "name": "printf", "line": 4, "type_full": "", "taint_role": "SINK"},
    ]


@pytest.fixture
def basic_edges():
    """Edges connecting source -> intermediate -> sinks."""
    return [
        {"src": "1", "dst": "2", "type": 0, "label": "AST"},
        {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
        {"src": "2", "dst": "3", "type": 2, "label": "DFG"},
        {"src": "2", "dst": "4", "type": 2, "label": "DFG"},
        {"src": "1", "dst": "3", "type": 1, "label": "CFG"},
        {"src": "3", "dst": "4", "type": 1, "label": "CFG"},
    ]


@pytest.fixture
def sanitized_nodes():
    """Nodes with a sanitizer in the path."""
    return [
        {"id": "1", "_label": "CALL", "code": "gets(buf)", "name": "gets", "line": 1, "type_full": "", "taint_role": "SOURCE"},
        {"id": "2", "_label": "CALL", "code": "strncpy(dst, buf, n)", "name": "strncpy", "line": 2, "type_full": "", "taint_role": "SANITIZER"},
        {"id": "3", "_label": "CALL", "code": "strcpy(out, dst)", "name": "strcpy", "line": 3, "type_full": "", "taint_role": "SINK"},
    ]


@pytest.fixture
def sanitized_edges():
    """DFG path goes through sanitizer."""
    return [
        {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
        {"src": "2", "dst": "3", "type": 2, "label": "DFG"},
    ]


# ── Edge type mapping tests ─────────────────────────────────────
class TestEdgeTypeMapping:
    def test_ast_maps_to_0(self):
        assert _map_edge_label("AST") == 0

    def test_cfg_maps_to_1(self):
        assert _map_edge_label("CFG") == 1

    def test_reaching_def_maps_to_dfg_2(self):
        """Joern 4.x uses 'REACHING_DEF' for data-flow edges."""
        assert _map_edge_label("REACHING_DEF") == 2

    def test_reaches_maps_to_dfg_2(self):
        """Legacy Joern 1.x 'REACHES' also maps to DFG for compatibility."""
        assert _map_edge_label("REACHES") == 2

    def test_cdg_is_dropped(self):
        """CDG must be dropped — storing as type>=4 would crash CUDA."""
        assert _map_edge_label("CDG") is None

    def test_dominate_is_dropped(self):
        assert _map_edge_label("DOMINATE") is None

    def test_post_dominate_is_dropped(self):
        assert _map_edge_label("POST_DOMINATE") is None

    def test_contains_is_dropped(self):
        """CONTAINS is a Joern 4.x edge type that must be dropped."""
        assert _map_edge_label("CONTAINS") is None

    def test_argument_is_dropped(self):
        """ARGUMENT is a Joern 4.x edge type that must be dropped."""
        assert _map_edge_label("ARGUMENT") is None

    def test_ref_is_dropped(self):
        """REF is a Joern 4.x edge type that must be dropped."""
        assert _map_edge_label("REF") is None

    def test_parameter_link_is_dropped(self):
        """PARAMETER_LINK is a Joern 4.x edge type that must be dropped."""
        assert _map_edge_label("PARAMETER_LINK") is None

    def test_call_is_dropped(self):
        assert _map_edge_label("CALL") is None

    def test_unknown_is_dropped(self):
        assert _map_edge_label("FOOBAR") is None

    def test_edge_type_map_values(self):
        assert EDGE_TYPE_MAP == {"AST": 0, "CFG": 1, "DFG": 2}

    def test_tpg_type_is_3(self):
        assert TPG_TYPE == 3

    def test_only_4_edge_types_exist(self):
        """We have exactly 4 edge types: 0,1,2,3. No more."""
        all_types = set(EDGE_TYPE_MAP.values()) | {TPG_TYPE}
        assert all_types == {0, 1, 2, 3}

    def test_joern_label_map_has_reaching_def(self):
        """Joern 4.x REACHING_DEF must be in the label map."""
        assert "REACHING_DEF" in JOERN_LABEL_TO_TYPE
        assert JOERN_LABEL_TO_TYPE["REACHING_DEF"] == 2

    def test_joern_label_map_has_reaches(self):
        """Legacy REACHES must be in the label map for compatibility."""
        assert "REACHES" in JOERN_LABEL_TO_TYPE
        assert JOERN_LABEL_TO_TYPE["REACHES"] == 2


# ── Taint role labeling tests ───────────────────────────────────
class TestTaintRoles:
    def test_source_gets(self):
        node = {"_label": "CALL", "code": "gets(buf)", "name": "gets"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_scanf(self):
        node = {"_label": "CALL", "code": "scanf(\"%s\", buf)", "name": "scanf"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_fgets(self):
        node = {"_label": "CALL", "code": "fgets(buf, 100, stdin)", "name": "fgets"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_getenv(self):
        node = {"_label": "CALL", "code": "getenv(\"PATH\")", "name": "getenv"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_recv(self):
        node = {"_label": "CALL", "code": "recv(sock, buf, 100, 0)", "name": "recv"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_read(self):
        node = {"_label": "CALL", "code": "read(fd, buf, n)", "name": "read"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_fread(self):
        node = {"_label": "CALL", "code": "fread(buf, 1, n, fp)", "name": "fread"}
        assert _get_taint_role(node) == "SOURCE"

    def test_source_getchar(self):
        node = {"_label": "CALL", "code": "getchar()", "name": "getchar"}
        assert _get_taint_role(node) == "SOURCE"

    def test_sink_strcpy(self):
        node = {"_label": "CALL", "code": "strcpy(dst, src)", "name": "strcpy"}
        assert _get_taint_role(node) == "SINK"

    def test_sink_strcat(self):
        node = {"_label": "CALL", "code": "strcat(dst, src)", "name": "strcat"}
        assert _get_taint_role(node) == "SINK"

    def test_sink_system(self):
        node = {"_label": "CALL", "code": "system(cmd)", "name": "system"}
        assert _get_taint_role(node) == "SINK"

    def test_sink_printf(self):
        node = {"_label": "CALL", "code": "printf(buf)", "name": "printf"}
        assert _get_taint_role(node) == "SINK"

    def test_sink_sprintf(self):
        node = {"_label": "CALL", "code": "sprintf(buf, fmt)", "name": "sprintf"}
        assert _get_taint_role(node) == "SINK"

    def test_sink_execve(self):
        node = {"_label": "CALL", "code": "execve(path, argv, envp)", "name": "execve"}
        assert _get_taint_role(node) == "SINK"

    def test_sink_memcpy(self):
        node = {"_label": "CALL", "code": "memcpy(dst, src, n)", "name": "memcpy"}
        assert _get_taint_role(node) == "SINK"

    def test_sanitizer_strncpy(self):
        node = {"_label": "CALL", "code": "strncpy(d, s, n)", "name": "strncpy"}
        assert _get_taint_role(node) == "SANITIZER"

    def test_sanitizer_snprintf(self):
        node = {"_label": "CALL", "code": "snprintf(b, n, fmt)", "name": "snprintf"}
        assert _get_taint_role(node) == "SANITIZER"

    def test_sanitizer_validate(self):
        node = {"_label": "CALL", "code": "validate(input)", "name": "validate"}
        assert _get_taint_role(node) == "SANITIZER"

    def test_sanitizer_escape(self):
        node = {"_label": "CALL", "code": "escape(s)", "name": "escape"}
        assert _get_taint_role(node) == "SANITIZER"

    def test_none_for_regular_function(self):
        node = {"_label": "CALL", "code": "foo(x)", "name": "foo"}
        assert _get_taint_role(node) == "NONE"

    def test_none_for_identifier(self):
        node = {"_label": "IDENTIFIER", "code": "buf", "name": "buf"}
        assert _get_taint_role(node) == "NONE"

    def test_none_for_literal(self):
        node = {"_label": "LITERAL", "code": "42", "name": ""}
        assert _get_taint_role(node) == "NONE"

    def test_source_set_completeness(self):
        """All documented sources are in the set."""
        expected = {"scanf", "gets", "fgets", "getenv", "recv", "read", "fread", "getchar"}
        assert expected.issubset(SOURCES)

    def test_sink_set_completeness(self):
        """All documented sinks are in the set."""
        expected = {
            "strcpy", "strcat", "system", "popen", "mysql_query", "sprintf",
            "printf", "fprintf", "execve", "execl", "execvp",
        }
        assert expected.issubset(SINKS)

    def test_sanitizer_set_completeness(self):
        """All documented sanitizers are in the set."""
        expected = {
            "strncpy", "strncat", "snprintf", "validate", "sanitize",
            "escape", "filter", "check",
        }
        assert expected.issubset(SANITIZERS)


# ── TPG edge construction tests ─────────────────────────────────
class TestTPGEdges:
    def test_tpg_created_on_unsanitized_path(self, basic_nodes, basic_edges):
        """SOURCE->intermediate->SINK with no sanitizer should produce TPG edges."""
        tpg = _build_tpg_edges(basic_nodes, basic_edges)
        assert len(tpg) > 0
        assert all(e["type"] == TPG_TYPE for e in tpg)

    def test_tpg_label_is_tpg(self, basic_nodes, basic_edges):
        tpg = _build_tpg_edges(basic_nodes, basic_edges)
        assert all(e["label"] == "TPG" for e in tpg)

    def test_tpg_path_source_to_sink(self, basic_nodes, basic_edges):
        """TPG edges should connect along the DFG path from source to sink."""
        tpg = _build_tpg_edges(basic_nodes, basic_edges)
        tpg_srcs = {e["src"] for e in tpg}
        tpg_dsts = {e["dst"] for e in tpg}
        assert "1" in tpg_srcs  # Source node
        assert "3" in tpg_dsts or "4" in tpg_dsts  # Sink nodes

    def test_tpg_blocked_by_sanitizer(self, sanitized_nodes, sanitized_edges):
        """Path through sanitizer should NOT produce TPG edges."""
        tpg = _build_tpg_edges(sanitized_nodes, sanitized_edges)
        assert len(tpg) == 0

    def test_intermediate_gets_propagation_role(self, basic_nodes, basic_edges):
        """Intermediate nodes on TPG path should get PROPAGATION role."""
        _build_tpg_edges(basic_nodes, basic_edges)
        roles = {n["id"]: n["taint_role"] for n in basic_nodes}
        assert roles["2"] == "PROPAGATION"

    def test_no_tpg_without_dfg(self):
        """If there are no DFG edges, no TPG edges should be created."""
        nodes = [
            {"id": "1", "taint_role": "SOURCE"},
            {"id": "2", "taint_role": "SINK"},
        ]
        edges = [{"src": "1", "dst": "2", "type": 0, "label": "AST"}]
        tpg = _build_tpg_edges(nodes, edges)
        assert len(tpg) == 0

    def test_no_tpg_without_sink(self):
        """No sinks -> no TPG edges."""
        nodes = [
            {"id": "1", "taint_role": "SOURCE"},
            {"id": "2", "taint_role": "NONE"},
        ]
        edges = [{"src": "1", "dst": "2", "type": 2, "label": "DFG"}]
        tpg = _build_tpg_edges(nodes, edges)
        assert len(tpg) == 0

    def test_no_tpg_without_source(self):
        """No sources -> no TPG edges."""
        nodes = [
            {"id": "1", "taint_role": "NONE"},
            {"id": "2", "taint_role": "SINK"},
        ]
        edges = [{"src": "1", "dst": "2", "type": 2, "label": "DFG"}]
        tpg = _build_tpg_edges(nodes, edges)
        assert len(tpg) == 0

    def test_tpg_edges_deduplicated(self, basic_nodes, basic_edges):
        """TPG edges should not have duplicates."""
        tpg = _build_tpg_edges(basic_nodes, basic_edges)
        pairs = [(e["src"], e["dst"]) for e in tpg]
        assert len(pairs) == len(set(pairs))

    def test_direct_source_to_sink(self):
        """Direct DFG edge from SOURCE to SINK should create 1 TPG edge."""
        nodes = [
            {"id": "1", "taint_role": "SOURCE"},
            {"id": "2", "taint_role": "SINK"},
        ]
        edges = [{"src": "1", "dst": "2", "type": 2, "label": "DFG"}]
        tpg = _build_tpg_edges(nodes, edges)
        assert len(tpg) == 1
        assert tpg[0]["src"] == "1"
        assert tpg[0]["dst"] == "2"


# ── Context slicing tests ───────────────────────────────────────
class TestContextSlice:
    def test_keeps_taint_nodes(self):
        """Taint-role nodes should be in the slice."""
        nodes = [{"id": str(i), "taint_role": "NONE"} for i in range(10)]
        nodes[0]["taint_role"] = "SOURCE"
        nodes[5]["taint_role"] = "SINK"
        edges = [
            {"src": str(i), "dst": str(i+1), "type": 2, "label": "DFG"}
            for i in range(9)
        ]
        sliced_n, sliced_e = _context_slice(nodes, edges)
        sliced_ids = {n["id"] for n in sliced_n}
        assert "0" in sliced_ids  # SOURCE
        assert "5" in sliced_ids  # SINK

    def test_max_200_nodes(self):
        """Slice should not exceed 200 nodes."""
        nodes = [{"id": str(i), "taint_role": "NONE"} for i in range(500)]
        nodes[0]["taint_role"] = "SOURCE"
        edges = [
            {"src": str(i), "dst": str(i+1), "type": 2, "label": "DFG"}
            for i in range(499)
        ]
        sliced_n, _ = _context_slice(nodes, edges, max_nodes=200)
        assert len(sliced_n) <= 200

    def test_2_hop_neighborhood(self):
        """2-hop BFS should include nodes within 2 edges of seeds."""
        nodes = [
            {"id": "A", "taint_role": "SOURCE"},
            {"id": "B", "taint_role": "NONE"},
            {"id": "C", "taint_role": "NONE"},
            {"id": "D", "taint_role": "NONE"},
            {"id": "E", "taint_role": "NONE"},  # 3 hops away
        ]
        edges = [
            {"src": "A", "dst": "B", "type": 2, "label": "DFG"},
            {"src": "B", "dst": "C", "type": 2, "label": "DFG"},
            {"src": "C", "dst": "D", "type": 1, "label": "CFG"},
            {"src": "D", "dst": "E", "type": 2, "label": "DFG"},
        ]
        sliced_n, _ = _context_slice(nodes, edges, hops=2)
        sliced_ids = {n["id"] for n in sliced_n}
        assert "A" in sliced_ids
        assert "B" in sliced_ids
        assert "C" in sliced_ids

    def test_no_taint_keeps_all_under_max(self):
        """If no taint nodes, keep all (up to max)."""
        nodes = [{"id": str(i), "taint_role": "NONE"} for i in range(5)]
        edges = [{"src": "0", "dst": "1", "type": 0, "label": "AST"}]
        sliced_n, sliced_e = _context_slice(nodes, edges)
        assert len(sliced_n) == 5

    def test_edges_only_between_kept_nodes(self):
        """Sliced edges should only connect sliced nodes."""
        nodes = [
            {"id": "1", "taint_role": "SOURCE"},
            {"id": "2", "taint_role": "NONE"},
            {"id": "3", "taint_role": "NONE"},
            {"id": "4", "taint_role": "NONE"},
            {"id": "5", "taint_role": "NONE"},
        ]
        edges = [
            {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
            {"src": "2", "dst": "3", "type": 2, "label": "DFG"},
            {"src": "3", "dst": "4", "type": 2, "label": "DFG"},
            {"src": "4", "dst": "5", "type": 2, "label": "DFG"},
        ]
        sliced_n, sliced_e = _context_slice(nodes, edges, hops=2)
        sliced_ids = {n["id"] for n in sliced_n}
        for e in sliced_e:
            assert e["src"] in sliced_ids
            assert e["dst"] in sliced_ids


# ── GraphSON property extraction tests ──────────────────────────
class TestGraphSONPropertyExtraction:
    def test_extract_name_property(self):
        """GraphSON NAME property with nested @value structure."""
        props = {
            "NAME": {
                "@type": "g:VertexProperty",
                "@value": {"@type": "g:List", "@value": ["strcpy"]},
                "id": {"@type": "g:Int64", "@value": 6},
            }
        }
        assert _extract_graphson_property(props, "NAME") == "strcpy"

    def test_extract_code_property(self):
        props = {
            "CODE": {
                "@type": "g:VertexProperty",
                "@value": {"@type": "g:List", "@value": ["strcpy(dst, src)"]},
                "id": {"@type": "g:Int64", "@value": 1},
            }
        }
        assert _extract_graphson_property(props, "CODE") == "strcpy(dst, src)"

    def test_extract_missing_property(self):
        """Missing property returns default."""
        assert _extract_graphson_property({}, "CODE") == ""
        assert _extract_graphson_property({}, "CODE", "N/A") == "N/A"

    def test_extract_line_number_plain(self):
        props = {
            "LINE_NUMBER": {
                "@type": "g:VertexProperty",
                "@value": {"@type": "g:List", "@value": [7]},
                "id": {"@type": "g:Int64", "@value": 2},
            }
        }
        assert _extract_graphson_property(props, "LINE_NUMBER", "-1") == "7"

    def test_extract_typed_int32_value(self):
        """Joern 4.x LINE_NUMBER uses g:Int32 typed values inside the list."""
        props = {
            "LINE_NUMBER": {
                "@type": "g:VertexProperty",
                "@value": {
                    "@type": "g:List",
                    "@value": [{"@type": "g:Int32", "@value": 13}],
                },
                "id": {"@type": "g:Int64", "@value": 3},
            }
        }
        assert _extract_graphson_property(props, "LINE_NUMBER", "-1") == "13"

    def test_extract_typed_order_value(self):
        """ORDER property also uses g:Int32 typed values."""
        props = {
            "ORDER": {
                "@type": "g:VertexProperty",
                "@value": {
                    "@type": "g:List",
                    "@value": [{"@type": "g:Int32", "@value": 1}],
                },
                "id": {"@type": "g:Int64", "@value": 5},
            }
        }
        assert _extract_graphson_property(props, "ORDER") == "1"


# ── GraphSON export parsing tests ───────────────────────────────
class TestParseGraphSONExport:
    def _write_graphson(self, tmp_path, method_name, vertices, edges):
        """Helper to write a GraphSON export.json file."""
        method_dir = tmp_path / f"{method_name}.json"
        method_dir.mkdir(parents=True)
        data = {
            "@type": "tinker:graph",
            "@value": {"vertices": vertices, "edges": edges},
        }
        (method_dir / "export.json").write_text(json.dumps(data))

    def test_empty_dir_returns_none(self, tmp_path):
        assert _parse_graphson_export(tmp_path) is None

    def test_valid_graphson_output(self, tmp_path):
        """Parse a realistic GraphSON export with AST, CFG, REACHING_DEF edges."""
        vertices = [
            {
                "@type": "g:Vertex",
                "id": {"@type": "g:Int64", "@value": 100},
                "label": "METHOD",
                "properties": {
                    "CODE": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["void foo()"]}, "id": {"@type": "g:Int64", "@value": 1}},
                    "NAME": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["foo"]}, "id": {"@type": "g:Int64", "@value": 2}},
                },
            },
            {
                "@type": "g:Vertex",
                "id": {"@type": "g:Int64", "@value": 200},
                "label": "CALL",
                "properties": {
                    "CODE": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["gets(buf)"]}, "id": {"@type": "g:Int64", "@value": 3}},
                    "NAME": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["gets"]}, "id": {"@type": "g:Int64", "@value": 4}},
                },
            },
            {
                "@type": "g:Vertex",
                "id": {"@type": "g:Int64", "@value": 300},
                "label": "CALL",
                "properties": {
                    "CODE": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["strcpy(dst, buf)"]}, "id": {"@type": "g:Int64", "@value": 5}},
                    "NAME": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["strcpy"]}, "id": {"@type": "g:Int64", "@value": 6}},
                },
            },
        ]
        edges = [
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 0},
             "outV": {"@type": "g:Int64", "@value": 100}, "inV": {"@type": "g:Int64", "@value": 200},
             "label": "AST", "outVLabel": "METHOD", "inVLabel": "CALL", "properties": {}},
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 1},
             "outV": {"@type": "g:Int64", "@value": 200}, "inV": {"@type": "g:Int64", "@value": 300},
             "label": "CFG", "outVLabel": "CALL", "inVLabel": "CALL", "properties": {}},
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 2},
             "outV": {"@type": "g:Int64", "@value": 200}, "inV": {"@type": "g:Int64", "@value": 300},
             "label": "REACHING_DEF", "outVLabel": "CALL", "inVLabel": "CALL", "properties": {}},
            # CDG edge — should be dropped
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 3},
             "outV": {"@type": "g:Int64", "@value": 100}, "inV": {"@type": "g:Int64", "@value": 200},
             "label": "CDG", "outVLabel": "METHOD", "inVLabel": "CALL", "properties": {}},
            # DOMINATE edge — should be dropped
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 4},
             "outV": {"@type": "g:Int64", "@value": 200}, "inV": {"@type": "g:Int64", "@value": 300},
             "label": "DOMINATE", "outVLabel": "CALL", "inVLabel": "CALL", "properties": {}},
            # CONTAINS edge — should be dropped
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 5},
             "outV": {"@type": "g:Int64", "@value": 100}, "inV": {"@type": "g:Int64", "@value": 200},
             "label": "CONTAINS", "outVLabel": "METHOD", "inVLabel": "CALL", "properties": {}},
        ]
        self._write_graphson(tmp_path, "foo", vertices, edges)

        result = _parse_graphson_export(tmp_path)
        assert result is not None
        parsed_nodes, parsed_edges = result
        assert len(parsed_nodes) == 3
        # Only AST + CFG + REACHING_DEF = 3 kept; CDG + DOMINATE + CONTAINS = 3 dropped
        assert len(parsed_edges) == 3
        edge_types = {e["type"] for e in parsed_edges}
        assert edge_types == {0, 1, 2}

    def test_reaching_def_mapped_to_dfg(self, tmp_path):
        """Joern 4.x 'REACHING_DEF' should become DFG type=2 with label 'DFG'."""
        vertices = [
            {"@type": "g:Vertex", "id": {"@type": "g:Int64", "@value": 1},
             "label": "CALL", "properties": {}},
            {"@type": "g:Vertex", "id": {"@type": "g:Int64", "@value": 2},
             "label": "CALL", "properties": {}},
        ]
        edges = [
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": 0},
             "outV": {"@type": "g:Int64", "@value": 1}, "inV": {"@type": "g:Int64", "@value": 2},
             "label": "REACHING_DEF", "outVLabel": "CALL", "inVLabel": "CALL", "properties": {}},
        ]
        self._write_graphson(tmp_path, "test", vertices, edges)

        _, parsed_edges = _parse_graphson_export(tmp_path)
        assert len(parsed_edges) == 1
        assert parsed_edges[0]["type"] == 2
        assert parsed_edges[0]["label"] == "DFG"

    def test_node_ids_extracted_from_nested_value(self, tmp_path):
        """Node IDs should be extracted from @value nested structure."""
        vertices = [
            {"@type": "g:Vertex", "id": {"@type": "g:Int64", "@value": 99999},
             "label": "IDENTIFIER", "properties": {
                 "NAME": {"@type": "g:VertexProperty", "@value": {"@type": "g:List", "@value": ["buf"]}, "id": {"@type": "g:Int64", "@value": 1}},
             }},
        ]
        self._write_graphson(tmp_path, "test", vertices, [])

        parsed_nodes, _ = _parse_graphson_export(tmp_path)
        assert parsed_nodes[0]["id"] == "99999"
        assert parsed_nodes[0]["name"] == "buf"

    def test_all_joern4_edge_types_filtered(self, tmp_path):
        """All Joern 4.x non-CPG edge types must be dropped."""
        vertices = [
            {"@type": "g:Vertex", "id": {"@type": "g:Int64", "@value": 1}, "label": "X", "properties": {}},
        ]
        dropped_labels = ["CDG", "DOMINATE", "POST_DOMINATE", "CONTAINS",
                          "ARGUMENT", "REF", "PARAMETER_LINK"]
        edges = [
            {"@type": "g:Edge", "id": {"@type": "g:Int64", "@value": i},
             "outV": {"@type": "g:Int64", "@value": 1}, "inV": {"@type": "g:Int64", "@value": 1},
             "label": label, "outVLabel": "X", "inVLabel": "X", "properties": {}}
            for i, label in enumerate(dropped_labels)
        ]
        self._write_graphson(tmp_path, "test", vertices, edges)

        _, parsed_edges = _parse_graphson_export(tmp_path)
        assert len(parsed_edges) == 0

    def test_duplicate_nodes_across_methods_deduped(self, tmp_path):
        """Same node ID appearing in multiple method exports should be deduped."""
        vertices = [
            {"@type": "g:Vertex", "id": {"@type": "g:Int64", "@value": 42},
             "label": "CALL", "properties": {}},
        ]
        self._write_graphson(tmp_path, "method_a", vertices, [])
        self._write_graphson(tmp_path, "method_b", vertices, [])

        parsed_nodes, _ = _parse_graphson_export(tmp_path)
        assert len(parsed_nodes) == 1


# ── Legacy Joern script output parsing tests ────────────────────
class TestParseJoernScriptOutput:
    def test_missing_files_returns_none(self, tmp_path):
        assert _parse_joern_script_output(tmp_path) is None

    def test_missing_edges_returns_none(self, tmp_path):
        (tmp_path / "nodes.json").write_text("[]")
        assert _parse_joern_script_output(tmp_path) is None

    def test_valid_joern_script_output(self, tmp_path):
        nodes = [
            {"id": 1, "_label": "METHOD", "code": "void foo()", "name": "foo",
             "lineNumber": 1, "typeFullName": "void"},
            {"id": 2, "_label": "CALL", "code": "gets(buf)", "name": "gets",
             "lineNumber": 2, "typeFullName": "char*"},
        ]
        edges = [
            {"_label": "AST", "_outV": 1, "_inV": 2},
            {"_label": "REACHING_DEF", "_outV": 1, "_inV": 2},
            {"_label": "CDG", "_outV": 1, "_inV": 2},  # Dropped
        ]
        (tmp_path / "nodes.json").write_text(json.dumps(nodes))
        (tmp_path / "edges.json").write_text(json.dumps(edges))

        result = _parse_joern_script_output(tmp_path)
        assert result is not None
        parsed_nodes, parsed_edges = result
        assert len(parsed_nodes) == 2
        assert len(parsed_edges) == 2  # CDG dropped
        assert parsed_edges[1]["type"] == 2  # REACHING_DEF -> DFG
        assert parsed_edges[1]["label"] == "DFG"

    def test_malformed_json_returns_none(self, tmp_path):
        (tmp_path / "nodes.json").write_text("{bad json")
        (tmp_path / "edges.json").write_text("[]")
        assert _parse_joern_script_output(tmp_path) is None


# ── Edge type safety tests ──────────────────────────────────────
class TestEdgeTypeSafety:
    def test_no_edge_type_4_exists(self):
        """Edge type 4 must NEVER exist — would crash CUDA OOB."""
        all_types = set(EDGE_TYPE_MAP.values()) | {TPG_TYPE}
        assert 4 not in all_types

    def test_edge_types_are_contiguous(self):
        """Edge types 0-3 for efficient CUDA indexing."""
        all_types = sorted(set(EDGE_TYPE_MAP.values()) | {TPG_TYPE})
        assert all_types == [0, 1, 2, 3]

    def test_cdg_not_in_map(self):
        assert "CDG" not in EDGE_TYPE_MAP

    def test_only_3_joern_types_mapped(self):
        """EDGE_TYPE_MAP has only our 3 canonical types."""
        assert len(EDGE_TYPE_MAP) == 3

    def test_joern_label_map_only_maps_to_0_1_2(self):
        """All Joern label mappings produce types 0, 1, or 2 only."""
        for label, etype in JOERN_LABEL_TO_TYPE.items():
            assert etype in {0, 1, 2}, f"{label} maps to {etype}, expected 0-2"


# ── Integration-like tests (no Joern) ───────────────────────────
class TestEndToEnd:
    def test_full_tpg_pipeline(self):
        """Simulate full pipeline: nodes -> taint -> TPG -> slice."""
        nodes = [
            {"id": "1", "_label": "CALL", "code": "fgets(buf, 100, stdin)", "name": "fgets", "line": 1, "type_full": ""},
            {"id": "2", "_label": "IDENTIFIER", "code": "buf", "name": "buf", "line": 2, "type_full": ""},
            {"id": "3", "_label": "IDENTIFIER", "code": "dst", "name": "dst", "line": 3, "type_full": ""},
            {"id": "4", "_label": "CALL", "code": "strcpy(dst, buf)", "name": "strcpy", "line": 4, "type_full": ""},
            {"id": "5", "_label": "RETURN", "code": "return 0", "name": "", "line": 5, "type_full": ""},
        ]
        edges = [
            {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
            {"src": "2", "dst": "4", "type": 2, "label": "DFG"},
            {"src": "3", "dst": "4", "type": 2, "label": "DFG"},
            {"src": "1", "dst": "4", "type": 1, "label": "CFG"},
            {"src": "4", "dst": "5", "type": 1, "label": "CFG"},
            {"src": "1", "dst": "2", "type": 0, "label": "AST"},
        ]

        # Step 1: Label taint roles
        for n in nodes:
            n["taint_role"] = _get_taint_role(n)

        assert nodes[0]["taint_role"] == "SOURCE"  # fgets
        assert nodes[3]["taint_role"] == "SINK"     # strcpy

        # Step 2: Build TPG
        tpg = _build_tpg_edges(nodes, edges)
        assert len(tpg) > 0
        all_edges = edges + tpg

        # Step 3: Context slice
        sliced_n, sliced_e = _context_slice(nodes, all_edges)
        assert len(sliced_n) >= 3
        assert len(sliced_n) <= 200

        # Step 4: Verify DFG > 0
        dfg_count = sum(1 for e in sliced_e if e["type"] == 2)
        assert dfg_count > 0

        # Step 5: Verify no edge_type >= 4
        for e in sliced_e:
            assert e["type"] in {0, 1, 2, 3}, f"Invalid edge type: {e['type']}"

    def test_sanitized_pipeline_no_tpg(self):
        """Pipeline with sanitizer blocks TPG creation."""
        nodes = [
            {"id": "1", "_label": "CALL", "code": "gets(buf)", "name": "gets", "line": 1, "type_full": ""},
            {"id": "2", "_label": "CALL", "code": "strncpy(safe, buf, n)", "name": "strncpy", "line": 2, "type_full": ""},
            {"id": "3", "_label": "CALL", "code": "strcpy(dst, safe)", "name": "strcpy", "line": 3, "type_full": ""},
        ]
        edges = [
            {"src": "1", "dst": "2", "type": 2, "label": "DFG"},
            {"src": "2", "dst": "3", "type": 2, "label": "DFG"},
        ]

        for n in nodes:
            n["taint_role"] = _get_taint_role(n)

        assert nodes[1]["taint_role"] == "SANITIZER"

        tpg = _build_tpg_edges(nodes, edges)
        assert len(tpg) == 0
