#!/usr/bin/env python3
"""
training/scripts/preprocessing/cfa_tier1.py

Tier 1: Rule-Based AST Mutation for CFA Generation.

Handles CWE-134 (format string), CWE-120 (buffer copy), CWE-476 (NULL deref).
These CWEs have deterministic, structurally-correct fixes that can be applied
via tree-sitter AST analysis without any LLM call.

CRITICAL safety: CWE-120 fixes using sizeof(dst) are only applied when dst is
declared as a local ARRAY (sizeof = full buffer size). If dst is a POINTER
PARAMETER, sizeof would return 4/8 bytes — producing a wrong fix. In that case,
the function returns None so the orchestrator can escalate to Tier 2 (LLM).

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2.1
"""
from __future__ import annotations

from tree_sitter import Language, Node, Parser

try:
    import tree_sitter_c as tsc
except ImportError:
    raise ImportError("tree_sitter_c required. Run: pip install tree-sitter-c")

C_LANGUAGE = Language(tsc.language())
_parser = Parser(C_LANGUAGE)

# ── Format-string functions by category ──────────────────────────
# Map: func_name -> index of the FORMAT argument (0-based)
# printf(fmt, ...)          -> fmt is arg 0
# fprintf(stream, fmt, ...) -> fmt is arg 1
_FORMAT_FUNCS: dict[str, int] = {
    "printf": 0, "vprintf": 0, "dprintf": 0, "wprintf": 0, "vwprintf": 0,
    "fprintf": 1, "fwprintf": 1,
    "sprintf": 1, "snprintf": 2, "swprintf": 2,
}

# Wide-char variants need L"..." prefix
_WIDE_FORMAT_FUNCS = {"wprintf", "fwprintf", "vwprintf", "swprintf"}

# ── CWE-120 unsafe functions ────────────────────────────────────
_UNSAFE_COPY_FUNCS = {"strcpy", "strcat", "gets", "sprintf", "vsprintf"}


# ── Tree-sitter helpers ─────────────────────────────────────────
def _parse(code: str) -> Node:
    """Parse C code and return root node."""
    tree = _parser.parse(code.encode("utf-8", errors="replace"))
    return tree.root_node


def _node_text(node: Node, code: str) -> str:
    """Extract text of a node from source code."""
    return code[node.start_byte:node.end_byte]


def _find_all(root: Node, node_type: str) -> list[Node]:
    """Iterative DFS to find all nodes of a given type."""
    result = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == node_type:
            result.append(n)
        stack.extend(reversed(n.children))
    return result


def _find_enclosing_function(node: Node) -> Node | None:
    """Walk up to find the enclosing function_definition."""
    cur = node.parent
    while cur is not None:
        if cur.type == "function_definition":
            return cur
        cur = cur.parent
    return None


def _get_function_return_type(func_node: Node, code: str) -> str:
    """Extract the return type string from a function_definition node."""
    # Return type is the child(ren) before the function_declarator.
    # Could be: primitive_type, type_identifier, struct_specifier, etc.
    # Possibly with type_qualifier (const, etc.) or pointer (*).
    parts = []
    for child in func_node.children:
        if child.type == "function_declarator":
            break
        if child.type == "compound_statement":
            break
        parts.append(_node_text(child, code))
    return " ".join(parts).strip()


def _get_call_args(call_node: Node) -> list[Node]:
    """Get argument nodes from a call_expression's argument_list."""
    arg_list = None
    for child in call_node.children:
        if child.type == "argument_list":
            arg_list = child
            break
    if arg_list is None:
        return []
    return [c for c in arg_list.children if c.type not in ("(", ")", ",")]


def _get_call_func_name(call_node: Node, code: str) -> str:
    """Get function name from a call_expression."""
    for child in call_node.children:
        if child.type == "identifier":
            return _node_text(child, code)
    return ""


def _get_pointer_params(func_node: Node, code: str) -> set[str]:
    """Get names of pointer parameters in a function definition."""
    params = set()
    decl = None
    for child in func_node.children:
        if child.type == "function_declarator":
            decl = child
            break
    if decl is None:
        return params
    for child in decl.children:
        if child.type == "parameter_list":
            for param in child.children:
                if param.type == "parameter_declaration":
                    # Check if the declarator is a pointer_declarator
                    for pc in param.children:
                        if pc.type == "pointer_declarator":
                            # The identifier inside pointer_declarator
                            for idc in pc.children:
                                if idc.type == "identifier":
                                    params.add(_node_text(idc, code))
    return params


def _get_local_arrays(func_node: Node, code: str) -> set[str]:
    """Get names of locally declared arrays in a function body."""
    arrays = set()
    body = None
    for child in func_node.children:
        if child.type == "compound_statement":
            body = child
            break
    if body is None:
        return arrays
    for decl in _find_all(body, "array_declarator"):
        for child in decl.children:
            if child.type == "identifier":
                arrays.add(_node_text(child, code))
                break
    return arrays


def _is_inside_null_check(node: Node, ptr_name: str, code: str) -> bool:
    """Check if node is inside an if-block that checks ptr_name for NULL."""
    cur = node.parent
    while cur is not None:
        if cur.type == "if_statement":
            # Check the condition
            for child in cur.children:
                if child.type == "parenthesized_expression":
                    cond_text = _node_text(child, code)
                    # Match: (ptr != NULL), (ptr == NULL), (NULL != ptr), (!ptr), (ptr)
                    if ptr_name in cond_text and "NULL" in cond_text:
                        return True
                    if cond_text.strip("()").strip() == f"!{ptr_name}":
                        return True
                    if cond_text.strip("()").strip() == ptr_name:
                        return True
        cur = cur.parent
    return False


# ── Edit application ────────────────────────────────────────────
def _apply_edits(code: str, edits: list[tuple[int, int, str]]) -> str:
    """
    Apply byte-offset edits to code. Edits are (start, end, replacement).
    Must be applied in reverse order to preserve offsets.
    """
    edits_sorted = sorted(edits, key=lambda e: e[0], reverse=True)
    result = code
    for start, end, replacement in edits_sorted:
        result = result[:start] + replacement + result[end:]
    return result


# ── CWE-134: Format String Fix ──────────────────────────────────
def fix_cwe134(code: str) -> str | None:
    """
    Fix CWE-134 format string vulnerabilities.

    Finds calls to printf-family functions where the format argument is NOT
    a string literal (i.e., a variable is passed directly as the format string).
    Inserts the appropriate format specifier.

    Returns fixed code, or None if no format string vulnerability found.
    """
    root = _parse(code)
    calls = _find_all(root, "call_expression")
    edits: list[tuple[int, int, str]] = []

    for call in calls:
        fname = _get_call_func_name(call, code)
        if fname not in _FORMAT_FUNCS:
            continue

        fmt_idx = _FORMAT_FUNCS[fname]
        args = _get_call_args(call)
        if len(args) <= fmt_idx:
            continue

        fmt_arg = args[fmt_idx]

        # If the format argument is already a string_literal -> not vulnerable
        if fmt_arg.type == "string_literal":
            continue

        # Vulnerable: variable used directly as format string
        # Insert format specifier before the variable arg
        var_text = _node_text(fmt_arg, code)

        if fname in _WIDE_FORMAT_FUNCS:
            fmt_str = 'L"%s", '
        else:
            fmt_str = '"%s", '

        edits.append((fmt_arg.start_byte, fmt_arg.start_byte, fmt_str))

    if not edits:
        return None

    return _apply_edits(code, edits)


# ── CWE-120: Buffer Copy Fix ────────────────────────────────────
def fix_cwe120(code: str) -> str | None:
    """
    Fix CWE-120 buffer copy vulnerabilities.

    Finds calls to unsafe copy functions (strcpy, strcat, gets, sprintf, vsprintf)
    and replaces them with safe bounded alternatives.

    CRITICAL: Only applies sizeof(dst) when dst is a local ARRAY.
    If dst is a pointer parameter, returns None (escalate to Tier 2 LLM).

    Returns fixed code, or None if no fixable vulnerability found.
    """
    root = _parse(code)
    calls = _find_all(root, "call_expression")

    # Collect function context
    func_node = None
    for call in calls:
        func_node = _find_enclosing_function(call)
        if func_node:
            break
    if func_node is None:
        # Not inside a function — try root-level function_definition
        for child in root.children:
            if child.type == "function_definition":
                func_node = child
                break
    if func_node is None:
        return None

    ptr_params = _get_pointer_params(func_node, code)
    local_arrays = _get_local_arrays(func_node, code)

    edits: list[tuple[int, int, str]] = []
    has_pointer_param_dest = False

    for call in calls:
        fname = _get_call_func_name(call, code)
        if fname not in _UNSAFE_COPY_FUNCS:
            continue

        args = _get_call_args(call)
        if not args:
            continue

        # Get the destination argument name
        if fname == "gets":
            dst_arg = args[0]
        else:
            dst_arg = args[0]  # strcpy, strcat, sprintf all have dst as arg 0

        dst_name = _node_text(dst_arg, code)

        # CRITICAL: check if dst is a pointer param
        if dst_name in ptr_params:
            has_pointer_param_dest = True
            continue  # skip this call site

        # Check if dst is a local array (safe to use sizeof)
        if dst_name not in local_arrays:
            # Unknown — could be global, struct field, etc. Skip to be safe.
            has_pointer_param_dest = True
            continue

        # Find the enclosing expression_statement to replace the whole statement
        stmt = call.parent
        if stmt is None or stmt.type != "expression_statement":
            continue

        # Get indentation from the statement
        line_start = code.rfind("\n", 0, stmt.start_byte)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        indent = ""
        for ch in code[line_start:stmt.start_byte]:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        if fname == "strcpy":
            src_text = _node_text(args[1], code) if len(args) > 1 else "src"
            replacement = (
                f"strncpy({dst_name}, {src_text}, sizeof({dst_name}) - 1);\n"
                f"{indent}{dst_name}[sizeof({dst_name}) - 1] = '\\0';"
            )
        elif fname == "strcat":
            src_text = _node_text(args[1], code) if len(args) > 1 else "src"
            replacement = (
                f"strncat({dst_name}, {src_text}, "
                f"sizeof({dst_name}) - strlen({dst_name}) - 1);"
            )
        elif fname == "gets":
            replacement = f"fgets({dst_name}, sizeof({dst_name}), stdin);"
        elif fname == "sprintf":
            # sprintf(buf, fmt, ...) -> snprintf(buf, sizeof(buf), fmt, ...)
            remaining_args = ", ".join(
                _node_text(a, code) for a in args[1:]
            )
            replacement = (
                f"snprintf({dst_name}, sizeof({dst_name}), {remaining_args});"
            )
        elif fname == "vsprintf":
            remaining_args = ", ".join(
                _node_text(a, code) for a in args[1:]
            )
            replacement = (
                f"vsnprintf({dst_name}, sizeof({dst_name}), {remaining_args});"
            )
        else:
            continue

        # Replace stmt from start_byte (which is after indentation) to end_byte
        edits.append((stmt.start_byte, stmt.end_byte, replacement))

    if not edits:
        if has_pointer_param_dest:
            return None  # escalate to Tier 2
        return None  # no vulnerable calls found

    return _apply_edits(code, edits)


# ── CWE-476: NULL Dereference Fix ────────────────────────────────
def fix_cwe476(code: str) -> str | None:
    """
    Fix CWE-476 NULL pointer dereference vulnerabilities.

    Finds pointer dereferences (*ptr, ptr->field) that are not guarded
    by a NULL check. Inserts a NULL guard before each unguarded dereference.

    The guard's return value matches the enclosing function's return type:
      void   -> return;
      int    -> return -1;
      pointer -> return NULL;

    Returns fixed code, or None if no unguarded dereferences found.
    """
    root = _parse(code)

    # Collect all pointer dereferences: pointer_expression (*ptr)
    # and field_expression with -> (ptr->field)
    deref_sites: list[tuple[Node, str]] = []  # (node, ptr_name)

    for node in _find_all(root, "pointer_expression"):
        # *ptr: pointer_expression has children: *, identifier
        for child in node.children:
            if child.type == "identifier":
                deref_sites.append((node, _node_text(child, code)))
                break

    for node in _find_all(root, "field_expression"):
        # ptr->field: field_expression has children: identifier, ->, field_identifier
        has_arrow = False
        obj_name = None
        for child in node.children:
            if child.type == "->":
                has_arrow = True
            if child.type == "identifier" and not has_arrow:
                obj_name = _node_text(child, code)
        if has_arrow and obj_name:
            deref_sites.append((node, obj_name))

    if not deref_sites:
        return None

    # Get enclosing function for return type
    func_node = None
    for site, _ in deref_sites:
        func_node = _find_enclosing_function(site)
        if func_node:
            break

    ret_type = _get_function_return_type(func_node, code) if func_node else "int"

    # Determine return statement based on return type
    if "void" in ret_type:
        ret_stmt = "return;"
    elif "*" in ret_type:
        ret_stmt = "return NULL;"
    else:
        ret_stmt = "return -1;"

    # Filter to unguarded dereferences only
    unguarded: list[tuple[Node, str]] = []
    seen_ptrs_at_line: set[tuple[str, int]] = set()  # deduplicate same ptr on same line

    for node, ptr_name in deref_sites:
        if _is_inside_null_check(node, ptr_name, code):
            continue
        key = (ptr_name, node.start_point[0])
        if key in seen_ptrs_at_line:
            continue
        seen_ptrs_at_line.add(key)
        unguarded.append((node, ptr_name))

    if not unguarded:
        return None

    # For each unguarded dereference, find the enclosing statement and
    # insert a NULL guard before it.  Group by (stmt_start, ptr_name) to
    # avoid duplicate guards.
    # Key: (stmt_start_byte, ptr_name) → (line_insert_pos, indent_str)
    stmt_guards: dict[tuple[int, str], tuple[int, str]] = {}

    for node, ptr_name in unguarded:
        # Walk up to find the statement (direct child of compound_statement)
        stmt = node
        while stmt.parent is not None and stmt.parent.type != "compound_statement":
            stmt = stmt.parent

        key = (stmt.start_byte, ptr_name)
        if key in stmt_guards:
            continue

        # Compute the start of the line containing this statement
        line_start = code.rfind("\n", 0, stmt.start_byte)
        line_start_pos = (line_start + 1) if line_start >= 0 else 0

        # Extract leading whitespace from that line
        indent = ""
        for ch in code[line_start_pos:]:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        stmt_guards[key] = (line_start_pos, indent)

    # Build edits: insert NULL guard on its own line before the statement.
    # Insert at line_start_pos so the guard gets the same indentation.
    edits: list[tuple[int, int, str]] = []
    for (_, ptr_name), (line_pos, indent) in sorted(stmt_guards.items()):
        guard = f"{indent}if ({ptr_name} == NULL) {{ {ret_stmt} }}\n"
        edits.append((line_pos, line_pos, guard))

    return _apply_edits(code, edits)


# ── Tier 1 entry point ──────────────────────────────────────────
def tier1_generate(code: str, cwe: str, config: object = None) -> list[str]:
    """
    Tier 1 CFA generator. Dispatches to the appropriate AST rule by CWE.

    Returns list of CFA code strings (0 or 1 element).
    Returns [] if the rule does not apply or is unsafe to apply.
    """
    if cwe == "CWE-134":
        result = fix_cwe134(code)
    elif cwe == "CWE-120":
        result = fix_cwe120(code)
    elif cwe == "CWE-476":
        result = fix_cwe476(code)
    else:
        return []

    if result is None:
        return []
    return [result]
