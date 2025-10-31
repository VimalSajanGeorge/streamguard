"""
Simple code metrics extraction (no external dependencies).

Fast pattern-based + AST extraction for security-relevant features.
Designed to catch SQL injection and other vulnerability signals.
"""

import re
import ast
from typing import Dict


def extract_basic_metrics(code: str) -> Dict[str, int]:
    """
    Extract 10 basic security-relevant metrics from code.

    Fast extraction without heavy dependencies (no radon needed).
    Combines pattern matching and AST analysis.

    Args:
        code: Source code string

    Returns:
        Dictionary with 10 metrics:
        - loc: Lines of code
        - sloc: Source lines of code (non-blank, non-comment)
        - sql_concat: SQL string concatenations (vulnerability signal)
        - execute_calls: Database execute/query calls
        - user_input: User input functions (request., input(), GET[], etc.)
        - loops: Number of loops (for, while)
        - conditionals: Number of if statements
        - function_calls: Total function calls
        - try_blocks: Exception handling blocks
        - string_ops: String operations (Add, Mult)

    Example:
        >>> code = "query = 'SELECT * FROM users WHERE id=' + user_id"
        >>> metrics = extract_basic_metrics(code)
        >>> metrics['sql_concat']  # Should be 1
        1
        >>> metrics['execute_calls']  # Should be 0
        0
    """
    lines = code.split('\n')

    # Basic counts
    loc = len(lines)
    sloc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])

    # Pattern-based detection (SQL injection signals)
    # Matches: "..." + "..." or f"...{...}..." (SQL concatenation)
    sql_concat = len(re.findall(r'["\'].*\+.*["\']|f["\'].*\{.*\}.*["\']', code))

    # Database execute/query calls
    execute_calls = len(re.findall(r'\.(execute|query|raw|exec)\s*\(', code))

    # User input sources (vulnerability entry points)
    user_input = len(re.findall(r'(request\.|input\s*\(|GET\[|POST\[|params\[)', code))

    # AST-based features (if parseable)
    try:
        tree = ast.parse(code)

        loops = len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))])
        conditionals = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
        function_calls = len([n for n in ast.walk(tree) if isinstance(n, ast.Call)])
        try_blocks = len([n for n in ast.walk(tree) if isinstance(n, ast.Try)])

        # String operations (concatenation, multiplication)
        string_ops = len([
            n for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mult))
        ])

    except SyntaxError:
        # Fallback: code not parseable (non-Python or syntax errors)
        loops = conditionals = function_calls = try_blocks = string_ops = 0

    # Return dictionary with capped values (prevent outliers)
    return {
        'loc': min(loc, 1000),            # Cap at 1000 lines
        'sloc': min(sloc, 1000),
        'sql_concat': min(sql_concat, 50),
        'execute_calls': min(execute_calls, 20),
        'user_input': min(user_input, 20),
        'loops': min(loops, 50),
        'conditionals': min(conditionals, 50),
        'function_calls': min(function_calls, 100),
        'try_blocks': min(try_blocks, 20),
        'string_ops': min(string_ops, 50)
    }


# Simple test
if __name__ == '__main__':
    # Test 1: SQL injection vulnerable code
    vulnerable_code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id=" + user_id
    cursor.execute(query)
    return cursor.fetchone()
"""

    metrics = extract_basic_metrics(vulnerable_code)
    print("Vulnerable code metrics:")
    print(f"  SQL concatenations: {metrics['sql_concat']}")
    print(f"  Execute calls: {metrics['execute_calls']}")
    print(f"  Function calls: {metrics['function_calls']}")

    # Test 2: Safe code
    safe_code = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id=?"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
"""

    metrics = extract_basic_metrics(safe_code)
    print("\nSafe code metrics:")
    print(f"  SQL concatenations: {metrics['sql_concat']}")
    print(f"  Execute calls: {metrics['execute_calls']}")
    print(f"  Function calls: {metrics['function_calls']}")
