#!/usr/bin/env python3
"""
training/scripts/preprocessing/cfa_tier3.py

Tier 3: Chain-of-Thought LLM CFA Generation for StreamGuard pipeline.

Handles CWE-89 (SQL injection), CWE-78 (OS command injection),
CWE-190 (integer overflow), CWE-79 (XSS/output injection in C).

Uses OpenRouter API (same client as Tier 2) with CoT prompts that ask the
model to reason through IDENTIFY → STRATEGY → GENERATE in 3 explicit steps.

Gate 6 (taint path broken) is active for CWE-89, CWE-78, CWE-79.
When Gate 6 fails, the failure feedback is injected into the NEXT attempt's
prompt so the model can self-correct.

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2.3, §3.3
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

# Re-use the OpenRouter call_llm and _tier5_refine from Tier 2
from training.scripts.preprocessing.cfa_tier2 import (
    _DEFAULT_MODEL,
    _tier5_refine,
    call_llm,
)

# ── Markers ──────────────────────────────────────────────────────
_START_MARKER = "[FIXED_CODE_START]"
_END_MARKER   = "[FIXED_CODE_END]"

# Patterns indicating start of a C function (shared with tier2)
_C_FUNC_START = re.compile(
    r'^(static\s+|inline\s+|extern\s+)?(const\s+)?'
    r'(void|int|long|char|short|unsigned|struct|enum|float|double|\w+)\s+\**\w+\s*\(',
    re.MULTILINE,
)

# ── Gate 6 taint feedback messages ───────────────────────────────
_GATE6_FEEDBACK: dict[str, str] = {
    "CWE-89": (
        "FEEDBACK: Your previous fix STILL has user input flowing into the SQL query "
        "without parameterization. The taint path (user_input → query string → SQL exec) "
        "is INTACT. You MUST use a parameterized query (sqlite3_prepare_v2 + sqlite3_bind_*, "
        "or mysql_real_escape_string) so that user input is never concatenated directly "
        "into the query string."
    ),
    "CWE-78": (
        "FEEDBACK: Your previous fix STILL has user input flowing directly into a shell "
        "command (system/popen/exec call). The taint path is INTACT. You MUST either: "
        "(1) remove user input from the command entirely and pass it as a DATA argument, "
        "or (2) add a strict allowlist check that rejects non-whitelisted input BEFORE "
        "any exec call."
    ),
    "CWE-79": (
        "FEEDBACK: Your previous fix STILL writes user input directly to output without "
        "encoding. The taint path is INTACT. You MUST apply HTML entity encoding to ALL "
        "user-controlled values before they appear in any output buffer or response."
    ),
}

# ── Tier 3 CoT prompts ───────────────────────────────────────────
TIER3_COT_CWE89 = (
    "You are fixing SQL INJECTION (CWE-89) in a C function.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find the line where user-controlled input flows into a SQL\n"
    "  query string. Look for: sprintf(query,...,user_input), strcat(query, user_input),\n"
    "  direct string concatenation into a variable used in mysql_query/sqlite3_exec.\n"
    "STEP 2 - STRATEGY: Choose the correct parameterization:\n"
    "  - For SQLite: replace with sqlite3_prepare_v2(db, sql_with_?, -1, &stmt, 0)\n"
    "    then sqlite3_bind_text(stmt, 1, user_input, -1, SQLITE_STATIC)\n"
    "  - For MySQL: replace with mysql_stmt_prepare() + mysql_stmt_bind_param()\n"
    "  - If DB API unavailable: use mysql_real_escape_string() on ALL user inputs\n"
    "    before they enter the query. Do NOT just add quotes around user input.\n"
    "STEP 3 - Generate the fixed function.\n"
    "CONSTRAINTS: Change ONLY the query construction lines. SQL logic (SELECT/INSERT)\n"
    "must remain identical. Do NOT change function signature.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

TIER3_COT_CWE78 = (
    "You are fixing OS COMMAND INJECTION (CWE-78) in a C function.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find system(user_input), popen(user_input,...),\n"
    "  execl/execlp/execv/execvp with user-controlled argument.\n"
    "STEP 2 - STRATEGY:\n"
    "  - Replace system(cmd) with: execve(\"/bin/sh\", (char*[]){cmd, NULL}, environ)\n"
    "    where cmd is a HARDCODED command string (not user input)\n"
    "  - Pass user input as a DATA argument to the fixed command, not as the command itself\n"
    "  - If user input must be part of command: add strict allowlist check first:\n"
    "    if (!is_allowed_input(user_input)) { return -1; }\n"
    "STEP 3 - Generate the fixed function.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

TIER3_COT_CWE190 = (
    "You are fixing INTEGER OVERFLOW (CWE-190) in a C function.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find the arithmetic operation that can overflow:\n"
    "  - Addition: a + b where a,b can be close to INT_MAX\n"
    "  - Multiplication: a * b where result can exceed INT_MAX\n"
    "  - Left shift: a << b where result can exceed bit width\n"
    "  - Size calculation: malloc(a * b) where a*b can overflow\n"
    "STEP 2 - STRATEGY: Add a pre-check IMMEDIATELY before the operation:\n"
    "  - Addition: if (a > INT_MAX - b) { return ERROR_CODE; }\n"
    "  - Multiplication: if (b != 0 && a > INT_MAX / b) { return ERROR_CODE; }\n"
    "  - Or use: __builtin_add_overflow(a, b, &result) which returns 1 if overflow\n"
    "  - Use the SAME error return value used elsewhere in the function.\n"
    "STEP 3 - Generate the fixed function.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

TIER3_COT_CWE79 = (
    "You are fixing OUTPUT INJECTION / XSS (CWE-79) in a C function.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find where user-controlled input is written directly to\n"
    "  an output buffer or HTTP response without encoding:\n"
    "  sprintf(response, ..., user_input), fprintf(out, user_input), strcat(buf, user_input).\n"
    "STEP 2 - STRATEGY: Apply HTML entity encoding to user input BEFORE output:\n"
    "  - Replace < with &lt;, > with &gt;, & with &amp;, \" with &quot;, ' with &#39;\n"
    "  - Add a helper function: void html_encode(const char *src, char *dst, size_t dst_size)\n"
    "    that performs these substitutions.\n"
    "  - Call html_encode(user_input, encoded, sizeof(encoded)) then use encoded in output.\n"
    "STEP 3 - Generate the fixed function (include html_encode helper if needed).\n"
    "CONSTRAINTS: Do NOT change function signature. Output must still be produced.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

TIER3_COT_CWE119 = (
    "You are fixing IMPROPER MEMORY BOUNDS (CWE-119) in a C function.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find the memory operation that exceeds buffer bounds:\n"
    "  - memcpy/memmove/memset with size > allocated buffer\n"
    "  - strcpy/strcat without length limit\n"
    "  - Direct array index arr[i] where i can exceed array size\n"
    "  - Pointer arithmetic ptr + offset beyond allocation\n"
    "STEP 2 - STRATEGY:\n"
    "  - For memcpy/memset: add size check before copy, cap at buffer size\n"
    "    if (len > buf_size) { len = buf_size; }\n"
    "  - For strcpy: replace with strncpy(dst, src, sizeof(dst)-1); dst[sizeof(dst)-1]='\\0';\n"
    "  - For array index: add bounds check: if (idx >= array_len) { return -1; }\n"
    "  - Use sizeof() for stack arrays, passed-in size param for heap buffers.\n"
    "STEP 3 - Generate the fixed function.\n"
    "CONSTRAINTS: Do NOT change function signature. Do NOT add includes.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

TIER3_COT_PROMPTS: dict[str, str] = {
    "CWE-89":  TIER3_COT_CWE89,
    "CWE-78":  TIER3_COT_CWE78,
    "CWE-190": TIER3_COT_CWE190,
    "CWE-79":  TIER3_COT_CWE79,
    "CWE-119": TIER3_COT_CWE119,
}

# CWEs for which Gate 6 feedback is injected on taint failure
_GATE6_CWES = {"CWE-89", "CWE-78", "CWE-79"}


# ── Code extraction (handles both [START]...[END] and fallbacks) ──
def extract_c_code_tier3(response: str) -> str:
    """
    Extract C code from a Tier 3 CoT response.

    Cascade:
    1. [FIXED_CODE_START] ... [FIXED_CODE_END] — extract between markers
    2. [FIXED_CODE_START] only — take everything after it (no end marker)
    3. ```c ... ``` markdown fence
    4. First C function signature line to end of response
    5. Raw response stripped (fallback)

    The [FIXED_CODE_END] marker is critical for CoT: it prevents reasoning
    text that follows the code block from being included in the CFA.
    """
    if not response:
        return ""

    start_pos = response.find(_START_MARKER)
    end_pos   = response.find(_END_MARKER)

    # Stage 1: both markers present — extract between them
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        inner = response[start_pos + len(_START_MARKER):end_pos].strip()
        # Strip "fixed_code_here" placeholder if LLM echoed the template
        if inner.lower() == "fixed_code_here":
            inner = ""
        if inner:
            return inner

    # Stage 2: only START marker — take everything after
    if start_pos != -1:
        candidate = response[start_pos + len(_START_MARKER):].strip()
        # Remove end marker if present at end
        if _END_MARKER in candidate:
            candidate = candidate[:candidate.find(_END_MARKER)].strip()
        if candidate and candidate.lower() != "fixed_code_here":
            return candidate

    # Stage 3: markdown fence
    m = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Stage 4: first C function signature
    m = _C_FUNC_START.search(response)
    if m:
        return response[m.start():].strip()

    # Stage 5: raw fallback
    return response.strip()


# ── Tier 3 entry point ────────────────────────────────────────────
def tier3_generate(
    code: str,
    cwe: str,
    config: Any = None,
    exemplar_db: dict | None = None,
) -> list[str]:
    """
    Tier 3 CFA generator — Chain-of-Thought LLM prompting via DeepSeek API.

    Returns list of validated CFA code strings.

    Config attributes used (with defaults):
      config.max_attempts_tier3     int   7
      config.target_ratio           int   1
      config.openrouter_api_key     str   None
      config.llm_timeout            float 60.0
    """
    from training.scripts.preprocessing.stage3_cfa import validate_cfa_v2

    prompt_template = TIER3_COT_PROMPTS.get(cwe)
    if prompt_template is None:
        logger.warning(f"Tier 3: no CoT prompt for {cwe}")
        return []

    max_attempts  = getattr(config, "max_attempts_tier3", 7)
    target_ratio  = getattr(config, "target_ratio", 1)
    api_key       = getattr(config, "openrouter_api_key", None)
    llm_timeout   = getattr(config, "llm_timeout", 60.0)

    results: list[str] = []
    consecutive_compile_fails = 0
    last_failed_cfa = ""
    gate6_feedback  = ""   # injected into next attempt when Gate 6 fails

    for attempt in range(max_attempts):
        if len(results) >= target_ratio:
            break

        # Build prompt — prepend gate6 feedback if available
        if gate6_feedback:
            full_prompt = (
                f"VULNERABLE C FUNCTION:\n```c\n{code}\n```\n\n"
                f"{gate6_feedback}\n\nTASK:\n{prompt_template}"
            )
            gate6_feedback = ""  # consume feedback
        else:
            full_prompt = (
                f"VULNERABLE C FUNCTION:\n```c\n{code}\n```\n\nTASK:\n{prompt_template}"
            )

        raw = call_llm(
            full_prompt,
            max_tokens=1536,   # CoT needs more tokens than Tier 2
            api_key=api_key,
            timeout=llm_timeout,
            attempt_index=attempt,
        )
        if raw is None:
            logger.debug(f"Tier 3 [{cwe}] attempt {attempt+1}: LLM returned None")
            continue

        cfa_code = extract_c_code_tier3(raw)
        if not cfa_code:
            logger.debug(f"Tier 3 [{cwe}] attempt {attempt+1}: empty extraction")
            continue

        is_valid, reason, cleaned, score = validate_cfa_v2(
            code, cfa_code, cwe, tier=3, config=config
        )

        if is_valid:
            results.append(cleaned)
            consecutive_compile_fails = 0
            logger.debug(
                f"Tier 3 [{cwe}] attempt {attempt+1}: ACCEPTED (score={score:.2f})"
            )
        else:
            logger.debug(
                f"Tier 3 [{cwe}] attempt {attempt+1}: rejected ({reason})"
            )

            if reason == "compile_fail":
                consecutive_compile_fails += 1
                last_failed_cfa = cfa_code
                # 3 consecutive compile fails → Tier 5 refinement
                if consecutive_compile_fails >= 3:
                    logger.debug(
                        f"Tier 3 [{cwe}]: 3 consecutive compile fails, "
                        "trying Tier 5 refinement"
                    )
                    t5 = _tier5_refine(code, last_failed_cfa, cwe, reason, config)
                    if t5:
                        results.append(t5["cfa_code"])
                    consecutive_compile_fails = 0
            else:
                consecutive_compile_fails = 0

            # Gate 6 failure: inject taint-path feedback into NEXT attempt
            if reason == "taint_path_intact" and cwe in _GATE6_CWES:
                gate6_feedback = _GATE6_FEEDBACK.get(cwe, "")

    return results
