#!/usr/bin/env python3
"""
training/scripts/preprocessing/cfa_tier4.py

Tier 4: Few-Shot Exemplar + Chain-of-Thought LLM CFA Generation.

Handles CWE-416 (use-after-free) and CWE-119 (improper memory bounds).

Strategy:
  1. Select up to 2 exemplar pairs from ExemplarDatabase (keyword-ranked)
  2. Build a prompt: exemplars first, then CoT instruction, then target code
  3. Call DeepSeek API (up to max_attempts_tier4 = 10)
  4. Extract C code from response
  5. Run validate_cfa_v2() 7-gate check
  6. After attempt 4 (if still no result): inject "think harder" feedback
  7. After 3 consecutive compile fails: escalate to Tier 5 refinement

CWE-416 gets an extra structural check (validate_cwe416_structural) after
the 7 gates to verify the free() call was actually removed or the pointer
was nulled/re-checked.

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2.4, §3.4
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

# Re-use OpenRouter call_llm and _tier5_refine from Tier 2
from training.scripts.preprocessing.cfa_tier2 import (
    _DEFAULT_MODEL,
    _tier5_refine,
    call_llm,
)

# ── Markers ──────────────────────────────────────────────────────
_START_MARKER = "[FIXED_CODE_START]"
_END_MARKER   = "[FIXED_CODE_END]"

# Patterns indicating start of a C function (same as tier3)
_C_FUNC_START = re.compile(
    r'^(static\s+|inline\s+|extern\s+)?(const\s+)?'
    r'(void|int|long|char|short|unsigned|struct|enum|float|double|\w+)\s+\**\w+\s*\(',
    re.MULTILINE,
)

# ── Tier 4 CoT prompt templates ──────────────────────────────────

# {exemplar_block} is filled in at call time by build_tier4_prompt()
_TIER4_COT_CWE416 = (
    "You are fixing USE-AFTER-FREE (CWE-416) in a C function.\n"
    "Here are examples of how similar vulnerabilities were fixed:\n"
    "EXEMPLAR_BLOCK_PLACEHOLDER\n"
    "Now fix the target function using the same pattern.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find the pointer that is free()'d and then used again.\n"
    "  Look for: free(ptr) followed by ptr->field, *ptr, or ptr passed to another\n"
    "  function WITHOUT re-allocation or NULL assignment in between.\n"
    "STEP 2 - STRATEGY: Choose ONE fix:\n"
    "  Option A (NULL after free): After free(ptr), add: ptr = NULL;\n"
    "    Then add NULL check before any subsequent use: if (ptr != NULL) ...\n"
    "  Option B (remove redundant free): If ptr is freed twice, remove the FIRST\n"
    "    free() call and keep the last one.\n"
    "  Option C (defer free): Move the free(ptr) call to AFTER the last use of ptr.\n"
    "STEP 3 - Generate the fixed function.\n"
    "CONSTRAINTS: Do NOT change the function signature or return type.\n"
    "Do NOT add includes. Output ONLY the fixed C function.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

_TIER4_COT_CWE119 = (
    "You are fixing IMPROPER MEMORY BOUNDS (CWE-119) in a C function.\n"
    "Here are examples of how similar vulnerabilities were fixed:\n"
    "EXEMPLAR_BLOCK_PLACEHOLDER\n"
    "Now fix the target function using the same pattern.\n"
    "Think step by step:\n"
    "STEP 1 - IDENTIFY: Find the memory operation that exceeds the buffer bounds:\n"
    "  - memcpy/memmove/memset with size > allocated buffer\n"
    "  - strcpy/strcat without length limit\n"
    "  - Direct array index access arr[i] where i can exceed array size\n"
    "STEP 2 - STRATEGY:\n"
    "  - For memcpy/memset: add size check before copy, cap at buffer size\n"
    "  - For strcpy: replace with strncpy + explicit null terminator\n"
    "  - For array index: add bounds check before access\n"
    "  - Use sizeof() for compile-time-known arrays, passed-in size param otherwise.\n"
    "STEP 3 - Generate the fixed function.\n"
    "CONSTRAINTS: Do NOT change the function signature. Do NOT add includes.\n"
    f"{_START_MARKER}fixed_code_here{_END_MARKER}"
)

TIER4_COT_PROMPTS: dict[str, str] = {
    "CWE-416": _TIER4_COT_CWE416,
    "CWE-119": _TIER4_COT_CWE119,
}

# Feedback injected after attempt 4 when no valid CFA found yet
_THINK_HARDER_FEEDBACK = (
    "FEEDBACK: Your previous attempts did not produce a valid fixed function. "
    "Think more carefully about the exact vulnerability pattern. "
    "Re-read the exemplars above and apply the SAME structural fix pattern. "
    "Make sure your fix: (1) compiles as valid C, (2) preserves the function "
    "signature exactly, (3) clearly removes the vulnerability, "
    "(4) uses only the markers [{start}] and [{end}] to delimit your code."
).format(start=_START_MARKER, end=_END_MARKER)


# ── Exemplar block formatter ──────────────────────────────────────

def build_tier4_prompt(
    code: str,
    cwe: str,
    exemplars: list[dict[str, str]],
    extra_feedback: str = "",
) -> str:
    """
    Build the full Tier 4 few-shot CoT prompt.

    ``exemplars`` is a list of {"vuln": str, "safe": str} dicts.
    ``extra_feedback`` is prepended when injecting think-harder or Gate-6 feedback.
    """
    template = TIER4_COT_PROMPTS.get(cwe, "")
    if not template:
        return ""

    # Format exemplar block
    if exemplars:
        parts = []
        for i, ex in enumerate(exemplars, 1):
            parts.append(
                f"--- Example {i} ---\n"
                f"VULNERABLE:\n```c\n{ex['vuln'].strip()}\n```\n"
                f"FIXED:\n```c\n{ex['safe'].strip()}\n```"
            )
        exemplar_block = "\n\n".join(parts)
    else:
        exemplar_block = "(No exemplars available — use your general knowledge.)"

    task_section = template.replace("EXEMPLAR_BLOCK_PLACEHOLDER", exemplar_block)

    feedback_part = f"{extra_feedback}\n\n" if extra_feedback else ""
    return (
        f"VULNERABLE C FUNCTION:\n```c\n{code}\n```\n\n"
        f"{feedback_part}"
        f"TASK:\n{task_section}"
    )


# ── CWE-416 structural post-check ────────────────────────────────

def validate_cwe416_structural(original: str, cfa_code: str) -> bool:
    """
    Post-gate structural check for CWE-416 (use-after-free).

    Returns True if ANY of:
      (a) the CFA contains "ptr = NULL" anywhere (null-after-free fix)
      (b) the number of free() calls in CFA < number in original (removed a free)
      (c) a NULL guard on pointer use is present (if (ptr != NULL) / if (ptr == NULL))
      (d) free() position changed — last free() appears later in the CFA than in
          the original relative to total line count (deferred free)
      (e) realloc replaces a free+malloc sequence

    False means the fix looks structurally insufficient.
    """
    # Count free() calls
    original_frees = len(re.findall(r'\bfree\s*\(', original))
    cfa_frees = len(re.findall(r'\bfree\s*\(', cfa_code))

    if cfa_frees < original_frees:
        return True  # Option B: removed a redundant free

    # Check for any "= NULL" assignment (null-after-free or null-before-use)
    if re.search(r'=\s*NULL\s*;', cfa_code):
        return True  # Option A: pointer nulled

    # Check for NULL guard on pointer use
    if re.search(r'if\s*\(\s*\w+\s*(!=|==)\s*NULL\s*\)', cfa_code):
        return True  # Option A or C: guarded use

    # Option D: deferred free — free() moved later relative to code length
    if original_frees > 0 and cfa_frees > 0:
        orig_lines = original.splitlines()
        cfa_lines = cfa_code.splitlines()
        orig_len = max(len(orig_lines), 1)
        cfa_len = max(len(cfa_lines), 1)

        # Find the line index of the last free() in each
        orig_last_free = max(
            i for i, ln in enumerate(orig_lines)
            if re.search(r'\bfree\s*\(', ln)
        )
        cfa_last_free = max(
            i for i, ln in enumerate(cfa_lines)
            if re.search(r'\bfree\s*\(', ln)
        )

        # Compare normalised position (0.0 = top, 1.0 = bottom)
        orig_pos = orig_last_free / orig_len
        cfa_pos = cfa_last_free / cfa_len
        if cfa_pos > orig_pos + 0.05:
            return True  # free moved down (deferred)

    # Option E: realloc used in CFA where original had free
    if re.search(r'\brealloc\s*\(', cfa_code) and not re.search(
        r'\brealloc\s*\(', original
    ):
        return True

    return False


# ── Code extraction ───────────────────────────────────────────────

def extract_c_code_tier4(response: str) -> str:
    """
    Extract C code from a Tier 4 few-shot CoT response.

    Cascade:
    1. [FIXED_CODE_START] ... [FIXED_CODE_END] markers
    2. [FIXED_CODE_START] only (no end marker)
    3. ```c ... ``` markdown fence
    4. First C function signature line to end of response
    5. Raw response stripped (fallback)
    """
    if not response:
        return ""

    start_pos = response.find(_START_MARKER)
    end_pos   = response.find(_END_MARKER)

    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        inner = response[start_pos + len(_START_MARKER):end_pos].strip()
        if inner.lower() == "fixed_code_here":
            inner = ""
        if inner:
            return inner

    if start_pos != -1:
        candidate = response[start_pos + len(_START_MARKER):].strip()
        if _END_MARKER in candidate:
            candidate = candidate[:candidate.find(_END_MARKER)].strip()
        if candidate and candidate.lower() != "fixed_code_here":
            return candidate

    m = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = _C_FUNC_START.search(response)
    if m:
        return response[m.start():].strip()

    return response.strip()


# ── Tier 4 entry point ───────────────────────────────────────────

def tier4_generate(
    code: str,
    cwe: str,
    config: Any = None,
    exemplar_db: Any = None,
) -> list[str]:
    """
    Tier 4 CFA generator — Few-shot exemplar + CoT via DeepSeek API.

    Returns list of validated CFA code strings.

    Config attributes used (with defaults):
      config.max_attempts_tier4     int   10
      config.target_ratio           int   1
      config.openrouter_api_key     str   None
      config.llm_timeout            float 60.0
    """
    from training.scripts.preprocessing.stage3_cfa import validate_cfa_v2

    prompt_template = TIER4_COT_PROMPTS.get(cwe)
    if prompt_template is None:
        logger.warning(f"Tier 4: no CoT prompt for {cwe}")
        return []

    # Skip if no exemplars available for this CWE — sending zero-shot to a
    # few-shot prompt template produces broken C code every time (wasted API calls)
    if exemplar_db is not None and hasattr(exemplar_db, 'get'):
        if not exemplar_db.get(cwe):
            logger.debug(f"Tier 4 [{cwe}]: no exemplars in DB — skipping to avoid wasted calls")
            return []

    max_attempts = getattr(config, "max_attempts_tier4", 10)
    target_ratio = getattr(config, "target_ratio", 1)
    api_key      = getattr(config, "openrouter_api_key", None)
    llm_timeout  = getattr(config, "llm_timeout", 60.0)

    # Retrieve exemplars (up to 2, keyword-ranked)
    exemplars: list[dict[str, str]] = []
    if exemplar_db is not None and hasattr(exemplar_db, "select_exemplars"):
        exemplars = exemplar_db.select_exemplars(cwe, code, n=2)

    results: list[str] = []
    consecutive_compile_fails = 0
    last_failed_cfa = ""
    extra_feedback = ""   # set after attempt 4 if still no result

    for attempt in range(max_attempts):
        if len(results) >= target_ratio:
            break

        # After attempt index 3 (i.e., 4 attempts done) with no result, inject feedback
        if attempt == 4 and not results:
            extra_feedback = _THINK_HARDER_FEEDBACK

        full_prompt = build_tier4_prompt(
            code, cwe, exemplars, extra_feedback=extra_feedback
        )
        # Consume feedback after it's been used once
        if extra_feedback and attempt > 4:
            extra_feedback = ""

        if not full_prompt:
            break

        raw = call_llm(
            full_prompt,
            max_tokens=1536,
            api_key=api_key,
            timeout=llm_timeout,
            attempt_index=attempt,
        )
        if raw is None:
            logger.debug(f"Tier 4 [{cwe}] attempt {attempt+1}: LLM returned None")
            continue

        cfa_code = extract_c_code_tier4(raw)
        if not cfa_code:
            logger.debug(f"Tier 4 [{cwe}] attempt {attempt+1}: empty extraction")
            continue

        is_valid, reason, cleaned, score = validate_cfa_v2(
            code, cfa_code, cwe, tier=4, config=config
        )

        if is_valid:
            # CWE-416: extra structural check after the 7 gates
            if cwe == "CWE-416":
                structurally_ok = validate_cwe416_structural(code, cleaned)
                if not structurally_ok:
                    logger.debug(
                        f"Tier 4 [{cwe}] attempt {attempt+1}: "
                        "REJECTED by structural CWE-416 check"
                    )
                    continue

            results.append(cleaned)
            consecutive_compile_fails = 0
            logger.debug(
                f"Tier 4 [{cwe}] attempt {attempt+1}: ACCEPTED (score={score:.2f})"
            )
        else:
            logger.debug(
                f"Tier 4 [{cwe}] attempt {attempt+1}: rejected ({reason})"
            )

            if reason == "compile_fail":
                consecutive_compile_fails += 1
                last_failed_cfa = cfa_code
                if consecutive_compile_fails >= 3:
                    logger.debug(
                        f"Tier 4 [{cwe}]: 3 consecutive compile fails, "
                        "trying Tier 5 refinement"
                    )
                    t5 = _tier5_refine(code, last_failed_cfa, cwe, reason, config)
                    if t5:
                        results.append(t5["cfa_code"])
                    consecutive_compile_fails = 0
            else:
                consecutive_compile_fails = 0

    return results
