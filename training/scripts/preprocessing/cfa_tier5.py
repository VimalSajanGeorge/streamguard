#!/usr/bin/env python3
"""
training/scripts/preprocessing/cfa_tier5.py

Tier 5: Critique-and-Refine Fallback for StreamGuard CFA pipeline.

Tier 5 is NOT a primary generation strategy.  It activates when any
primary tier (1-4) has accumulated 3+ consecutive failures.  It takes
the best failed CFA draft and a structured critique of WHY it failed,
then asks the LLM to correct only that specific issue.

Exports:
  tier5_refine(original_code, failed_cfa, cwe, failure_reason, config)
      -> dict | None
  inject_feedback(prompt, code, failed_cfa, cwe, reason) -> str

Registration in TIER_GENERATORS[5]:
  The registry is called with the signature
    tier5_fn(original_code, cwe, config, failed_cfa=..., failure_reason=...)
  so the registered wrapper (_tier5_registry_wrapper) adapts to that.

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2.5
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

# Re-use call_llm and extract_c_code_v2 from Tier 2
from training.scripts.preprocessing.cfa_tier2 import (
    call_llm,
    extract_c_code_v2,
)

# Markers
_START_MARKER = "[FIXED_CODE_START]"
_END_MARKER   = "[FIXED_CODE_END]"

# Tier 5 uses DeepSeek V3.2 via OpenRouter (same as all other tiers)
_TIER5_MODEL = "deepseek/deepseek-v3.2"

# ── Per-CWE vuln pattern descriptions (for feedback messages) ────
_CWE_PATTERN_DESCRIPTIONS: dict[str, str] = {
    "CWE-134": "printf/fprintf called with a non-literal format string (e.g. printf(var))",
    "CWE-120": "unsafe string copy: strcpy, strcat, gets, or sprintf without bounds",
    "CWE-476": "pointer dereference (*ptr or ptr->field) without a NULL check",
    "CWE-121": "array write without index bounds check, or strcpy/gets without size limit",
    "CWE-122": "write to malloc'd buffer without size validation or NULL check",
    "CWE-125": "array read at index without bounds check",
    "CWE-89":  "sprintf/strcat of user input directly into SQL query string",
    "CWE-78":  "system/popen/exec called with user-controlled argument",
    "CWE-190": "arithmetic operation (add/mul/shift) without overflow pre-check",
    "CWE-79":  "user input written to output buffer without HTML encoding",
    "CWE-416": "pointer used after free() without re-allocation or NULL assignment",
    "CWE-119": "memory operation (memcpy/strcpy/index) that can exceed buffer bounds",
}

# Per-CWE sink names for taint feedback
_CWE_SINK_NAMES: dict[str, str] = {
    "CWE-89":  "sqlite3_exec / mysql_query (SQL execution sink)",
    "CWE-78":  "system / popen / exec (shell execution sink)",
    "CWE-134": "printf / fprintf (format string output sink)",
    "CWE-79":  "printf / fprintf / HTTP response write (output sink)",
}

# Per-CWE fix descriptions for no_fix_signature feedback
_CWE_FIX_DESCRIPTIONS: dict[str, str] = {
    "CWE-134": 'a format string literal like "%s" or "%d"',
    "CWE-120": "strncpy, strncat, snprintf, or fgets with explicit size",
    "CWE-476": "a NULL check: if (ptr == NULL) { return ...; }",
    "CWE-121": "a bounds check (idx < size) or strncpy/snprintf",
    "CWE-122": "a NULL check after malloc and a size cap before write",
    "CWE-125": "a bounds check before the array read",
    "CWE-89":  "sqlite3_prepare_v2 / sqlite3_bind or mysql_real_escape_string",
    "CWE-78":  "execve with a hardcoded command, or an allowlist check",
    "CWE-190": "INT_MAX - b overflow pre-check or __builtin_add_overflow",
    "CWE-79":  "HTML entity encoding before output",
    "CWE-416": "ptr = NULL after free(), or a NULL guard before use",
    "CWE-119": "a size cap (if len > buf_size) before the memory operation",
}


# ── Feedback templates ────────────────────────────────────────────

def get_feedback(cwe: str, reason: str) -> str:
    """
    Return a specific, actionable feedback string for a given failure reason.

    Substitutes CWE-specific details (pattern description, sink name, fix desc).
    """
    pattern_desc = _CWE_PATTERN_DESCRIPTIONS.get(cwe, f"the {cwe} vulnerability pattern")
    sink_name    = _CWE_SINK_NAMES.get(cwe, "the dangerous sink function")
    fix_desc     = _CWE_FIX_DESCRIPTIONS.get(cwe, f"the {cwe} fix pattern")

    templates: dict[str, str] = {
        "compile_fail": (
            "PREVIOUS ATTEMPT FAILED: The code had a C syntax error. "
            "Ensure: (1) all braces match, (2) no missing semicolons, "
            "(3) all variables declared before use, (4) no undefined functions called. "
            "Generate ONLY syntactically valid C."
        ),
        "vuln_pattern_remains": (
            f"PREVIOUS ATTEMPT FAILED: The {cwe} vulnerability pattern is still present. "
            f"You MUST remove: {pattern_desc}. "
            "The fix must make this pattern completely absent from the function."
        ),
        "taint_path_intact": (
            "PREVIOUS ATTEMPT FAILED: User input still flows to the dangerous sink. "
            f"You must BREAK the data flow path from user-controlled input to {sink_name}. "
            "Either sanitize/validate the input OR replace the sink with a safe alternative."
        ),
        "ptr_not_nulled": (
            "PREVIOUS ATTEMPT FAILED: After free(ptr), ptr = NULL is missing. "
            "Immediately after EVERY free() call, add: ptr = NULL; on the very next line."
        ),
        "no_fix_signature": (
            "PREVIOUS ATTEMPT may have removed the vulnerability but lacks the expected fix. "
            f"The fix must still contain {fix_desc}. "
            "Do not produce a trivially altered or empty function."
        ),
        "too_similar": (
            "PREVIOUS ATTEMPT made no meaningful change to address the vulnerability. "
            f"A correct fix MUST change at least 1-3 lines. Apply the actual {cwe} fix pattern."
        ),
        "too_different": (
            "PREVIOUS ATTEMPT changed too much. "
            "Change ONLY the vulnerability site. Keep all other logic IDENTICAL to the original."
        ),
        "identical_to_original": (
            "PREVIOUS ATTEMPT returned the same code as the original — no fix was applied. "
            f"You MUST modify the function to address {cwe}."
        ),
    }
    return templates.get(reason, f"PREVIOUS ATTEMPT FAILED for reason: {reason}. Try a different approach.")


# ── FEEDBACK_TEMPLATES (public alias for test introspection) ──────
FEEDBACK_TEMPLATES = {
    "compile_fail":         get_feedback("CWE-120", "compile_fail"),
    "vuln_pattern_remains": "PREVIOUS ATTEMPT FAILED: The {cwe} vulnerability pattern is still present.",
    "taint_path_intact":    "PREVIOUS ATTEMPT FAILED: User input still flows to the dangerous sink.",
    "ptr_not_nulled":       get_feedback("CWE-416", "ptr_not_nulled"),
    "no_fix_signature":     "PREVIOUS ATTEMPT may have removed the vulnerability but lacks the expected fix.",
    "too_similar":          get_feedback("CWE-120", "too_similar"),
    "too_different":        get_feedback("CWE-120", "too_different"),
}


# ── inject_feedback ───────────────────────────────────────────────

def inject_feedback(
    prompt: str,
    code: str,
    failed_cfa: str,
    cwe: str,
    reason: str,
) -> str:
    """
    Append a 'PREVIOUS ATTEMPT NOTES:' section to an existing prompt.

    Used by Tier 2/3/4 generators for lightweight in-prompt feedback
    (lighter weight than a full tier5_refine call).
    """
    feedback = get_feedback(cwe, reason)
    notes = (
        f"\n\nPREVIOUS ATTEMPT NOTES:\n{feedback}\n"
        "Please correct only this specific issue in your next attempt."
    )
    return prompt + notes


# ── tier5_refine ──────────────────────────────────────────────────

def tier5_refine(
    original_code: str,
    failed_cfa: str,
    cwe: str,
    failure_reason: str,
    config: Any = None,
) -> dict | None:
    """
    Tier 5 critique-and-refine.

    Builds a refinement prompt showing:
      1. The original vulnerable code
      2. The failed CFA attempt
      3. Specific feedback about WHY it failed
      4. Instructions to fix ONLY the specific issue

    Returns:
      {'cfa_code': str, 'tier': 5, 'quality_score': float} if refinement succeeds
      None if the refinement also fails (no infinite loop)

    Config attributes used:
      config.openrouter_api_key str    None
      config.llm_timeout        float  60.0
    """
    from training.scripts.preprocessing.stage3_cfa import validate_cfa_v2

    feedback = get_feedback(cwe, failure_reason)

    failed_block = (
        failed_cfa.strip() if failed_cfa else "[generation failed entirely]"
    )

    prompt = (
        f"ORIGINAL VULNERABLE CODE ({cwe}):\n"
        f"```c\n{original_code}\n```\n\n"
        f"PREVIOUS FIX ATTEMPT (INCORRECT):\n"
        f"```c\n{failed_block}\n```\n\n"
        f"CRITIQUE: {feedback}\n\n"
        f"Generate a CORRECTED fix that addresses the critique above.\n"
        f"Apply ONLY the minimum change to fix {cwe}. Keep all other logic IDENTICAL.\n"
        f"{_START_MARKER}corrected_function_here{_END_MARKER}"
    )

    # Model: prefer config override, then fall back to haiku
    model = _TIER5_MODEL
    api_key = getattr(config, "openrouter_api_key", None)
    llm_timeout = getattr(config, "llm_timeout", 60.0)

    raw = call_llm(
        prompt,
        max_tokens=1200,
        api_key=api_key,
        timeout=llm_timeout,
        attempt_index=1,  # kept for call-site compat
    )
    if raw is None:
        logger.debug(f"Tier 5 [{cwe}]: LLM returned None")
        return None

    cfa_code = extract_c_code_v2(raw)
    if not cfa_code:
        logger.debug(f"Tier 5 [{cwe}]: empty extraction")
        return None

    is_valid, reason, cleaned, base_score = validate_cfa_v2(
        original_code, cfa_code, cwe, tier=5, config=config
    )

    if is_valid:
        quality_score = round(base_score * 0.85, 4)
        logger.debug(
            f"Tier 5 [{cwe}]: refinement ACCEPTED "
            f"(base_score={base_score:.2f}, final={quality_score:.4f})"
        )
        return {"cfa_code": cleaned, "tier": 5, "quality_score": quality_score}

    logger.debug(f"Tier 5 [{cwe}]: refinement failed ({reason}) — no retry")
    return None


# ── Registry wrapper ─────────────────────────────────────────────
# TIER_GENERATORS[5] is called by _tier5_refine in cfa_tier2.py as:
#   tier5_fn(original_code, cwe, config, failed_cfa=..., failure_reason=...)
# This wrapper adapts that signature to tier5_refine's positional signature.

def _tier5_registry_wrapper(
    original_code: str,
    cwe: str,
    config: Any,
    failed_cfa: str = "",
    failure_reason: str = "compile_fail",
) -> list[str]:
    """
    Adapter so TIER_GENERATORS[5] can be called by the orchestrator.

    Returns a list (possibly empty) for consistency with other tiers.
    The raw dict result is unwrapped by the calling _tier5_refine stub.
    """
    result = tier5_refine(original_code, failed_cfa, cwe, failure_reason, config)
    if result:
        return [result["cfa_code"]]
    return []
