#!/usr/bin/env python3
"""
training/scripts/preprocessing/cfa_tier2.py

Tier 2: Zero-Shot LLM CFA Generation for StreamGuard pipeline.

Handles CWE-121 (stack buffer overflow), CWE-122 (heap buffer overflow),
CWE-125 (out-of-bounds read).

Uses OpenRouter API (OpenAI-compatible REST) via httpx.
API key: OPENROUTER_API_KEY environment variable.
Model: deepseek/deepseek-v3.2

Flow per sample:
  1. Select prompt by CWE
  2. Call OpenRouter API (up to max_attempts_tier2 tries, cycling models)
  3. Extract C code via extract_c_code_v2()
  4. Run validate_cfa_v2() 7-gate check
  5. Collect valid CFAs up to target_ratio
  6. If 3 consecutive compile fails: attempt Tier 5 refinement (stub)

Source: docs/New Docs/StreamGuard_CFA_Generation_Research.docx §2.2, §3
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from loguru import logger

# ── Tier 2 prompts (exact spec from prompt) ──────────────────────
TIER2_PROMPT_CWE121 = """
You are fixing a C function with a STACK BUFFER OVERFLOW (CWE-121).
An array is written to using an index or copy operation without validating bounds.
CHOOSE EXACTLY ONE fix that applies to this code:
  Option A - Index bounds check: Before 'array[idx] = value', add:
    if (idx < 0 || idx >= ARRAY_SIZE) { return; }
    (use the actual array size variable or sizeof(array)/sizeof(array[0]))
  Option B - String copy bounds: Replace strcpy(arr, src) with:
    strncpy(arr, src, sizeof(arr)-1); arr[sizeof(arr)-1] = '\\0';
  Option C - Memory copy bounds: Before memcpy(arr, src, len), add:
    if (len > sizeof(arr)) { len = sizeof(arr); }
RULES: Modify ONLY the overflowing line and add ONE check before it.
Do NOT change function signature, return type, or any other logic.
Do NOT add includes. Output ONLY the corrected C function. [FIXED_CODE_START]
"""

TIER2_PROMPT_CWE122 = """
You are fixing a C function with a HEAP BUFFER OVERFLOW (CWE-122).
A heap allocation result is used without size validation.
Apply BOTH of these fixes:
  Fix 1 - After malloc/calloc: if (ptr == NULL) { return NULL; }
  Fix 2 - Before writing to allocated buffer: validate write size <= allocated size
    if (write_size > allocated_size) { write_size = allocated_size; }
RULES: Modify ONLY the allocation and write lines. Add exactly these two checks.
Do NOT add includes. Output ONLY the corrected C function. [FIXED_CODE_START]
"""

TIER2_PROMPT_CWE125 = """
You are fixing a C function with an OUT-OF-BOUNDS READ (CWE-125).
An array is read at an index that may exceed the array bounds.
Apply this fix: Before 'value = array[idx]' or similar read, add:
  if (idx < 0 || idx >= array_length) { return -1; }
  (use actual array length variable, or sizeof(array)/sizeof(array[0]) for stack arrays)
For pointer reads: if (ptr == NULL || ptr >= end_of_allocation) { return NULL; }
RULES: Modify ONLY the array read line and add ONE bounds check before it.
Do NOT change function signature or any other logic.
Do NOT add includes. Output ONLY the corrected C function. [FIXED_CODE_START]
"""

TIER2_PROMPT_CWE476 = """
You are fixing a C function with a NULL POINTER DEREFERENCE (CWE-476).
A pointer is dereferenced (*ptr, ptr->field) without checking if it is NULL first.
Apply this fix: Before EACH dereference of the pointer, add a NULL check:
  if (ptr == NULL) { return <appropriate_default>; }
  Use return; for void functions, return NULL; for pointer returns, return -1; for int.
If the pointer is a function parameter, add the check at the top of the function.
If the pointer comes from malloc/calloc, add the check right after allocation.
RULES: ONLY add NULL checks before pointer dereferences. Do NOT change any other logic.
Do NOT change function signature. Do NOT add includes.
Output ONLY the corrected C function. [FIXED_CODE_START]
"""

TIER2_PROMPTS: dict[str, str] = {
    "CWE-121": TIER2_PROMPT_CWE121,
    "CWE-122": TIER2_PROMPT_CWE122,
    "CWE-125": TIER2_PROMPT_CWE125,
    "CWE-476": TIER2_PROMPT_CWE476,
}

# Marker the LLM is instructed to emit before the fixed code
_FIXED_CODE_MARKER = "[FIXED_CODE_START]"

# Patterns that indicate start of a C function definition
_C_FUNC_START = re.compile(
    r'^(static\s+|inline\s+|extern\s+)?(const\s+)?'
    r'(void|int|long|char|short|unsigned|struct|enum|float|double|\w+)\s+\**\w+\s*\(',
    re.MULTILINE,
)


# ── Code extraction ───────────────────────────────────────────────
def extract_c_code_v2(response: str) -> str:
    """
    Extract C code from LLM response using a 4-stage cascade:

    1. [FIXED_CODE_START] marker — take everything after it
    2. ```c ... ``` or ``` ... ``` markdown fence
    3. First line that looks like a C function definition
    4. Raw response stripped (fallback)
    """
    if not response:
        return ""

    # Stage 1: custom marker
    marker_pos = response.find(_FIXED_CODE_MARKER)
    if marker_pos != -1:
        candidate = response[marker_pos + len(_FIXED_CODE_MARKER):].strip()
        if candidate:
            return candidate

    # Stage 2: markdown fence
    m = re.search(r'```(?:c|cpp)?\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Stage 3: first C function definition
    m = _C_FUNC_START.search(response)
    if m:
        return response[m.start():].strip()

    # Stage 4: raw fallback
    return response.strip()


# ── OpenRouter API client ─────────────────────────────────────────
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary model for all tiers
_DEFAULT_MODEL = "deepseek/deepseek-v3.2"


def call_llm(
    prompt: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1024,
    api_key: str | None = None,
    timeout: float = 60.0,
    attempt_index: int = 0,  # kept for call-site compatibility, unused
) -> str | None:
    """
    Call OpenRouter (OpenAI-compatible) chat completions API using DeepSeek V3.2.

    Returns the response text, or None on any error (network, auth, rate limit).
    Uses OPENROUTER_API_KEY environment variable if api_key not provided.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        logger.warning("OPENROUTER_API_KEY not set — LLM call skipped")
        return None

    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed. Run: pip install httpx")
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/streamguard",
        "X-Title": "StreamGuard CFA",
    }

    try:
        resp = httpx.post(
            _OPENROUTER_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.debug(f"OpenRouter [{model}]: OK")
        return content
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 429:
            logger.warning(f"OpenRouter rate limit hit (429) on {model}")
        elif status in (401, 403):
            logger.error(f"OpenRouter auth error ({status})")
        else:
            logger.warning(f"OpenRouter HTTP error {status} on {model}: {e.response.text[:200]}")
        return None
    except httpx.TimeoutException:
        logger.warning(f"OpenRouter timeout on {model}")
        return None
    except Exception as e:
        logger.warning(f"OpenRouter API error: {e}")
        return None


# ── Tier 5 refinement stub ────────────────────────────────────────
def _tier5_refine(
    original_code: str,
    failed_cfa: str,
    cwe: str,
    failure_reason: str,
    config: Any,
) -> dict | None:
    """
    Tier 5 critique-and-refine stub.

    Will be implemented in the Tier 5 story. Returns None until built.
    Registered in TIER_GENERATORS[5] when ready.
    """
    from training.scripts.preprocessing.stage3_cfa import TIER_GENERATORS
    tier5_fn = TIER_GENERATORS.get(5)
    if tier5_fn is None:
        return None  # tier 5 not yet built
    results = tier5_fn(original_code, cwe, config, failed_cfa=failed_cfa, failure_reason=failure_reason)
    if results:
        return {"cfa_code": results[0], "tier": 5, "quality_score": 0.8}
    return None


# ── Tier 2 entry point ────────────────────────────────────────────
def tier2_generate(
    code: str,
    cwe: str,
    config: Any = None,
    exemplar_db: dict | None = None,
) -> list[str]:
    """
    Tier 2 CFA generator — zero-shot LLM prompting via DeepSeek API.

    Returns list of CFA code strings (validated, ready for further 7-gate check
    in the orchestrator). Empty list if API unavailable or no valid CFAs found.

    Config attributes used (with defaults):
      config.max_attempts_tier2     int   6
      config.target_ratio           int   1
      config.openrouter_api_key     str   None  (falls back to env var)
      config.llm_timeout            float 60.0
    """
    # Lazy import to avoid circular imports at module load time
    from training.scripts.preprocessing.stage3_cfa import validate_cfa_v2

    prompt_template = TIER2_PROMPTS.get(cwe)
    if prompt_template is None:
        logger.warning(f"Tier 2: no prompt for {cwe}")
        return []

    # Config values with defaults
    max_attempts = getattr(config, "max_attempts_tier2", 6)
    target_ratio = getattr(config, "target_ratio", 1)
    api_key = getattr(config, "openrouter_api_key", None)
    llm_timeout = getattr(config, "llm_timeout", 60.0)

    full_prompt = (
        f"VULNERABLE C FUNCTION:\n```c\n{code}\n```\n\nTASK:\n{prompt_template}"
    )

    results: list[str] = []
    consecutive_compile_fails = 0
    last_failed_cfa = ""

    for attempt in range(max_attempts):
        if len(results) >= target_ratio:
            break

        raw = call_llm(
            full_prompt,
            max_tokens=1024,
            api_key=api_key,
            timeout=llm_timeout,
            attempt_index=attempt,
        )
        if raw is None:
            logger.debug(f"Tier 2 [{cwe}] attempt {attempt+1}: LLM returned None")
            continue

        cfa_code = extract_c_code_v2(raw)
        if not cfa_code:
            logger.debug(f"Tier 2 [{cwe}] attempt {attempt+1}: empty extraction")
            continue

        is_valid, reason, cleaned, score = validate_cfa_v2(
            code, cfa_code, cwe, tier=2, config=config
        )

        if is_valid:
            results.append(cleaned)
            consecutive_compile_fails = 0
            logger.debug(
                f"Tier 2 [{cwe}] attempt {attempt+1}: ACCEPTED (score={score:.2f})"
            )
        else:
            logger.debug(
                f"Tier 2 [{cwe}] attempt {attempt+1}: rejected ({reason})"
            )
            if reason == "compile_fail":
                consecutive_compile_fails += 1
                last_failed_cfa = cfa_code
            else:
                consecutive_compile_fails = 0

            # 3 consecutive compile fails: attempt Tier 5 refinement
            if consecutive_compile_fails >= 3:
                logger.debug(
                    f"Tier 2 [{cwe}]: 3 consecutive compile fails, trying Tier 5 refinement"
                )
                t5_result = _tier5_refine(code, last_failed_cfa, cwe, reason, config)
                if t5_result:
                    results.append(t5_result["cfa_code"])
                consecutive_compile_fails = 0

    return results
