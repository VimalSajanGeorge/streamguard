"""
LR Finder Cache Manager

Provides secure, atomic caching of LR Finder results with dataset fingerprinting.
Cache keys are computed from dataset checksum + model config + batch size.

Features:
- Atomic writes (tmp file → rename) for preemptible environments
- Dataset fingerprinting via mtime + size
- Auto-cleanup of corrupted cache files
- Configurable cache expiry (default: 1 week)
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Cache directory
LR_CACHE_DIR = Path("models/transformer/.lr_cache")
LR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_cache_key(
    dataset_path: Path,
    model_name: str,
    batch_size: int,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a stable cache key by hashing dataset fingerprint + config.

    Args:
        dataset_path: Path to training data
        model_name: Model name (e.g., "microsoft/codebert-base")
        batch_size: Batch size used for training
        extra: Additional config (e.g., {'max_seq_len': 512})

    Returns:
        SHA1 hash of configuration (40 chars hex)

    Example:
        >>> key = compute_cache_key(
        ...     Path("data/train.jsonl"),
        ...     "microsoft/codebert-base",
        ...     32,
        ...     {'max_seq_len': 512}
        ... )
    """
    extra = extra or {}

    # Use file stats if file exists (mtime + size for fingerprinting)
    if dataset_path.exists():
        stat = dataset_path.stat()
        checksum_source = f"{str(dataset_path.resolve())}:{stat.st_mtime_ns}:{stat.st_size}"
    else:
        # Fallback if file doesn't exist (shouldn't happen in practice)
        checksum_source = f"{str(dataset_path.resolve())}:missing"

    # Combine all config parameters
    key_raw = f"{checksum_source}|{model_name}|{batch_size}|{json.dumps(extra, sort_keys=True)}"

    # Hash to get stable key
    return hashlib.sha1(key_raw.encode("utf-8")).hexdigest()


def cache_path_for_key(key: str) -> Path:
    """Get cache file path for a given key."""
    return LR_CACHE_DIR / f"{key}.json"


def save_lr_cache(
    key: str,
    suggested_lr: float,
    lr_history: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save LR Finder results to cache (atomic write).

    Uses atomic write pattern: write to temp file, then rename.
    This is safe for preemptible environments (e.g., Colab with interruptions).

    Args:
        key: Cache key from compute_cache_key()
        suggested_lr: Suggested learning rate
        lr_history: Summary of LR history (min/max/argmin, not full array)
        metadata: Additional metadata (analysis, validation results, etc.)

    Example:
        >>> save_lr_cache(
        ...     "abc123",
        ...     1.5e-5,
        ...     {'min_loss': 0.5, 'max_loss': 1.2, 'num_points': 100},
        ...     {'confidence': 'high', 'note': 'accepted'}
        ... )
    """
    metadata = metadata or {}

    payload = {
        "suggested_lr": float(suggested_lr),
        "timestamp": datetime.utcnow().isoformat(),
        "history_summary": lr_history,
        "metadata": metadata
    }

    dest = cache_path_for_key(key)
    tmp = dest.with_suffix(".json.tmp")

    # Write to temp file
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Atomic rename (safe on most OSes)
    tmp.replace(dest)


def load_lr_cache(
    key: str,
    max_age_hours: int = 168
) -> Optional[Dict[str, Any]]:
    """
    Load cached LR Finder results.

    Args:
        key: Cache key from compute_cache_key()
        max_age_hours: Maximum cache age in hours (default: 168 = 1 week)

    Returns:
        Cached data dict or None if cache miss/expired/corrupted

    Example:
        >>> cached = load_lr_cache("abc123")
        >>> if cached:
        ...     lr = cached['suggested_lr']
        ...     timestamp = cached['timestamp']
    """
    p = cache_path_for_key(key)

    if not p.exists():
        return None

    try:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        # Check expiry
        ts = datetime.fromisoformat(payload.get("timestamp"))
        if datetime.utcnow() - ts > timedelta(hours=max_age_hours):
            return None

        return payload

    except Exception:
        # If corrupted, remove and return None
        try:
            p.unlink()
        except Exception:
            pass
        return None


def invalidate_cache(key: str) -> None:
    """
    Invalidate (delete) cached LR for a given key.

    Args:
        key: Cache key to invalidate

    Example:
        >>> invalidate_cache("abc123")  # Force re-run on next training
    """
    p = cache_path_for_key(key)
    if p.exists():
        p.unlink()


# Example usage
if __name__ == "__main__":
    print("LR Cache Manager - Example Usage:")
    print("""
    from training.utils.lr_cache import compute_cache_key, save_lr_cache, load_lr_cache

    # Compute cache key
    key = compute_cache_key(
        Path("data/train.jsonl"),
        "microsoft/codebert-base",
        32,
        {'max_seq_len': 512}
    )

    # Check if cached
    cached = load_lr_cache(key)
    if cached:
        print(f"Using cached LR: {cached['suggested_lr']:.2e}")
    else:
        # Run LR finder...
        suggested_lr = 1.5e-5

        # Save to cache
        save_lr_cache(
            key,
            suggested_lr,
            {'min_loss': 0.5, 'max_loss': 1.2},
            {'confidence': 'high'}
        )
    """)
