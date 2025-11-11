"""
Atomic JSON Write Utility

Provides safe, crash-resistant JSON file writing using the tmp → rename pattern.
This prevents file corruption if the process crashes or is interrupted mid-write.

Usage:
    from docs.snippets.atomic_write_json import atomic_write_json

    data = {"epoch": 10, "loss": 0.45, "f1": 0.89}
    atomic_write_json("outputs/metrics.json", data)

Pattern:
    1. Write to temporary file (path.tmp)
    2. Verify write succeeded
    3. Atomically rename tmp → final (atomic on POSIX, near-atomic on Windows)

Benefits:
    - No partial writes (all-or-nothing)
    - No file corruption from crashes
    - Safe for concurrent reads (readers never see partial data)
"""

import json
from pathlib import Path
from typing import Any, Dict, Union
import logging

logger = logging.getLogger(__name__)


def safe_jsonify(obj: Any) -> Any:
    """
    Convert Python objects to JSON-serializable format.

    Handles common non-serializable types:
    - torch.Tensor → list
    - numpy.ndarray → list
    - Path → str
    - set → list

    Args:
        obj: Any Python object

    Returns:
        JSON-serializable version of obj
    """
    import torch
    import numpy as np

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: safe_jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_jsonify(v) for v in obj]
    else:
        return obj


def atomic_write_json(
    path: Union[str, Path],
    obj: Dict[str, Any],
    indent: int = 2,
    safe: bool = True
) -> None:
    """
    Write JSON file atomically using tmp → rename pattern.

    Args:
        path: Target file path
        obj: Dictionary to serialize as JSON
        indent: JSON indentation (default: 2)
        safe: Convert non-serializable types (default: True)

    Raises:
        TypeError: If obj contains non-serializable types and safe=False
        IOError: If write fails

    Example:
        >>> data = {"epoch": 10, "tensor": torch.tensor([1, 2, 3])}
        >>> atomic_write_json("metrics.json", data, safe=True)
        # metrics.json now contains: {"epoch": 10, "tensor": [1, 2, 3]}
    """
    path = Path(path)
    tmp_path = Path(str(path) + ".tmp")

    try:
        # Convert non-serializable types if requested
        if safe:
            obj = safe_jsonify(obj)

        # Write to temporary file
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=indent)
            f.flush()  # Ensure OS writes to disk

        # Atomic rename (POSIX) or near-atomic (Windows)
        # On Windows, this may fail if target exists; handled below
        try:
            tmp_path.replace(path)
        except OSError as e:
            # Windows: target may be locked by another process
            # Try removing target first (less safe but works)
            if path.exists():
                path.unlink()
            tmp_path.replace(path)

        logger.debug(f"Atomically wrote JSON to {path}")

    except Exception as e:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error(f"Failed to write JSON to {path}: {e}")
        raise


def atomic_write_jsonl(
    path: Union[str, Path],
    records: list,
    safe: bool = True
) -> None:
    """
    Write JSONL (JSON Lines) file atomically.

    Each record is written as a single line of JSON.

    Args:
        path: Target file path
        records: List of dictionaries to write as JSONL
        safe: Convert non-serializable types (default: True)

    Example:
        >>> records = [
        ...     {"id": 1, "text": "example 1"},
        ...     {"id": 2, "text": "example 2"}
        ... ]
        >>> atomic_write_jsonl("data.jsonl", records)
    """
    path = Path(path)
    tmp_path = Path(str(path) + ".tmp")

    try:
        # Convert non-serializable types if requested
        if safe:
            records = [safe_jsonify(rec) for rec in records]

        # Write to temporary file
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
            f.flush()

        # Atomic rename
        try:
            tmp_path.replace(path)
        except OSError:
            if path.exists():
                path.unlink()
            tmp_path.replace(path)

        logger.debug(f"Atomically wrote JSONL to {path}")

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error(f"Failed to write JSONL to {path}: {e}")
        raise


# Example usage
if __name__ == "__main__":
    import torch

    # Example 1: Write simple dict
    data = {
        "epoch": 10,
        "loss": 0.4523,
        "metrics": {
            "accuracy": 0.91,
            "f1": 0.89
        }
    }
    atomic_write_json("example_metrics.json", data)
    print("✓ Wrote example_metrics.json")

    # Example 2: Write dict with tensors (auto-converted)
    data_with_tensors = {
        "predictions": torch.tensor([0, 1, 1, 0]),
        "scores": torch.tensor([0.8, 0.9, 0.7, 0.6])
    }
    atomic_write_json("example_predictions.json", data_with_tensors, safe=True)
    print("✓ Wrote example_predictions.json (tensors converted to lists)")

    # Example 3: Write JSONL
    records = [
        {"id": 1, "label": 0, "score": 0.8},
        {"id": 2, "label": 1, "score": 0.9}
    ]
    atomic_write_jsonl("example.jsonl", records)
    print("✓ Wrote example.jsonl")
