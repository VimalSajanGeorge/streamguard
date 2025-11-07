"""
JSON Safety Utilities for Production Training

Provides safe JSON serialization for PyTorch tensors, NumPy arrays, and other
non-primitive types commonly encountered in ML training.

Features:
- safe_jsonify(): Convert tensors/numpy to JSON-safe primitives
- atomic_write_json(): Atomic write-replace pattern for crash-safety
- validate_json_safe(): Pre-check for JSON serializability

Author: StreamGuard Team
Version: 1.0.0
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def safe_jsonify(obj: Any) -> Any:
    """
    Convert potentially unsafe objects to JSON-safe primitives.

    Handles:
    - PyTorch Tensors → Python lists
    - NumPy arrays → Python lists
    - pathlib.Path → strings
    - torch.device → strings
    - Nested dicts/lists recursively

    Args:
        obj: Object to convert

    Returns:
        JSON-safe version of object

    Raises:
        TypeError: If object cannot be safely converted

    Examples:
        >>> tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> safe_jsonify(tensor)
        [1.0, 2.0, 3.0]

        >>> safe_jsonify({"loss": torch.tensor(0.5), "path": Path("/tmp/model.pt")})
        {"loss": 0.5, "path": "/tmp/model.pt"}
    """
    # None, bool, int, float, str are already JSON-safe
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # PyTorch Tensor
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        # Detach from computation graph, move to CPU, convert to numpy, then list
        tensor_np = obj.detach().cpu().numpy()

        # Handle scalar tensors
        if tensor_np.ndim == 0:
            return tensor_np.item()

        # Handle multi-dimensional tensors
        return tensor_np.tolist()

    # NumPy array
    if isinstance(obj, np.ndarray):
        # Handle scalar arrays
        if obj.ndim == 0:
            return obj.item()

        # Handle multi-dimensional arrays
        return obj.tolist()

    # NumPy scalar types (np.int64, np.float32, etc.)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # torch.device
    if TORCH_AVAILABLE and isinstance(obj, torch.device):
        return str(obj)

    # Dict: recursively convert values
    if isinstance(obj, dict):
        return {key: safe_jsonify(value) for key, value in obj.items()}

    # List/Tuple: recursively convert elements
    if isinstance(obj, (list, tuple)):
        return [safe_jsonify(item) for item in obj]

    # Unsupported type
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable. "
        f"Use safe_jsonify() to convert tensors, numpy arrays, or paths first."
    )


def atomic_write_json(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    indent: int = 2,
    sort_keys: bool = False
) -> None:
    """
    Atomically write JSON to file using temp-file-replace pattern.

    Prevents corruption if process crashes during write.

    Steps:
    1. Write to temporary file ({file_path}.tmp)
    2. Flush to disk
    3. Atomically replace original file with os.replace()

    Args:
        data: Dictionary to write (will be passed through safe_jsonify)
        file_path: Target file path
        indent: JSON indentation (default: 2)
        sort_keys: Sort dictionary keys (default: False)

    Raises:
        OSError: If write fails
        TypeError: If data contains unserializable objects

    Examples:
        >>> metadata = {"loss": torch.tensor(0.5), "epoch": 10}
        >>> atomic_write_json(metadata, "training_metadata.json")
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert to JSON-safe format
        safe_data = safe_jsonify(data)

        # Write to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(safe_data, f, indent=indent, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Atomically replace original file
        # os.replace() is atomic on both POSIX and Windows
        os.replace(temp_path, file_path)

    except Exception as e:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise


def validate_json_safe(obj: Any, path: str = "root") -> List[str]:
    """
    Validate that an object can be safely JSON-serialized.

    Returns list of problematic paths if validation fails.

    Args:
        obj: Object to validate
        path: Current path in object tree (for error reporting)

    Returns:
        List of error messages (empty if validation passes)

    Examples:
        >>> errors = validate_json_safe({"loss": torch.tensor(0.5)})
        >>> if errors:
        ...     print("Validation failed:", errors)
        Validation failed: ['root.loss: torch.Tensor not JSON-safe']
    """
    errors = []

    # Primitives are safe
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return errors

    # Check for unsafe types
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        errors.append(f"{path}: torch.Tensor not JSON-safe (use safe_jsonify)")
        return errors

    if isinstance(obj, np.ndarray):
        errors.append(f"{path}: numpy.ndarray not JSON-safe (use safe_jsonify)")
        return errors

    if isinstance(obj, Path):
        errors.append(f"{path}: pathlib.Path not JSON-safe (use safe_jsonify)")
        return errors

    if TORCH_AVAILABLE and isinstance(obj, torch.device):
        errors.append(f"{path}: torch.device not JSON-safe (use safe_jsonify)")
        return errors

    # Recursively validate containers
    if isinstance(obj, dict):
        for key, value in obj.items():
            errors.extend(validate_json_safe(value, f"{path}.{key}"))

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            errors.extend(validate_json_safe(item, f"{path}[{i}]"))

    else:
        # Unknown type
        errors.append(f"{path}: {type(obj).__name__} may not be JSON-safe")

    return errors


def load_json_safe(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file with error handling.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Example usage and tests
if __name__ == "__main__":
    print("Testing JSON Safety Utilities\n")
    print("=" * 60)

    # Test 1: safe_jsonify with tensors
    if TORCH_AVAILABLE:
        print("\n[Test 1] safe_jsonify with PyTorch tensor")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = safe_jsonify(tensor)
        print(f"Input: {tensor}")
        print(f"Output: {result}")
        print(f"Type: {type(result)}")
        assert result == [1.0, 2.0, 3.0]
        print("✅ PASS")

    # Test 2: safe_jsonify with nested dict
    print("\n[Test 2] safe_jsonify with nested dict")
    data = {
        "epoch": 10,
        "loss": np.array(0.5),
        "path": Path("/tmp/model.pt"),
        "metrics": {
            "accuracy": np.float32(0.95),
            "f1": np.array([0.9, 0.92, 0.94])
        }
    }
    result = safe_jsonify(data)
    print(f"Input: {data}")
    print(f"Output: {result}")
    print("✅ PASS")

    # Test 3: atomic_write_json
    print("\n[Test 3] atomic_write_json")
    temp_file = Path(tempfile.gettempdir()) / "test_metadata.json"
    metadata = {
        "epoch": 5,
        "loss": np.array(0.3),
        "path": Path("/models/checkpoint.pt")
    }
    atomic_write_json(metadata, temp_file)

    # Verify file was written
    loaded = load_json_safe(temp_file)
    print(f"Written: {metadata}")
    print(f"Loaded: {loaded}")
    assert loaded["epoch"] == 5
    assert loaded["loss"] == 0.3
    print("✅ PASS")

    # Clean up
    temp_file.unlink()

    # Test 4: validate_json_safe
    print("\n[Test 4] validate_json_safe")
    unsafe_data = {"tensor": torch.tensor(1.0)} if TORCH_AVAILABLE else {"array": np.array(1.0)}
    errors = validate_json_safe(unsafe_data)
    print(f"Input: {unsafe_data}")
    print(f"Errors: {errors}")
    assert len(errors) > 0
    print("✅ PASS")

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
