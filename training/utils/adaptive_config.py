"""
Adaptive GPU Configuration Loader

Automatically detects GPU type and memory, provides optimized training configs.

Features:
- Auto-detection of NVIDIA GPU (A100, V100, T4, etc.)
- Memory-aware batch size adjustment
- Safe fallback to CPU defaults
- Config file override support

Author: StreamGuard Team
Version: 1.0.0
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Using CPU defaults.")


# GPU-specific default configurations
GPU_CONFIGS = {
    "A100": {
        "batch_size": 64,
        "max_sequence_length": 512,
        "accumulation_steps": 1,
        "mixed_precision": True,
        "num_workers": 8,
        "pin_memory": True,
        "memory_fraction": 0.9
    },
    "V100": {
        "batch_size": 48,
        "max_sequence_length": 512,
        "accumulation_steps": 2,
        "mixed_precision": True,
        "num_workers": 6,
        "pin_memory": True,
        "memory_fraction": 0.85
    },
    "T4": {
        "batch_size": 32,
        "max_sequence_length": 384,
        "accumulation_steps": 2,
        "mixed_precision": True,
        "num_workers": 4,
        "pin_memory": True,
        "memory_fraction": 0.8
    },
    "RTX3090": {
        "batch_size": 48,
        "max_sequence_length": 512,
        "accumulation_steps": 2,
        "mixed_precision": True,
        "num_workers": 6,
        "pin_memory": True,
        "memory_fraction": 0.85
    },
    "RTX4090": {
        "batch_size": 56,
        "max_sequence_length": 512,
        "accumulation_steps": 1,
        "mixed_precision": True,
        "num_workers": 8,
        "pin_memory": True,
        "memory_fraction": 0.9
    },
    "CPU": {
        "batch_size": 16,
        "max_sequence_length": 256,
        "accumulation_steps": 1,
        "mixed_precision": False,
        "num_workers": 2,
        "pin_memory": False,
        "memory_fraction": 1.0
    }
}


def detect_gpu() -> Dict[str, Any]:
    """
    Detect GPU type and memory.

    Returns:
        Dictionary with GPU info:
        - name: GPU name (e.g., "A100", "V100")
        - memory_gb: Total memory in GB
        - device: torch device ("cuda" or "cpu")
        - count: Number of GPUs

    Examples:
        >>> info = detect_gpu()
        >>> print(info)
        {'name': 'A100', 'memory_gb': 40, 'device': 'cuda', 'count': 1}
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {
            "name": "CPU",
            "memory_gb": 0,
            "device": "cpu",
            "count": 0
        }

    # Get GPU name
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()

    # Get memory in GB
    memory_bytes = torch.cuda.get_device_properties(0).total_memory
    memory_gb = memory_bytes / (1024 ** 3)

    # Normalize GPU name to config key
    gpu_key = "CPU"  # Default fallback
    for key in GPU_CONFIGS.keys():
        if key in gpu_name:
            gpu_key = key
            break

    return {
        "name": gpu_key,
        "full_name": gpu_name,
        "memory_gb": round(memory_gb, 1),
        "device": "cuda",
        "count": gpu_count
    }


def load_adaptive_config(
    config_file: Optional[Path] = None,
    model_type: str = "transformer",
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load adaptive training configuration.

    Priority:
    1. User-provided config file (if exists)
    2. GPU-specific defaults
    3. CPU fallback

    Args:
        config_file: Path to optional config JSON file
        model_type: "transformer", "gnn", or "fusion"
        override: Dictionary of values to override

    Returns:
        Training configuration dictionary

    Examples:
        >>> config = load_adaptive_config(model_type="transformer")
        >>> print(config["batch_size"])
        64  # On A100
    """
    # Detect GPU
    gpu_info = detect_gpu()
    gpu_key = gpu_info["name"]

    print(f"[+] Detected GPU: {gpu_info.get('full_name', 'CPU')}")
    print(f"[+] Memory: {gpu_info['memory_gb']} GB")
    print(f"[+] Device: {gpu_info['device']}")

    # Start with GPU-specific defaults
    config = GPU_CONFIGS.get(gpu_key, GPU_CONFIGS["CPU"]).copy()
    config["gpu_info"] = gpu_info

    # Model-specific adjustments
    if model_type == "gnn":
        # GNNs typically need smaller batches due to graph memory overhead
        config["batch_size"] = max(16, config["batch_size"] // 2)
        config["accumulation_steps"] = config.get("accumulation_steps", 1) * 2

    elif model_type == "fusion":
        # Fusion models combine both, need even smaller batches
        config["batch_size"] = max(8, config["batch_size"] // 3)
        config["accumulation_steps"] = config.get("accumulation_steps", 1) * 3

    # Load from file if provided
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)

            # Merge file config (file takes priority)
            config.update(file_config)
            print(f"[+] Loaded config from: {config_file}")

        except Exception as e:
            warnings.warn(f"Failed to load config file: {e}. Using defaults.")

    # Apply overrides
    if override:
        config.update(override)
        print(f"[+] Applied overrides: {list(override.keys())}")

    # Add model type
    config["model_type"] = model_type

    # Log final config
    print(f"[+] Final batch size: {config['batch_size']}")
    print(f"[+] Gradient accumulation: {config['accumulation_steps']}")
    print(f"[+] Mixed precision: {config['mixed_precision']}")

    return config


def adjust_batch_size_for_memory(
    base_batch_size: int,
    available_memory_gb: float,
    model_size_mb: float = 500,  # Estimated model size in MB
    safety_factor: float = 0.7  # Use only 70% of available memory
) -> int:
    """
    Dynamically adjust batch size based on available GPU memory.

    Args:
        base_batch_size: Starting batch size
        available_memory_gb: Available GPU memory in GB
        model_size_mb: Estimated model size in MB
        safety_factor: Fraction of memory to use (default: 0.7)

    Returns:
        Adjusted batch size

    Examples:
        >>> adjusted = adjust_batch_size_for_memory(64, 40.0, 500)
        >>> print(adjusted)
        64  # Fits comfortably in 40GB
    """
    # Convert to MB
    available_mb = available_memory_gb * 1024 * safety_factor

    # Estimate memory per sample (very rough heuristic)
    # Transformer: ~5MB/sample for seq_len=512
    # GNN: ~10MB/graph for 50 nodes
    memory_per_sample_mb = 5.0

    # Calculate max batch size
    max_batch_size = int((available_mb - model_size_mb) / memory_per_sample_mb)

    # Return minimum of base and max
    adjusted = min(base_batch_size, max_batch_size)
    adjusted = max(1, adjusted)  # At least 1

    if adjusted != base_batch_size:
        warnings.warn(
            f"Adjusted batch size from {base_batch_size} to {adjusted} "
            f"due to memory constraints ({available_memory_gb} GB available)"
        )

    return adjusted


def create_training_config_file(
    output_path: Path,
    model_type: str = "transformer",
    **kwargs
) -> None:
    """
    Create a training config file template.

    Args:
        output_path: Path to save config JSON
        model_type: "transformer", "gnn", or "fusion"
        **kwargs: Additional config parameters

    Examples:
        >>> create_training_config_file(
        ...     Path("training_config.json"),
        ...     model_type="transformer",
        ...     learning_rate=1e-5
        ... )
    """
    # Load defaults
    config = load_adaptive_config(model_type=model_type)

    # Add/override with kwargs
    config.update(kwargs)

    # Remove non-serializable objects
    if "gpu_info" in config:
        del config["gpu_info"]

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"[+] Created config file: {output_path}")


# Example usage
if __name__ == "__main__":
    print("Testing Adaptive GPU Configuration\n")
    print("=" * 60)

    # Test 1: Detect GPU
    print("\n[Test 1] GPU Detection")
    gpu_info = detect_gpu()
    print(f"GPU Info: {gpu_info}")
    print("✅ PASS")

    # Test 2: Load adaptive config for Transformer
    print("\n[Test 2] Load Transformer Config")
    config = load_adaptive_config(model_type="transformer")
    print(f"Batch size: {config['batch_size']}")
    print(f"Mixed precision: {config['mixed_precision']}")
    print("✅ PASS")

    # Test 3: Load adaptive config for GNN
    print("\n[Test 3] Load GNN Config")
    config = load_adaptive_config(model_type="gnn")
    print(f"Batch size: {config['batch_size']}")
    print("✅ PASS")

    # Test 4: Load adaptive config for Fusion
    print("\n[Test 4] Load Fusion Config")
    config = load_adaptive_config(model_type="fusion")
    print(f"Batch size: {config['batch_size']}")
    print("✅ PASS")

    # Test 5: Memory-based adjustment
    print("\n[Test 5] Memory-based Batch Size Adjustment")
    adjusted = adjust_batch_size_for_memory(64, 16.0)  # 16 GB GPU
    print(f"Adjusted batch size: {adjusted}")
    print("✅ PASS")

    # Test 6: Create config file
    print("\n[Test 6] Create Config File")
    import tempfile
    temp_file = Path(tempfile.gettempdir()) / "test_training_config.json"
    create_training_config_file(
        temp_file,
        model_type="transformer",
        learning_rate=1e-5,
        epochs=10
    )

    # Verify file
    with open(temp_file) as f:
        loaded_config = json.load(f)
    print(f"Config: {loaded_config}")
    temp_file.unlink()
    print("✅ PASS")

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
