"""
Single-Batch Memory Test

Tests memory usage with a full production-size batch to detect OOM before wasting hours.

Features:
- Tests forward + backward pass with production batch size
- Reports peak memory usage
- Warns if memory usage > 90% of available
- Supports mixed precision testing

Usage:
    from training.utils.memory_test import test_single_batch_memory

    # Before training
    test_single_batch_memory(
        model=model,
        sample_batch=batch,
        device=device,
        scaler=scaler  # Optional, for AMP
    )
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import warnings


def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """
    Get GPU memory information.

    Args:
        device: Torch device

    Returns:
        Dictionary with memory stats in MB
    """
    if not torch.cuda.is_available() or device.type == 'cpu':
        return {
            "total_mb": 0,
            "allocated_mb": 0,
            "reserved_mb": 0,
            "free_mb": 0
        }

    total = torch.cuda.get_device_properties(device.index or 0).total_memory / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    free = total - allocated

    return {
        "total_mb": total,
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "free_mb": free
    }


def test_single_batch_memory(
    model: nn.Module,
    sample_batch: Any,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    safety_margin: float = 0.9,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test memory usage with a single batch.

    Args:
        model: PyTorch model
        sample_batch: Sample batch (dict, tensor, or PyG Batch)
        device: Device to test on
        criterion: Loss function (optional, uses CrossEntropyLoss if None)
        scaler: GradScaler for mixed precision (optional)
        safety_margin: Warn if memory > this fraction of total (default: 0.9)
        verbose: Print results

    Returns:
        Dictionary with memory statistics

    Raises:
        MemoryError: If peak memory exceeds safety margin
    """
    if verbose:
        print("\n" + "=" * 80)
        print("SINGLE-BATCH MEMORY TEST")
        print("=" * 80)

    # Get initial memory
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    initial_mem = get_gpu_memory_info(device)

    if verbose:
        print(f"\n[+] Initial Memory:")
        print(f"    Total:     {initial_mem['total_mb']:.0f} MB")
        print(f"    Allocated: {initial_mem['allocated_mb']:.0f} MB")
        print(f"    Free:      {initial_mem['free_mb']:.0f} MB")

    # Default criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Prepare batch
    model.train()
    model.zero_grad()

    try:
        # Forward pass
        if verbose:
            print("\n[*] Running forward pass...")

        # Handle different batch types
        if isinstance(sample_batch, dict):
            # Transformer batch
            if 'input_ids' in sample_batch:
                input_ids = sample_batch['input_ids'].to(device)
                attention_mask = sample_batch['attention_mask'].to(device)
                labels = sample_batch['label'].to(device)

                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                else:
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                raise ValueError("Unknown batch format")

        elif hasattr(sample_batch, 'x'):
            # PyG batch
            sample_batch = sample_batch.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(sample_batch)
                    loss = criterion(logits, sample_batch.y)
            else:
                logits = model(sample_batch)
                loss = criterion(logits, sample_batch.y)

        else:
            raise ValueError("Unknown batch type")

        # Backward pass
        if verbose:
            print("[*] Running backward pass...")

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Get peak memory
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            peak_mem = 0

        final_mem = get_gpu_memory_info(device)

        # Calculate usage
        usage_pct = (peak_mem / initial_mem['total_mb']) if initial_mem['total_mb'] > 0 else 0

        results = {
            "success": True,
            "peak_memory_mb": peak_mem,
            "total_memory_mb": initial_mem['total_mb'],
            "usage_percent": usage_pct * 100,
            "batch_size": len(labels) if isinstance(sample_batch, dict) else sample_batch.num_graphs,
            "safety_margin": safety_margin * 100,
            "is_safe": usage_pct <= safety_margin
        }

        if verbose:
            print(f"\n[+] Memory Test Results:")
            print(f"    Peak Memory:  {peak_mem:.0f} MB")
            print(f"    Total Memory: {initial_mem['total_mb']:.0f} MB")
            print(f"    Usage:        {usage_pct*100:.1f}%")
            print(f"    Batch Size:   {results['batch_size']}")

        # Check if safe
        if usage_pct > safety_margin:
            error_msg = (
                f"\n{'='*80}\n"
                f"⚠️  MEMORY WARNING!\n"
                f"{'='*80}\n"
                f"Peak memory usage ({usage_pct*100:.1f}%) exceeds safety margin ({safety_margin*100:.0f}%).\n"
                f"\n"
                f"Single batch uses {peak_mem:.0f} MB out of {initial_mem['total_mb']:.0f} MB available.\n"
                f"\n"
                f"Recommendations:\n"
                f"  1. Reduce batch size (currently {results['batch_size']})\n"
                f"  2. Enable gradient accumulation\n"
                f"  3. Use mixed precision (AMP)\n"
                f"  4. Reduce model hidden dimensions\n"
                f"\n"
                f"DO NOT proceed with full training - likely to OOM!\n"
                f"{'='*80}\n"
            )

            if verbose:
                warnings.warn(error_msg)

            raise MemoryError(error_msg)

        if verbose:
            print(f"\n✅ PASS - Memory usage within safety margin")
            print(f"   → Safe to proceed with training")

        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            results = {
                "success": False,
                "error": "OOM during test",
                "peak_memory_mb": 0,
                "total_memory_mb": initial_mem['total_mb'],
                "usage_percent": 100.0,
                "batch_size": 0,
                "is_safe": False
            }

            if verbose:
                print(f"\n❌ FAIL - Out of memory during test!")
                print(f"   → Batch size too large for this GPU")

            raise MemoryError(f"OOM during single-batch test: {str(e)}")

        else:
            raise

    finally:
        # Cleanup
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    print("Memory Test Utility")
    print("=" * 80)
    print("\nThis module provides memory testing for production training.")
    print("\nUsage:")
    print("  from training.utils.memory_test import test_single_batch_memory")
    print("  test_single_batch_memory(model, batch, device, scaler)")
    print("\nSee docstring for full API.\n")
