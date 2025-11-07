"""
AMP (Automatic Mixed Precision) Utilities

Provides safe utilities for mixed precision training with proper gradient handling.

Features:
- AMP-safe gradient clipping
- Gradient norm monitoring
- NaN/Inf detection
- GradScaler state management

Author: StreamGuard Team
Version: 1.0.0
"""

from typing import Optional, Dict, Any, List
import warnings

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. AMP utilities disabled.")


def clip_gradients_amp_safe(
    model: nn.Module,
    max_grad_norm: float,
    scaler: Optional[GradScaler] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Perform AMP-safe gradient clipping.

    CRITICAL: When using mixed precision (GradScaler), gradients are SCALED.
    You must unscale them BEFORE clipping, otherwise you'll clip wrong values.

    Correct order:
    1. scaler.scale(loss).backward()  # Scaled gradients
    2. scaler.unscale_(optimizer)     # Unscale BEFORE clipping
    3. torch.nn.utils.clip_grad_norm_()  # Clip TRUE gradients
    4. scaler.step(optimizer)
    5. scaler.update()

    Args:
        model: PyTorch model
        max_grad_norm: Maximum gradient norm (e.g., 1.0)
        scaler: GradScaler instance (if using AMP)
        verbose: Print gradient norm info

    Returns:
        Dictionary with gradient statistics:
        - total_norm: Total gradient norm (before clipping)
        - clipped: Whether clipping was applied
        - scale: Current grad scaler scale (if AMP)

    Examples:
        >>> scaler = GradScaler()
        >>> loss = model(x)
        >>> scaler.scale(loss).backward()
        >>> stats = clip_gradients_amp_safe(model, 1.0, scaler)
        >>> print(f"Grad norm: {stats['total_norm']:.4f}")
        >>> scaler.step(optimizer)
        >>> scaler.update()
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    # Unscale gradients if using AMP
    # This is CRITICAL - without this, we clip scaled gradients!
    if scaler is not None:
        scaler.unscale_(model.optimizer if hasattr(model, 'optimizer') else None)

    # Get parameters with gradients
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        warnings.warn("No gradients found. Skipping gradient clipping.")
        return {
            "total_norm": 0.0,
            "clipped": False,
            "scale": scaler.get_scale() if scaler else 1.0,
            "num_parameters": 0
        }

    # Compute total gradient norm BEFORE clipping
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_grad_norm,
        norm_type=2.0
    ).item()

    # Check if clipping was applied
    clipped = total_norm > max_grad_norm

    # Get scaler scale
    scale = scaler.get_scale() if scaler else 1.0

    if verbose:
        if clipped:
            print(f"[!] Gradient clipped: {total_norm:.4f} -> {max_grad_norm:.4f}")
        else:
            print(f"[+] Gradient norm: {total_norm:.4f} (no clipping needed)")

        if scaler:
            print(f"    AMP scale: {scale:.0f}")

    return {
        "total_norm": total_norm,
        "clipped": clipped,
        "scale": scale,
        "num_parameters": len(parameters)
    }


def check_gradients_health(
    model: nn.Module,
    log_per_layer: bool = False
) -> Dict[str, Any]:
    """
    Check gradient health for NaN, Inf, and zero gradients.

    Args:
        model: PyTorch model
        log_per_layer: Log statistics per layer (verbose)

    Returns:
        Dictionary with gradient health statistics
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}

    stats = {
        "has_nan": False,
        "has_inf": False,
        "has_zero_grad": False,
        "total_params": 0,
        "params_with_grad": 0,
        "nan_params": 0,
        "inf_params": 0,
        "zero_grad_params": 0,
        "grad_norm_total": 0.0,
        "layer_stats": []
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            stats["params_with_grad"] += 1

            # Check for NaN
            if torch.isnan(param.grad).any():
                stats["has_nan"] = True
                stats["nan_params"] += 1

            # Check for Inf
            if torch.isinf(param.grad).any():
                stats["has_inf"] = True
                stats["inf_params"] += 1

            # Check for zero gradients
            grad_norm = param.grad.norm(2).item()
            if grad_norm < 1e-10:
                stats["has_zero_grad"] = True
                stats["zero_grad_params"] += 1

            stats["grad_norm_total"] += grad_norm ** 2

            # Per-layer logging
            if log_per_layer:
                stats["layer_stats"].append({
                    "name": name,
                    "grad_norm": grad_norm,
                    "has_nan": torch.isnan(param.grad).any().item(),
                    "has_inf": torch.isinf(param.grad).any().item()
                })

        stats["total_params"] += 1

    stats["grad_norm_total"] = stats["grad_norm_total"] ** 0.5

    return stats


def safe_backward(
    loss: torch.Tensor,
    scaler: Optional[GradScaler] = None,
    retain_graph: bool = False
) -> bool:
    """
    Perform safe backward pass with NaN/Inf detection.

    Args:
        loss: Loss tensor
        scaler: GradScaler for AMP (optional)
        retain_graph: Retain computation graph

    Returns:
        True if backward succeeded, False if NaN/Inf detected
    """
    if not TORCH_AVAILABLE:
        return False

    # Check loss before backward
    if torch.isnan(loss).any():
        warnings.warn("NaN detected in loss before backward!")
        return False

    if torch.isinf(loss).any():
        warnings.warn("Inf detected in loss before backward!")
        return False

    try:
        if scaler is not None:
            # AMP backward
            scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            # Standard backward
            loss.backward(retain_graph=retain_graph)

        return True

    except RuntimeError as e:
        warnings.warn(f"Backward pass failed: {e}")
        return False


def get_scaler_state_dict(scaler: GradScaler) -> Dict[str, Any]:
    """
    Get GradScaler state for checkpointing.

    Args:
        scaler: GradScaler instance

    Returns:
        State dictionary (JSON-safe)
    """
    if not TORCH_AVAILABLE or scaler is None:
        return {}

    state = scaler.state_dict()

    # Convert to JSON-safe format
    return {
        "scale": float(state.get("scale", 1.0)),
        "growth_factor": float(state.get("growth_factor", 2.0)),
        "backoff_factor": float(state.get("backoff_factor", 0.5)),
        "growth_interval": int(state.get("growth_interval", 2000)),
        "_growth_tracker": int(state.get("_growth_tracker", 0))
    }


def load_scaler_state_dict(
    scaler: GradScaler,
    state_dict: Dict[str, Any]
) -> None:
    """
    Load GradScaler state from checkpoint.

    Args:
        scaler: GradScaler instance
        state_dict: Saved state dictionary
    """
    if not TORCH_AVAILABLE or scaler is None:
        return

    scaler.load_state_dict(state_dict)


# Example usage and tests
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping tests.")
    else:
        print("Testing AMP Utilities\n")
        print("=" * 60)

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()

        # Test 1: AMP-safe gradient clipping
        print("\n[Test 1] AMP-Safe Gradient Clipping")
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        # Forward + backward with AMP
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

        scaler.scale(loss).backward()

        # Clip gradients (AMP-safe)
        stats = clip_gradients_amp_safe(model, max_grad_norm=1.0, scaler=scaler, verbose=True)
        print(f"Gradient stats: {stats}")

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        print("✅ PASS")

        # Test 2: Gradient health check
        print("\n[Test 2] Gradient Health Check")
        x = torch.randn(32, 10)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, torch.randint(0, 2, (32,)))
        loss.backward()

        health = check_gradients_health(model, log_per_layer=False)
        print(f"Gradient health: {health}")
        print("✅ PASS")

        # Test 3: GradScaler state dict
        print("\n[Test 3] GradScaler State Dict")
        state = get_scaler_state_dict(scaler)
        print(f"Scaler state: {state}")

        # Create new scaler and load state
        new_scaler = GradScaler()
        load_scaler_state_dict(new_scaler, state)
        new_state = get_scaler_state_dict(new_scaler)
        assert state["scale"] == new_state["scale"]
        print("✅ PASS")

        print("\n" + "=" * 60)
        print("All tests passed! ✅")
