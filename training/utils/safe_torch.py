"""Safe helpers for loading torch checkpoints across PyTorch versions.

PyTorch 2.6 flipped the default value of ``weights_only`` in ``torch.load``
from ``False`` to ``True``.  That is great for security, but it breaks older
checkpoints that pickle custom objects (e.g. PyG ``Data``).  The helpers below
register PyG classes when available and transparently retry with
``weights_only=False`` so existing datasets continue to load on both the old
and new defaults without forcing callers to know the details.
"""

from __future__ import annotations

import inspect
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import torch

try:  # PyTorch <=2.5 exposes torch.serialization.add_safe_globals
    from torch.serialization import add_safe_globals  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - depends on torch version
    add_safe_globals = None

__all__ = ["safe_torch_load", "register_safe_graph_globals"]


_HAS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters
_SAFE_GLOBALS_REGISTERED = False
_FALLBACK_WARNED = False


def register_safe_graph_globals() -> bool:
    """Allowlist common PyG classes with torch's safe loader.

    Returns ``True`` when registration succeeds (or already happened).  The
    helper is safe to call multiple times.
    """

    global _SAFE_GLOBALS_REGISTERED

    if _SAFE_GLOBALS_REGISTERED or add_safe_globals is None:
        return _SAFE_GLOBALS_REGISTERED

    try:
        from torch_geometric.data import Data  # type: ignore
    except ImportError:
        return False

    try:
        add_safe_globals([Data])  # type: ignore[arg-type]
        _SAFE_GLOBALS_REGISTERED = True
    except Exception:  # pragma: no cover - extremely unlikely
        return False

    return True


def _should_retry_unpickling(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "weights only load failed" in msg or "not an allowed global" in msg


def _build_loader_kwargs(
    map_location: Optional[Any],
    weights_only: Optional[bool],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(extra_kwargs)
    if map_location is not None:
        kwargs["map_location"] = map_location
    if weights_only is not None and _HAS_WEIGHTS_ONLY:
        kwargs["weights_only"] = weights_only
    return kwargs


def safe_torch_load(
    obj: Union[str, Path, Any],
    *,
    map_location: Optional[Any] = None,
    weights_only: Optional[bool] = None,
    allow_fallback: bool = True,
    register_pyg_data: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a checkpoint/dataset with compatibility fallbacks.

    ``torch.load`` semantics changed in PyTorch 2.6.  This wrapper keeps older
    checkpoints usable by registering PyG ``Data`` and retrying with
    ``weights_only=False`` when the safe unpickler refuses to load a known type.
    """

    if register_pyg_data:
        register_safe_graph_globals()

    load_kwargs = _build_loader_kwargs(map_location, weights_only, kwargs)

    global _FALLBACK_WARNED
    try:
        return torch.load(obj, **load_kwargs)
    except pickle.UnpicklingError as exc:
        if not allow_fallback or not _should_retry_unpickling(exc):
            raise

        fallback_kwargs = _build_loader_kwargs(map_location, None, kwargs)
        if _HAS_WEIGHTS_ONLY:
            fallback_kwargs["weights_only"] = False

        if register_pyg_data:
            register_safe_graph_globals()

        if not _FALLBACK_WARNED:
            warnings.warn(
                f"torch.load fallback triggered for '{obj}'. Retrying with "
                "weights_only=False (subsequent fallbacks suppressed).",
                RuntimeWarning,
                stacklevel=2,
            )
            _FALLBACK_WARNED = True
        return torch.load(obj, **fallback_kwargs)
