"""Compatibility helpers for RecBole with newer PyTorch/NumPy."""

import numpy as np
import torch


def patch_numpy_for_recbole() -> None:
    """Restore NumPy aliases removed in 2.x that RecBole 1.2.1 still uses."""
    if getattr(patch_numpy_for_recbole, "_applied", False):
        return

    np.float = np.float64
    np.int = np.int64
    np.bool = np.bool_
    np.object = object
    np.str = str
    patch_numpy_for_recbole._applied = True


def patch_torch_load_for_recbole() -> None:
    """
    RecBole checkpoints include optimizer/config objects, not just tensors.

    PyTorch 2.6+ defaults torch.load(weights_only=True), which breaks loading
    these checkpoints unless weights_only=False is passed explicitly.
    """
    if getattr(torch.load, "_recbole_compat_patch_applied", False):
        return

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    _patched_load._recbole_compat_patch_applied = True
    torch.load = _patched_load


def patch_recbole_compat() -> None:
    patch_numpy_for_recbole()
    patch_torch_load_for_recbole()
