"""Compatibility aliases for SASRecF training now owned by Baseline."""

from importlib import import_module
import sys


_ALIASES = {
    "checkpoint_command": "fashionrec.baseline.models.sasrecf.checkpoint_service",
    "checkpoint_selection": "fashionrec.baseline.models.sasrecf.checkpoint_selection",
    "checkpoints": "fashionrec.baseline.models.sasrecf.checkpoints",
    "command": "fashionrec.baseline.models.sasrecf.training_service",
}

for _name, _target in _ALIASES.items():
    _module = import_module(_target)
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

from fashionrec.baseline.models.sasrecf.checkpoints import discover_checkpoint_candidates, install_validation_checkpoint_shortlist

__all__ = ["discover_checkpoint_candidates", "install_validation_checkpoint_shortlist"]
