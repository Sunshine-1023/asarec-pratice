"""Convenience entrypoint for training SASRecF with default config."""

from __future__ import annotations

import sys

from run_sasrec import main


def _ensure_default_sasrecf_config() -> None:
    """Inject sasrecf config when user does not pass --config."""
    if "--config" in sys.argv:
        return
    sys.argv.extend(["--config", "configs/sasrecf.yaml"])


if __name__ == "__main__":
    _ensure_default_sasrecf_config()
    main()
