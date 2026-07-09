"""Export SASRecF top-100 recall with default config."""

from __future__ import annotations

import sys

from src.recall.sasrec_recall import main


def _ensure_default_sasrecf_config() -> None:
    """Inject sasrecf config when user does not pass --config."""
    if "--config" in sys.argv:
        return
    sys.argv.extend(["--config", "configs/sasrecf.yaml"])


if __name__ == "__main__":
    _ensure_default_sasrecf_config()
    main()
