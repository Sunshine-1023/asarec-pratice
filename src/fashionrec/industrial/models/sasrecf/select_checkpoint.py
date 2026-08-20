from __future__ import annotations

from fashionrec.industrial.settings import SASRECF_CONFIG
from fashionrec.shared.runtime.argv import force_option, normalized_argv
from fashionrec.industrial.models.sasrecf.checkpoint_service import main as run_selection


def main(argv: list[str] | None = None) -> None:
    run_selection(force_option(normalized_argv(argv), "--config", str(SASRECF_CONFIG)))
