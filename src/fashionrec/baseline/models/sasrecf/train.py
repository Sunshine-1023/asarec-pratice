from __future__ import annotations

from fashionrec.baseline.settings import SASRECF_CONFIG
from fashionrec.shared.runtime.argv import force_option, normalized_argv
from fashionrec.baseline.models.sasrecf.training_service import main as run_training


def main(argv: list[str] | None = None) -> None:
    run_training(force_option(normalized_argv(argv), "--config", str(SASRECF_CONFIG)))
