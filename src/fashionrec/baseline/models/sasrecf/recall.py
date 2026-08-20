from __future__ import annotations

from fashionrec.baseline.settings import SASRECF_CONFIG
from fashionrec.baseline.models.sasrecf.recall_service import main as run_recall
from fashionrec.shared.runtime.argv import force_option, normalized_argv


def main(argv: list[str] | None = None) -> None:
    run_recall(force_option(normalized_argv(argv), "--config", str(SASRECF_CONFIG)))
